"""BM25: пример методички, реальные словоформы, границы и сохранность индекса."""

import copy
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.rag.bm25 import BM25Index

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = json.loads((ROOT / "tests/fixtures/bm25_methodical.json").read_text())


class BM25Test(unittest.TestCase):
    def setUp(self):
        self.index = BM25Index(FIXTURE["documents"])

    def test_methodical_index_tf_df_idf_and_lengths(self):
        index = self.index
        self.assertEqual(index.n, 3)
        self.assertEqual(index.doc_lengths, {"d1": 5, "d2": 6, "d3": 6})
        self.assertAlmostEqual(index.avgdl, 17 / 3)
        self.assertEqual(index.postings["граф"], (("d1", 1), ("d2", 1)))
        self.assertEqual(index.postings["матрица"], (("d2", 1), ("d3", 1)))
        self.assertEqual(index.postings["лапласиан"], (("d1", 1),))
        self.assertEqual(index.postings["обучение"], (("d3", 1),))
        self.assertEqual(index.df["граф"], 2)
        self.assertAlmostEqual(index.idf["граф"], 0.47000362924573563, places=14)
        self.assertAlmostEqual(index.idf["лапласиан"], 0.9808292530117263, places=14)

    def test_methodical_independently_calculated_scores_and_ranking(self):
        scores = self.index.scores(FIXTURE["query"])
        for doc_id, expected in FIXTURE["expected_scores"].items():
            self.assertAlmostEqual(scores[doc_id], expected, places=14)
        self.assertEqual(
            [r.doc_id for r in self.index.search_tokens(FIXTURE["query"])],
            FIXTURE["expected_ranking"],
        )

    def test_repeated_graph_in_original_is_not_lost(self):
        documents = copy.deepcopy(FIXTURE["documents"])
        documents["d1"].append("граф")
        index = BM25Index(documents)
        self.assertEqual(index.postings["граф"], (("d1", 2), ("d2", 1)))
        self.assertEqual(index.df["граф"], 2)
        self.assertEqual(index.avgdl, 6)
        expected = {
            "d1": 0.6714337560653366,
            "d2": 0.9400072584914712,
            "d3": 0.4700036292457356,
        }
        for doc_id, score in index.scores(FIXTURE["query"]).items():
            self.assertAlmostEqual(score, expected[doc_id], places=14)

    def test_real_preprocessing_of_original_methodical_sentences(self):
        sentences = [
            "Лапласиан графа используется для анализа структуры графа",
            "Матрица смежности графа описывает связи между вершинами",
            "Методы машинного обучения используют матрицы данных",
        ]
        index = BM25Index.from_documents(
            [{"doc_id": f"d{i}", "text": text} for i, text in enumerate(sentences, 1)]
        )
        self.assertEqual(index.doc_lengths, {"d1": 6, "d2": 7, "d3": 6})
        # «между» не является стоп-словом проекта: не подгоняем предобработку под PDF.
        results = index.search("Графы матрицы")
        self.assertEqual([r.doc_id for r in results], ["d2", "d1", "d3"])
        expected = {
            "d1": 0.682988065443134,
            "d2": 0.8974943674039173,
            "d3": 0.48140533453741113,
        }
        for result in results:
            self.assertAlmostEqual(result.score, expected[result.doc_id], places=14)

    def test_query_multiplicity_and_order(self):
        once = self.index.scores(["граф", "матрица"])
        twice = self.index.scores(["матрица", "граф", "граф", "матрица"])
        self.assertEqual(twice, {d: 2 * s for d, s in once.items()})
        self.assertEqual(once, self.index.scores(["матрица", "граф"]))

    def test_unknown_terms_do_not_add_candidates_or_change_scores(self):
        self.assertEqual(
            self.index.scores(["граф", "несуществующий"]), self.index.scores(["граф"])
        )
        self.assertNotIn("d3", self.index.scores(["граф"]))
        self.assertEqual(self.index.search_tokens(["несуществующий"]), [])

    def test_empty_corpus_documents_and_query(self):
        for docs in [{}, {"a": []}, {"a": [], "b": []}]:
            index = BM25Index(docs)
            self.assertEqual(index.n, len(docs))
            self.assertEqual(index.avgdl, 0)
            self.assertEqual(index.search_tokens(["слово"]), [])
        self.assertEqual(self.index.search_tokens([]), [])
        self.assertEqual(self.index.search("и на"), [])
        mixed = BM25Index({"empty": [], "a": ["слово", "слово"]})
        self.assertEqual(mixed.avgdl, 1)
        self.assertEqual(mixed.n, 2)
        self.assertEqual([r.doc_id for r in mixed.search_tokens(["слово"])], ["a"])

    def test_k_validation_and_limits(self):
        self.assertEqual(self.index.search_tokens(["граф"], 0), [])
        self.assertEqual(len(self.index.search_tokens(["граф"], 1)), 1)
        self.assertEqual(len(self.index.search_tokens(["граф"], 100)), 2)
        with self.assertRaises(ValueError):
            self.index.search_tokens([], -1)
        for k in [True, 1.5, "5", None]:
            with self.subTest(k=k), self.assertRaises(TypeError):
                self.index.search_tokens([], k)

    def test_ties_independent_of_document_order(self):
        first = BM25Index({"z": ["a"], "a": ["a"]})
        second = BM25Index({"a": ["a"], "z": ["a"]})
        self.assertEqual(first.search_tokens(["a"]), second.search_tokens(["a"]))
        self.assertEqual([r.doc_id for r in first.search_tokens(["a"])], ["a", "z"])

    def test_parameter_validation(self):
        for options in [
            {"k1": -1},
            {"b": -0.1},
            {"b": 1.1},
            {"k1": math.inf},
            {"b": math.nan},
            {"k1": -math.inf},
        ]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                BM25Index({}, **options)
        for options in [
            {"k1": True},
            {"b": "0.75"},
            {"lemmatize": 1},
            {"remove_stopwords": None},
        ]:
            with self.subTest(options=options), self.assertRaises(TypeError):
                BM25Index({}, **options)

    def test_tf_saturation_and_length_normalization(self):
        documents = {"short": ["a"], "long": ["a"] + ["b"] * 9}
        no_norm = BM25Index(documents, b=0).scores(["a"])
        self.assertEqual(no_norm["short"], no_norm["long"])
        for b in [0.75, 1]:
            scores = BM25Index(documents, b=b).scores(["a"])
            self.assertGreater(scores["short"], scores["long"])
        index = BM25Index({"one": ["a"], "many": ["a"] * 10}, b=0)
        scores = index.scores(["a"])
        self.assertGreater(scores["many"], scores["one"])
        self.assertLess(scores["many"], 10 * scores["one"])
        binary = BM25Index({"one": ["a"], "many": ["a"] * 10}, k1=0)
        self.assertEqual(binary.scores(["a"])["one"], binary.scores(["a"])["many"])

    def test_large_finite_k1_does_not_overflow(self):
        index = BM25Index({"d": ["x", "x"]}, k1=1e308)
        score = index.scores(["x"])["d"]
        self.assertTrue(math.isfinite(score))
        self.assertAlmostEqual(score, 2 * math.log(4 / 3), places=14)

    def test_common_term_has_positive_idf(self):
        index = BM25Index({"a": ["x"], "b": ["x"]})
        self.assertAlmostEqual(index.idf["x"], math.log(1.2), places=14)
        self.assertTrue(all(r.score > 0 for r in index.search_tokens(["x"])))

    def test_corpus_records_validation_and_no_metadata_indexing(self):
        with self.assertRaises(ValueError):
            BM25Index.from_documents([{"doc_id": "a", "text": "x"}] * 2)
        for docs in [[{"doc_id": "", "text": "x"}], [{"text": "x"}]]:
            with self.assertRaises(ValueError):
                BM25Index.from_documents(docs)
        for docs in [[{"doc_id": "a", "text": None}], {"a": "x"}, ["text"]]:
            with self.assertRaises(TypeError):
                BM25Index.from_documents(docs)
        index = BM25Index.from_documents(
            [{"doc_id": "a", "text": "кошка", "source": "собака", "topic": "training"}]
        )
        self.assertEqual(index.search("собака training"), [])
        self.assertEqual(index.search("кошками")[0].doc_id, "a")

    def test_tokens_validation_and_input_copy(self):
        for tokens in ["a", b"a", None]:
            with self.assertRaises(TypeError):
                BM25Index({"d": tokens})
        for tokens in [[""], ["two words"], [1], [None]]:
            with self.assertRaises(ValueError):
                self.index.scores(tokens)
        source = {"a": ["x"]}
        index = BM25Index(source)
        source["a"].append("y")
        self.assertNotIn("y", index.df)
        with self.assertRaises(TypeError):
            index.doc_lengths["a"] = 100

    def test_preprocessing_options_and_corpus_fingerprint(self):
        documents = [{"doc_id": "a", "text": "И подходами"}]
        index = BM25Index.from_documents(
            documents, lemmatize=False, remove_stopwords=False
        )
        self.assertEqual(index.search("подход"), [])
        self.assertEqual(index.search("И")[0].doc_id, "a")
        self.assertTrue(index.matches_documents(documents))
        self.assertTrue(
            index.matches_documents([dict(documents[0], source="metadata")])
        )
        self.assertFalse(index.matches_documents([dict(documents[0], text="И подход")]))
        self.assertFalse(index.matches_documents([dict(documents[0], doc_id="b")]))

    def test_save_load_preserves_scores_statistics_and_settings(self):
        documents = [
            {"doc_id": "b", "text": "И граф граф"},
            {"doc_id": "a", "text": "матрица"},
        ]
        index = BM25Index.from_documents(
            documents, k1=2, b=0.3, lemmatize=False, remove_stopwords=False
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested/index.json"
            index.save(path)
            # Загрузка не пересчитывает токены и не запускает морфологию.
            with patch("app.rag.bm25.preprocess", side_effect=AssertionError):
                loaded = BM25Index.load(path)
            self.assertEqual(
                loaded.search("И граф матрица"), index.search("И граф матрица")
            )
            for attr in [
                "n",
                "avgdl",
                "df",
                "idf",
                "postings",
                "doc_lengths",
                "k1",
                "b",
                "lemmatize",
                "remove_stopwords",
                "corpus_sha256",
            ]:
                self.assertEqual(getattr(loaded, attr), getattr(index, attr))
            self.assertTrue(loaded.matches_documents(list(reversed(documents))))
            original = path.read_bytes()
            loaded.save(path)
            self.assertEqual(path.read_bytes(), original)

    def test_empty_index_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index.json"
            BM25Index({"empty": []}).save(path)
            loaded = BM25Index.load(path)
            self.assertEqual(loaded.n, 1)
            self.assertEqual(loaded.search_tokens(["anything"]), [])

    def test_corrupted_index_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index.json"
            self.index.save(path)
            original = json.loads(path.read_text())
            corruptions = [
                ("version", 99),
                ("version", True),
                ("parameters", {"k1": 1}),
                ("preprocess", {}),
                ("corpus_sha256", "bad"),
                ("doc_lengths", {"d1": -1}),
                ("doc_lengths", {"d1": True}),
                ("postings", {"граф": [["unknown", 1]]}),
                ("postings", {"граф": [["d1", 1], ["d1", 1]]}),
                ("postings", {"граф": [["d1", 0]]}),
                ("postings", {"граф": [["d1", True]]}),
                ("postings", {"граф": [["d1", 1.5]]}),
                ("postings", {"граф": [["d1", 1]]}),
                ("postings", {"граф": []}),
                ("postings", {"граф": ["d1"]}),
            ]
            for field, value in corruptions:
                with self.subTest(field=field, value=value):
                    data = copy.deepcopy(original)
                    data[field] = value
                    path.write_text(json.dumps(data))
                    with self.assertRaises((TypeError, ValueError)):
                        BM25Index.load(path)
            path.write_text("[]")
            with self.assertRaises(ValueError):
                BM25Index.load(path)
            path.write_text("{broken")
            with self.assertRaises(ValueError):
                BM25Index.load(path)

    def test_atomic_save_keeps_previous_index_when_replace_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index.json"
            path.write_text("previous")
            with patch("app.rag.bm25.os.replace", side_effect=OSError("test")):
                with self.assertRaises(OSError):
                    self.index.save(path)
            self.assertEqual(path.read_text(), "previous")
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_build_command_and_corpus_overwrite_protection(self):
        with tempfile.TemporaryDirectory() as directory:
            corpus, target = (
                Path(directory) / "corpus.json",
                Path(directory) / "index.json",
            )
            corpus.write_text(json.dumps([{"doc_id": "a", "text": "граф граф"}]))
            command = [
                sys.executable,
                "-m",
                "app.rag.build_index",
                "--corpus",
                str(corpus),
                "--output",
                str(target),
                "--no-lemmatize",
            ]
            result = subprocess.run(
                command, cwd=ROOT, text=True, capture_output=True, check=True
            )
            self.assertEqual(json.loads(result.stdout)["documents"], 1)
            self.assertEqual(BM25Index.load(target).search("граф")[0].doc_id, "a")
            original = corpus.read_bytes()
            result = subprocess.run(
                command + ["--output", str(corpus)],
                cwd=ROOT,
                text=True,
                capture_output=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(corpus.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
