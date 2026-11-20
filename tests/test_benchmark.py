"""Проверка метрик на ручных примерах, разметки и изоляции запросов на отказ."""

import copy
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.rag.bm25 import BM25Index, SearchResult
from benchmark.run import (
    TOPICS,
    REFUSAL_CATEGORIES,
    evaluate,
    json_hash,
    macro_average,
    retrieval_metrics,
    run,
    text_hash,
    validate_dataset,
)


def fixture():
    docs = [
        {"doc_id": f"d{i}", "text": "общий подтверждённый фрагмент"} for i in range(3)
    ]
    index = BM25Index.from_documents(docs, lemmatize=False)
    queries = []
    for i, topic in enumerate(TOPICS):
        queries.append(
            {
                "id": f"q{i}",
                "topic": topic,
                "query": f"общий {i}",
                "expected": "answer",
                "relevant": [d["doc_id"] for d in docs],
            }
        )
    for i, category in enumerate(REFUSAL_CATEGORIES):
        queries.append(
            {
                "id": f"r{i}",
                "topic": None,
                "query": f"unknownword {i}",
                "expected": "refuse",
                "relevant": [],
                "refusal_category": category,
                "reason": "Ответ вне задач корпуса",
            }
        )
    reviews = {
        q["id"]: {
            "candidate_ids": [d["doc_id"] for d in docs],
            "judgments": [
                {
                    "doc_id": d["doc_id"],
                    "relevant": True,
                    "evidence": d["text"],
                    "reason": "Содержит ответ",
                    "text_sha256": text_hash(d["text"]),
                }
                for d in docs
            ]
            if q["expected"] == "answer"
            else [],
        }
        for q in queries
    }
    annotations = {
        "version": 1,
        "corpus_sha256": index.corpus_sha256,
        "queries_sha256": json_hash(queries),
        "queries": reviews,
    }
    return docs, index, queries, annotations


class BenchmarkTest(unittest.TestCase):
    def test_recall_cutoffs_and_first_relevant_rank(self):
        ranking = ["x1", "x2", "x3", "x4", "a", "x6", "x7", "b", "x9", "x10", "c"]
        self.assertEqual(
            retrieval_metrics(ranking, ["a", "b", "c"]),
            {"recall_at_5": 1 / 3, "recall_at_10": 2 / 3, "mrr": 1 / 5},
        )

    def test_mrr_is_not_truncated_at_ten(self):
        result = retrieval_metrics(
            [f"x{i}" for i in range(10)] + ["a"], ["a", "b", "c"]
        )
        self.assertEqual(result, {"recall_at_5": 0, "recall_at_10": 0, "mrr": 1 / 11})

    def test_empty_retrieval_is_zero_but_empty_gold_is_invalid(self):
        self.assertEqual(
            retrieval_metrics([], ["a"]),
            {"recall_at_5": 0, "recall_at_10": 0, "mrr": 0},
        )
        with self.assertRaises(ValueError):
            retrieval_metrics(["a"], [])
        with self.assertRaises(ValueError):
            retrieval_metrics(["a", "a"], ["a"])

    def test_macro_average_is_per_query_not_pooled(self):
        rows = [
            {"metrics": retrieval_metrics(["a"], ["a"])},
            {"metrics": retrieval_metrics([], [str(i) for i in range(100)])},
        ]
        self.assertEqual(
            macro_average(rows),
            {"queries": 2, "recall_at_5": 0.5, "recall_at_10": 0.5, "mrr": 0.5},
        )

    def test_refusals_are_separate_and_threshold_does_not_change_retrieval(self):
        _, index, queries, annotations = fixture()
        baseline = evaluate(index, queries, annotations=annotations)
        higher = evaluate(index, queries, 1000, annotations)
        self.assertEqual(baseline["overall"]["queries"], 4)
        self.assertEqual(baseline["overall"]["mrr"], 1)
        self.assertEqual(baseline["overall"], higher["overall"])
        self.assertEqual(
            baseline["refusal_diagnostics"]["expected_refusals"]["rate"], 1
        )
        self.assertEqual(
            baseline["refusal_diagnostics"]["false_refusals_on_answerable"]["rate"], 0
        )
        self.assertEqual(
            higher["refusal_diagnostics"]["false_refusals_on_answerable"]["rate"], 1
        )
        self.assertEqual(set(baseline["by_topic"]), set(TOPICS))
        self.assertTrue(all(r["queries"] == 1 for r in baseline["by_topic"].values()))

    def test_threshold_equality_refuses_and_positive_lexical_match_can_fail_policy(
        self,
    ):
        _, index, queries, _ = fixture()
        queries[-1]["query"] = "общий"
        score = index.search("общий")[0].score
        low = evaluate(index, queries, 0)
        boundary = evaluate(index, queries, score)
        self.assertFalse(low["per_query"][-1]["score_only_refusal"])
        self.assertTrue(boundary["per_query"][-1]["score_only_refusal"])

    def test_runner_requests_full_ranking_and_marks_unjudged(self):
        class FakeIndex:
            n = 20

            def search(self, text, k):
                assert k == self.n
                return [SearchResult(f"x{i}", 20 - i) for i in range(10)] + [
                    SearchResult("d0", 1)
                ]

        _, _, queries, annotations = fixture()
        report = evaluate(FakeIndex(), queries, annotations=annotations)
        self.assertAlmostEqual(report["overall"]["mrr"], 1 / 11)
        self.assertEqual(report["per_query"][0]["top_10"][0]["judgment"], "unjudged")
        self.assertEqual(report["per_query"][0]["relevant_positions"]["d0"], 11)
        self.assertIsNone(report["per_query"][0]["relevant_positions"]["d1"])

    def test_threshold_validation(self):
        _, index, queries, _ = fixture()
        for value in [-1, math.inf, math.nan, True, "1"]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                evaluate(index, queries, value)

    def test_repository_queries_and_evidence_are_consistent(self):
        root = Path(__file__).resolve().parents[1]
        docs = json.loads((root / "data/corpus/documents.json").read_text())
        queries = json.loads((root / "benchmark/queries.json").read_text())
        annotations = json.loads((root / "benchmark/judgments.json").read_text())
        # Отпечаток не зависит от морфологии; тест не требует локального index.json.
        index = BM25Index.from_documents(docs, lemmatize=False)
        counts = validate_dataset(queries, docs, annotations, index.corpus_sha256)
        self.assertEqual(counts["by_topic"], {topic: 6 for topic in TOPICS})
        self.assertEqual(
            counts["refusals_by_category"],
            {category: 3 for category in REFUSAL_CATEGORIES},
        )

    def test_valid_balanced_dataset(self):
        docs, index, queries, annotations = fixture()
        counts = validate_dataset(queries, docs, annotations, index.corpus_sha256)
        self.assertEqual(counts["answerable"], 4)
        self.assertEqual(counts["refusals"], 4)

    def test_malformed_queries_rejected_even_if_hash_updated(self):
        docs, index, queries, annotations = fixture()
        mutations = [
            lambda q: q[0].update(relevant=["d0", "d0", "d1"]),
            lambda q: q[0].update(relevant=["d0", "d1", "missing"]),
            lambda q: q[0].update(relevant=["d0"]),
            lambda q: q[0].update(query=" "),
            lambda q: q[0].update(topic="nutrition"),
            lambda q: q[0].update(id=q[1]["id"]),
            lambda q: q[0].update(query=q[1]["query"]),
            lambda q: q[-1].update(relevant=["d0"]),
            lambda q: q[-1].update(refusal_category="other"),
            lambda q: q[-1].update(reason=""),
        ]
        for mutate in mutations:
            q, a = copy.deepcopy(queries), copy.deepcopy(annotations)
            mutate(q)
            a["queries_sha256"] = json_hash(q)
            with self.assertRaises(ValueError):
                validate_dataset(q, docs, a, index.corpus_sha256)

    def test_stale_or_unsubstantiated_annotation_rejected(self):
        docs, index, queries, annotations = fixture()
        mutations = [
            lambda a: a.update(corpus_sha256="changed"),
            lambda a: a.update(queries_sha256="changed"),
            lambda a: a.update(version=True),
            lambda a: a["queries"]["q0"]["judgments"][0].update(
                evidence="Нет такой цитаты"
            ),
            lambda a: a["queries"]["q0"]["judgments"][0].update(text_sha256="changed"),
            lambda a: a["queries"]["q0"]["judgments"][0].update(relevant=False),
            lambda a: a["queries"]["q0"]["judgments"][0].update(relevant=1),
            lambda a: a["queries"]["q0"]["judgments"].append(
                a["queries"]["q0"]["judgments"][0]
            ),
            lambda a: a["queries"].pop("r0"),
        ]
        for mutate in mutations:
            a = copy.deepcopy(annotations)
            mutate(a)
            with self.assertRaises(ValueError):
                validate_dataset(queries, docs, a, index.corpus_sha256)

    def test_disk_run_and_stale_index_and_output_protection(self):
        docs, index, queries, annotations = fixture()
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            paths = [
                base / name
                for name in [
                    "index.json",
                    "corpus.json",
                    "queries.json",
                    "judgments.json",
                    "out.json",
                ]
            ]
            index.save(paths[0])
            for path, value in zip(paths[1:4], [docs, queries, annotations]):
                path.write_text(json.dumps(value, ensure_ascii=False))
            with patch("sys.stdout", new=io.StringIO()):
                report = run(*paths)
            self.assertEqual(
                json.loads(paths[4].read_text())["overall"], report["overall"]
            )
            self.assertIn("queries_sha256", report["reproducibility"])
            with self.assertRaises(ValueError):
                run(*paths[:4], paths[1])
            docs[0]["text"] = "изменённый текст"
            paths[1].write_text(json.dumps(docs))
            with self.assertRaises(ValueError):
                run(*paths)


if __name__ == "__main__":
    unittest.main()
