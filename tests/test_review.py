"""Бланк различает top-3, контекст и независимую смысловую оценку."""

import json
import tempfile
import unittest
from pathlib import Path
from shutil import copytree

from app.rag.bm25 import SearchResult
from app.rag.pipeline import ROOT
from benchmark.review import document_position, load_suite


class ReviewTests(unittest.TestCase):
    def test_position_does_not_treat_rank_four_as_top_three(self):
        hits = [SearchResult(str(i), 10 - i) for i in range(1, 6)]
        for relevant, label, rank in [
            (["1"], "1", 1),
            (["3"], "2–3", 3),
            (["4"], "нет", 4),
            (["absent"], "нет", None),
        ]:
            position = document_position(hits, relevant)
            self.assertEqual((position["label"], position["rank"]), (label, rank))

    def test_frozen_balanced_suite_and_exact_evidence(self):
        documents = {
            d["doc_id"]: d
            for d in json.loads((ROOT / "data/corpus/documents.json").read_text())
        }
        suite, _ = load_suite(ROOT / "benchmark/review-15", documents)
        self.assertEqual(15, len(suite))
        self.assertEqual(3, sum(q["expected"] == "refuse" for q in suite))
        for topic in ("training", "nutrition", "supplements", "recovery"):
            self.assertEqual(3, sum(q["topic"] == topic for q in suite))
        public = json.loads((ROOT / "benchmark/review-15/questions.json").read_text())
        self.assertTrue(all(set(q) == {"id", "query"} for q in public))
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "suite"
            copytree(ROOT / "benchmark/review-15", folder)
            (folder / "questions.json").write_text("[]")
            with self.assertRaises(ValueError):
                load_suite(folder, documents)


if __name__ == "__main__":
    unittest.main()
