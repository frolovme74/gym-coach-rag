"""Защита исходного holdout и честного сравнения на расширенных qrels."""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmark.audit_pool import (
    FIRST_REPORT,
    HOLDOUT,
    TOPICS,
    V2,
    augment_metrics,
    pool_from_report,
    read_json,
    run,
    validate_extension,
    verify_lock,
    verify_ranking,
)


class PoolAuditTests(unittest.TestCase):
    def fixture(self):
        return [
            read_json(HOLDOUT / "queries.json"),
            read_json(HOLDOUT / "judgments.json"),
            read_json(V2 / "queries.json"),
            read_json(V2 / "judgments.json"),
            read_json(V2 / "pool.json"),
            read_json(FIRST_REPORT),
        ]

    def test_actual_symmetric_pool_is_complete(self):
        args = self.fixture()
        result = validate_extension(*args)
        self.assertEqual(result["new_pairs"], 213)
        self.assertEqual(result["selected_unjudged"], 173)
        self.assertEqual(result["baseline_only_additional"], 40)
        self.assertEqual(result["preserved_v1_judgments"], 153)
        self.assertEqual(len(pool_from_report(args[-1])), 213)

    def test_missing_extra_or_duplicate_pool_rejected(self):
        for change in ("missing", "extra", "duplicate"):
            args = self.fixture()
            if change == "missing":
                args[4].pop()
            else:
                row = copy.deepcopy(args[4][0])
                if change == "extra":
                    row["doc_id"] = "not_in_pool"
                args[4].append(row)
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_extension(*args)

    def test_query_or_refusal_changes_rejected(self):
        for qindex in (0, -1):
            args = self.fixture()
            args[2][qindex]["query"] += " изменено"
            with (
                self.subTest(qindex=qindex),
                self.assertRaisesRegex(ValueError, "Изменён вопрос"),
            ):
                validate_extension(*args)

    def test_old_judgment_cannot_be_rewritten(self):
        args = self.fixture()
        qid = args[0][0]["id"]
        args[3]["queries"][qid]["judgments"][0]["reason"] = "переписано"
        with self.assertRaisesRegex(ValueError, "Изменено решение v1"):
            validate_extension(*args)

    def test_pool_and_annotations_must_agree(self):
        args = self.fixture()
        args[4][0]["relevant"] = not args[4][0]["relevant"]
        with self.assertRaisesRegex(ValueError, "Pool расходится"):
            validate_extension(*args)

    def test_baseline_origin_and_disclosure_required(self):
        for field, value in (("profiles", ["selected"]), ("full_text_read", False)):
            args = self.fixture()
            row = next(r for r in args[4] if "baseline" in r["profiles"])
            row[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_extension(*args)
        args = self.fixture()
        args[3]["ranking_seen_before_freeze"] = False
        with self.assertRaisesRegex(ValueError, "Выдача уже известна"):
            validate_extension(*args)

    def test_ranking_scores_and_refusals_cannot_change(self):
        original = read_json(FIRST_REPORT)["selected"]
        verify_ranking(original, copy.deepcopy(original))
        for change in ("score", "order", "refusal", "old_position"):
            current = copy.deepcopy(original)
            row = current["per_query"][0]
            if change == "score":
                row["top_10"][0]["score"] += 0.1
            elif change == "order":
                row["top_10"].reverse()
            elif change == "refusal":
                row["score_only_refusal"] = not row["score_only_refusal"]
            else:
                row["relevant_positions"][next(iter(row["relevant_positions"]))] = -1
            with self.subTest(change=change), self.assertRaises(ValueError):
                verify_ranking(original, current)

    def test_precision_hit_and_unjudged_guard(self):
        result = {
            "overall": {},
            "by_topic": {t: {} for t in TOPICS},
            "per_query": [
                {
                    "expected": "answer",
                    "topic": topic,
                    "metrics": {},
                    "top_10": [
                        {"judgment": "relevant" if i in (1, 7) else "non_relevant"}
                        for i in range(1, 11)
                    ],
                }
                for topic in TOPICS
            ],
        }
        augment_metrics(result)
        self.assertEqual(result["overall"]["precision_at_5"], 0.2)
        self.assertEqual(result["overall"]["precision_at_10"], 0.2)
        self.assertEqual(result["overall"]["hit_at_5"], 1)
        self.assertEqual(result["top_10_judgment_coverage"]["non_relevant"], 32)
        result["per_query"][0]["top_10"][0]["judgment"] = "unjudged"
        with self.assertRaisesRegex(ValueError, "Остались unjudged"):
            augment_metrics(result)

    def test_lock_requires_complete_fingerprints(self):
        with self.assertRaisesRegex(ValueError, "Неполный lock"):
            verify_lock(
                {
                    "version": 2,
                    "dataset_role": "holdout_v2_post_retrieval_pool_audit",
                    "files_sha256": {},
                }
            )

    def test_existing_report_and_input_never_overwritten(self):
        with (
            patch("benchmark.audit_pool.check_branch"),
            patch("benchmark.audit_pool.evaluate") as search,
        ):
            with self.assertRaisesRegex(ValueError, "Выход совпадает"):
                run(FIRST_REPORT)
            with tempfile.TemporaryDirectory() as tmp:
                target = Path(tmp) / "report.json"
                target.write_text("original")
                with self.assertRaisesRegex(ValueError, "не перезаписывается"):
                    run(target)
                self.assertEqual(target.read_text(), "original")
            search.assert_not_called()


if __name__ == "__main__":
    unittest.main()
