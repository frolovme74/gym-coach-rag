"""Защита истории разметки и разделения калибровки/проверки."""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmark.audit_v3 import (
    FIRST,
    V3,
    V31,
    cross_validate_threshold,
    read_json,
    run,
    validate_revision,
    verify_lock,
)
from benchmark.run import ROOT


class V3AuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = read_json(FIRST)
        cls.old_queries = read_json(ROOT / "benchmark/holdout-v2/queries.json")
        cls.old_ann = read_json(V3 / "diagnostic-judgments.json")
        cls.queries = read_json(V31 / "queries.json")
        cls.ann = read_json(V31 / "judgments.json")
        cls.review = read_json(V31 / "annotation-review.json")

    def validate(self, ann=None, review=None):
        return validate_revision(
            self.old_queries,
            self.old_ann,
            self.queries,
            self.ann if ann is None else ann,
            self.review if review is None else review,
            self.report,
        )

    def test_exact_extension_and_one_reassessment(self):
        self.assertEqual(
            self.validate(),
            {
                "new_pairs": 8,
                "new_positive": 1,
                "new_negative": 7,
                "reassessed_pairs": 1,
            },
        )

    def test_unrelated_label_changes_rejected(self):
        ann = copy.deepcopy(self.ann)
        ann["queries"]["ht01"]["judgments"][0]["relevant"] ^= True
        with self.assertRaisesRegex(ValueError, "другое решение"):
            self.validate(ann=ann)

    def test_missing_audit_and_lost_history_rejected(self):
        review = copy.deepcopy(self.review)
        review["decisions"].pop(0)
        with self.assertRaisesRegex(ValueError, "Неполный/лишний"):
            self.validate(review=review)
        review = copy.deepcopy(self.review)
        review["decisions"][-1]["previous_judgment"]["relevant"] = True
        with self.assertRaisesRegex(ValueError, "Потеряна история"):
            self.validate(review=review)

    def test_validation_scores_do_not_select_own_fold_threshold(self):
        queries = read_json(ROOT / "benchmark/queries.json")
        rows = self.report["comparisons"]["development"]["combined"]["per_query"]
        before = cross_validate_threshold(queries, rows)
        changed = copy.deepcopy(rows)
        heldout = before["folds"][0]["validation_ids"]
        for row in changed:
            if row["id"] in heldout:
                row["max_score"] = 10000.0
        after = cross_validate_threshold(queries, changed)
        self.assertEqual(
            before["folds"][0]["calibration"], after["folds"][0]["calibration"]
        )
        ids = [r["id"] for r in before["out_of_fold"]["per_query"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {q["id"] for q in queries})

    def test_lock_requires_all_inputs_and_matching_hashes(self):
        with self.assertRaisesRegex(ValueError, "Неполный протокол"):
            verify_lock(
                {
                    "version": "3.1",
                    "role": "diagnostic_v3_1_post_retrieval_annotation_audit",
                    "files_sha256": {},
                }
            )

    def test_refuses_overwriting_either_output(self):
        with (
            tempfile.TemporaryDirectory() as temp,
            patch("benchmark.audit_v3.check_branch"),
        ):
            existing = Path(temp) / "kept.json"
            existing.write_text("original")
            missing = Path(temp) / "new.json"
            for output, profile in ((existing, missing), (missing, existing)):
                with self.assertRaisesRegex(ValueError, "не перезаписываются"):
                    run(output, profile)
            self.assertEqual(existing.read_text(), "original")
            self.assertFalse(missing.exists())
