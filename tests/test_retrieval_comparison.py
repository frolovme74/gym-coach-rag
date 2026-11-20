"""Выбор поискового варианта и неизменность меток сравнения."""

import copy
import unittest

from benchmark.compare_retrieval import VARIANTS, check_fixed_labels, choose_variant


class RetrievalComparisonTests(unittest.TestCase):
    def test_selection_priority_and_tie_are_deterministic(self):
        results = {
            name: {"overall": {"recall_at_5": 0.3, "recall_at_10": 0.5, "mrr": 0.6}}
            for name in VARIANTS
        }
        self.assertEqual(choose_variant(results), "plain")
        results["concepts"]["overall"]["mrr"] = 0.8
        self.assertEqual(choose_variant(results), "concepts")
        results["dedup"]["overall"]["recall_at_10"] = 0.6
        self.assertEqual(choose_variant(results), "dedup")
        results["concepts_dedup"]["overall"]["recall_at_5"] = 0.4
        self.assertEqual(choose_variant(results), "concepts_dedup")

    def test_evidence_may_change_but_labels_cannot(self):
        original = {
            "queries": {
                "q1": {
                    "judgments": [
                        {"doc_id": "a", "relevant": True, "evidence": "старое"}
                    ]
                }
            }
        }
        revised = copy.deepcopy(original)
        revised["queries"]["q1"]["judgments"][0]["evidence"] = "новое"
        check_fixed_labels(original, revised)
        revised["queries"]["q1"]["judgments"][0]["relevant"] = False
        with self.assertRaisesRegex(ValueError, "булевы метки"):
            check_fixed_labels(original, revised)


if __name__ == "__main__":
    unittest.main()
