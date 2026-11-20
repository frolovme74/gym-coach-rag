"""Сетка, выбор параметров, порог и изоляция отложенных запросов."""

import copy
import io
import math
import unittest
from unittest.mock import patch

from app.rag.bm25 import BM25Index
from benchmark.run import METRICS, REFUSAL_CATEGORIES, TOPICS, json_hash
from benchmark.tune import (
    BASELINE,
    calculate_grid,
    calibrate_threshold,
    choose_configuration,
    compare_metrics,
    configurations,
    cross_validate,
    selection_key,
    stratified_folds,
    summary,
    threshold_stats,
    verify_baseline,
)


def sample_queries():
    queries = []
    for topic in TOPICS:
        for i in range(6):
            queries.append(
                dict(
                    id=f"{topic}{i}",
                    topic=topic,
                    expected="answer",
                    query=f"текст {topic} {i}",
                    relevant=["d0", "d1", "d2"],
                )
            )
    for category in REFUSAL_CATEGORIES:
        for i in range(3):
            queries.append(
                dict(
                    id=f"{category}{i}",
                    topic=None,
                    expected="refuse",
                    query=f"нет {category} {i}",
                    relevant=[],
                    refusal_category=category,
                )
            )
    return queries


def score_rows(answerable, refusals):
    return [dict(expected="answer", max_score=s) for s in answerable] + [
        dict(expected="refuse", max_score=s) for s in refusals
    ]


def synthetic_grid(queries):
    first_fold = set(stratified_folds(queries)[0])
    grid = []
    for name, config in [("a", BASELINE), ("b", dict(BASELINE, k1=2.0))]:
        rows = []
        for q in queries:
            row = {k: q[k] for k in ("id", "topic", "expected")}
            row["max_score"] = 10 if q["expected"] == "answer" else 2
            if q["expected"] == "answer":
                score = (
                    (0.1 if q["id"] in first_fold else 0.9)
                    if name == "a"
                    else (1.0 if q["id"] in first_fold else 0.2)
                )
                row["metrics"] = dict.fromkeys(METRICS, score)
            else:
                row["refusal_category"] = q["refusal_category"]
            rows.append(row)
        grid.append(dict(id=name, config=config, per_query=rows))
    return grid


class TuningTest(unittest.TestCase):
    def test_grid_has_120_unique_configs_and_all_four_baseline_ablations(self):
        grid = configurations()
        self.assertEqual(len(grid), 120)
        self.assertEqual(len({json_hash(c) for c in grid}), 120)
        self.assertIn(BASELINE, grid)
        ablations = [c for c in grid if c["k1"] == 1.5 and c["b"] == 0.75]
        self.assertEqual(
            {(c["lemmatize"], c["remove_stopwords"]) for c in ablations},
            {(True, True), (True, False), (False, True), (False, False)},
        )

    def test_stratification_is_stable_disjoint_and_balanced(self):
        queries = sample_queries()
        folds = stratified_folds(queries)
        self.assertEqual(folds, stratified_folds(list(reversed(queries))))
        self.assertEqual(len(set(sum(folds, []))), 36)
        by_id = {q["id"]: q for q in queries}
        for fold in folds:
            self.assertEqual(len(fold), 12)
            rows = [by_id[qid] for qid in fold]
            for topic in TOPICS:
                self.assertEqual(sum(q["topic"] == topic for q in rows), 2)
            for category in REFUSAL_CATEGORIES:
                self.assertEqual(
                    sum(q.get("refusal_category") == category for q in rows), 1
                )
        with self.assertRaises(ValueError):
            stratified_folds(queries, 4)
        with self.assertRaises(ValueError):
            stratified_folds(queries + queries[:1])

    def test_lexicographic_selection_prefers_recall_before_mrr(self):
        better_recall = dict(recall_at_5=0.21, recall_at_10=0.3, mrr=0.5)
        better_mrr = dict(recall_at_5=0.20, recall_at_10=0.9, mrr=1.0)
        self.assertGreater(
            selection_key(better_recall, BASELINE), selection_key(better_mrr, BASELINE)
        )
        tied = dict.fromkeys(METRICS, 0.5)
        self.assertGreater(
            selection_key(tied, BASELINE), selection_key(tied, dict(BASELINE, k1=2))
        )
        self.assertGreater(
            selection_key(tied, BASELINE),
            selection_key(tied, dict(BASELINE, remove_stopwords=False)),
        )
        nearly = dict(tied, mrr=0.5 + 1e-14)
        self.assertEqual(selection_key(tied, BASELINE), selection_key(nearly, BASELINE))

    def test_threshold_equality_no_hits_and_confusion_counts(self):
        rows = score_rows([None, 3, 10], [None, 3, 8])
        stats = threshold_stats(rows, 3)
        self.assertEqual(stats["correct_refusals"], 2)
        self.assertEqual(stats["false_refusals"], 2)
        self.assertEqual(stats["accepted_answerable"], 1)
        self.assertEqual(stats["missed_refusals"], 1)
        self.assertAlmostEqual(stats["balanced_accuracy"], 0.5)

    def test_calibration_checks_all_boundaries_and_picks_smallest_tie(self):
        fit = calibrate_threshold(score_rows([10, 20], [1, 2, 3]), 0)
        self.assertEqual([p["threshold"] for p in fit["curve"]], [0, 1, 2, 3, 10, 20])
        self.assertEqual(fit["selected"]["threshold"], 3)
        self.assertEqual(fit["selected"]["false_refusals"], 0)
        self.assertEqual(fit["selected"]["correct_refusal_rate"], 1)

    def test_ten_percent_constraint_is_not_rounded_up(self):
        fit = calibrate_threshold(score_rows(list(range(1, 17)), [0.5, 1.5, 2.5]))
        self.assertEqual(fit["selected"]["threshold"], 1.5)
        self.assertEqual(fit["selected"]["false_refusals"], 1)
        self.assertEqual(fit["selected"]["correct_refusals"], 2)

    def test_infeasible_constraint_is_explicit_not_silently_relaxed(self):
        fit = calibrate_threshold(score_rows([None, 10], [1, 2]))
        self.assertEqual(fit["status"], "infeasible")
        self.assertIsNone(fit["selected"])
        self.assertTrue(all(p["false_refusal_rate"] > 0.1 for p in fit["curve"]))

    def test_unconstrained_balanced_accuracy_is_not_recommended_threshold(self):
        fit = calibrate_threshold(score_rows([1, 2, 3, 10], [4, 5, 6]), 0)
        self.assertEqual(fit["selected"]["threshold"], 0)
        self.assertEqual(fit["unconstrained_balanced_accuracy_optimum"]["threshold"], 6)
        self.assertEqual(
            fit["unconstrained_balanced_accuracy_optimum"]["false_refusals"], 3
        )

    def test_invalid_scores_rates_and_thresholds_rejected(self):
        for value in [math.nan, math.inf, -1, True, "2"]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                calibrate_threshold(score_rows([value], [1]))
            with self.assertRaises(ValueError):
                threshold_stats(score_rows([1], [2]), value)
        for rate in [-0.1, 1.1, math.nan, True, None]:
            with self.assertRaises(ValueError):
                calibrate_threshold(score_rows([1], [2]), rate)
        with self.assertRaises(ValueError):
            calibrate_threshold(score_rows([], [2]))

    def test_validation_values_do_not_influence_fold_selection_or_calibration(self):
        queries = sample_queries()
        grid = synthetic_grid(queries)
        folds = stratified_folds(queries)
        original = cross_validate(grid, queries, folds)
        first = original["folds"][0]
        self.assertEqual(first["selected_config"], BASELINE)
        changed = copy.deepcopy(grid)
        for item in changed:
            for row in item["per_query"]:
                if row["id"] in folds[0]:
                    row["max_score"] = 1000000 if row["expected"] == "refuse" else 0
                    if "metrics" in row:
                        row["metrics"] = dict.fromkeys(METRICS, 0)
        second = cross_validate(changed, queries, folds)["folds"][0]
        self.assertEqual(first["selected_config"], second["selected_config"])
        self.assertEqual(
            first["threshold_calibration_train_only"],
            second["threshold_calibration_train_only"],
        )
        self.assertNotEqual(
            first["threshold_validation"], second["threshold_validation"]
        )
        training = set(first["training_ids"])
        self.assertEqual(choose_configuration(grid, training)["id"], "a")

    def test_oof_covers_every_query_once_and_aggregates_per_query(self):
        queries = sample_queries()
        folds = stratified_folds(queries)
        report = cross_validate(synthetic_grid(queries), queries, folds)
        self.assertEqual(len(report["per_query"]), 36)
        self.assertEqual(len({r["id"] for r in report["per_query"]}), 36)
        self.assertEqual(report["overall"]["queries"], 24)
        expected = (
            math.fsum(
                r["metrics"]["mrr"]
                for r in report["per_query"]
                if r["expected"] == "answer"
            )
            / 24
        )
        self.assertAlmostEqual(report["overall"]["mrr"], expected)
        self.assertEqual(report["refusal_diagnostics"]["expected_refusals"]["rate"], 1)
        with self.assertRaises(ValueError):
            cross_validate(synthetic_grid(queries), queries, folds + [folds[0]])

    def test_cached_token_grid_matches_regular_index_preprocessing(self):
        docs = [
            dict(doc_id=f"d{i}", text=text)
            for i, text in enumerate(
                ["подходами и мышцами", "подход мышца мышца", "тренировки"]
            )
        ]
        queries = sample_queries()
        queries[0]["query"] = "И мышц подход"
        configs = [BASELINE, dict(BASELINE, lemmatize=False, remove_stopwords=False)]
        with patch("sys.stdout", new=io.StringIO()):
            grid = calculate_grid(docs, queries, configs)
        for item in grid:
            index = BM25Index.from_documents(docs, **item["config"])
            expected = index.search(queries[0]["query"])[0].score
            self.assertEqual(item["per_query"][0]["max_score"], expected)

    def test_metric_deltas_and_baseline_data_guard(self):
        queries = sample_queries()
        rows = synthetic_grid(queries)[0]["per_query"]
        current = summary(rows)
        self.assertTrue(
            all(x == 0 for x in compare_metrics(current, current)["overall"].values())
        )
        archived = {
            **current,
            "index": BASELINE,
            "reproducibility": {
                "corpus_sha256": "corpus",
                "queries_sha256": json_hash(queries),
                "judgments_sha256": json_hash({}),
            },
        }
        verify_baseline(current, archived, queries, {}, "corpus")
        with self.assertRaises(ValueError):
            verify_baseline(current, archived, queries, {}, "different")
        changed = copy.deepcopy(archived)
        changed["overall"]["mrr"] += 0.1
        with self.assertRaises(ValueError):
            verify_baseline(current, changed, queries, {}, "corpus")


if __name__ == "__main__":
    unittest.main()
