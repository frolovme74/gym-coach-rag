"""Фиксированная сетка BM25 и калибровка score-порога на development-наборе."""

import argparse
import hashlib
import itertools
import json
import math
import platform
import subprocess
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from zoneinfo import ZoneInfo

from app.rag.bm25 import BM25Index
from app.rag.preprocess import preprocess
from benchmark.run import (
    METRICS,
    REFUSAL_CATEGORIES,
    ROOT,
    TOPICS,
    evaluate,
    json_hash,
    macro_average,
    require,
    retrieval_metrics,
    validate_dataset,
)

K1_VALUES = (0.5, 0.9, 1.2, 1.5, 2.0, 3.0)
B_VALUES = (0.0, 0.25, 0.5, 0.75, 1.0)
BASELINE = dict(lemmatize=True, remove_stopwords=True, k1=1.5, b=0.75)
MAX_FALSE_REFUSAL_RATE = 0.10


def configurations():
    return [
        dict(lemmatize=lemma, remove_stopwords=stop, k1=k1, b=b)
        for lemma, stop, k1, b in itertools.product(
            (True, False), (True, False), K1_VALUES, B_VALUES
        )
    ]


def config_id(config):
    return (
        f"L{int(config['lemmatize'])}_S{int(config['remove_stopwords'])}"
        f"_k{config['k1']:g}_b{config['b']:g}"
    )


def stratified_folds(queries, n_splits=3):
    require(type(n_splits) is int and n_splits >= 2, "Нужно не менее двух folds")
    ids = [q["id"] for q in queries]
    require(len(ids) == len(set(ids)), "Повтор query id")
    groups = [
        [q for q in queries if q["expected"] == "answer" and q["topic"] == topic]
        for topic in TOPICS
    ]
    groups += [
        [
            q
            for q in queries
            if q["expected"] == "refuse" and q["refusal_category"] == category
        ]
        for category in REFUSAL_CATEGORIES
    ]
    require(sum(map(len, groups)) == len(queries), "Не все запросы попали в страты")
    folds = [[] for _ in range(n_splits)]
    for group in groups:
        require(len(group) >= n_splits, "В страте меньше вопросов, чем folds")
        for ordinal, query in enumerate(sorted(group, key=lambda q: q["id"])):
            folds[ordinal % n_splits].append(query["id"])
    return [sorted(fold) for fold in folds]


def summary(rows, ids=None):
    selected = [
        r for r in rows if r["expected"] == "answer" and (ids is None or r["id"] in ids)
    ]
    return {
        "overall": macro_average(selected),
        "by_topic": {
            topic: macro_average([r for r in selected if r["topic"] == topic])
            for topic in TOPICS
        },
    }


def selection_key(metrics, config):
    return (
        *[round(metrics[m], 12) for m in METRICS],
        -abs(config["k1"] - 1.5),
        -abs(config["b"] - 0.75),
        int(config["lemmatize"]),
        int(config["remove_stopwords"]),
        -config["k1"],
        -config["b"],
    )


def choose_configuration(grid, training_ids):
    require(bool(grid), "Пустая сетка")
    return max(
        grid,
        key=lambda item: selection_key(
            summary(item["per_query"], training_ids)["overall"], item["config"]
        ),
    )


def threshold_stats(rows, threshold):
    require(
        isinstance(threshold, (float, int))
        and not isinstance(threshold, bool)
        and math.isfinite(threshold)
        and threshold >= 0,
        "Некорректный порог",
    )
    answerable = [r for r in rows if r["expected"] == "answer"]
    negatives = [r for r in rows if r["expected"] == "refuse"]
    require(bool(answerable) and bool(negatives), "Нужны обе группы для калибровки")
    for row in rows:
        score = row["max_score"]
        require(
            score is None
            or (
                isinstance(score, (int, float))
                and not isinstance(score, bool)
                and math.isfinite(score)
                and score >= 0
            ),
            "Некорректный max_score",
        )

    def refuse(row):
        return row["max_score"] is None or row["max_score"] <= threshold

    correct_refusals = sum(refuse(r) for r in negatives)
    false_refusals = sum(refuse(r) for r in answerable)
    true_rate = correct_refusals / len(negatives)
    false_rate = false_refusals / len(answerable)
    return {
        "threshold": threshold,
        "answerable": len(answerable),
        "refusals": len(negatives),
        "correct_refusals": correct_refusals,
        "missed_refusals": len(negatives) - correct_refusals,
        "false_refusals": false_refusals,
        "accepted_answerable": len(answerable) - false_refusals,
        "correct_refusal_rate": true_rate,
        "false_refusal_rate": false_rate,
        "balanced_accuracy": (true_rate + 1 - false_rate) / 2,
    }


def calibrate_threshold(rows, max_false_refusal_rate=MAX_FALSE_REFUSAL_RATE):
    require(
        isinstance(max_false_refusal_rate, (float, int))
        and not isinstance(max_false_refusal_rate, bool)
        and math.isfinite(max_false_refusal_rate)
        and 0 <= max_false_refusal_rate <= 1,
        "Некорректное ограничение отказов",
    )
    # Сначала валидация, чтобы NaN/неверные типы не попали в сортировку границ.
    threshold_stats(rows, 0.0)
    candidates = sorted(
        {0.0, *[r["max_score"] for r in rows if r["max_score"] is not None]}
    )
    curve = [threshold_stats(rows, threshold) for threshold in candidates]
    feasible = [
        point
        for point in curve
        if point["false_refusal_rate"] <= max_false_refusal_rate
    ]
    selected = (
        max(
            feasible,
            key=lambda p: (
                p["correct_refusals"],
                -p["false_refusals"],
                -p["threshold"],
            ),
        )
        if feasible
        else None
    )
    unconstrained = max(
        curve,
        key=lambda p: (p["balanced_accuracy"], -p["false_refusals"], -p["threshold"]),
    )
    return {
        "status": "selected" if selected is not None else "infeasible",
        "max_false_refusal_rate": max_false_refusal_rate,
        "selected": selected,
        "unconstrained_balanced_accuracy_optimum": unconstrained,
        "curve": curve,
    }


def calculate_grid(documents, queries, configs=None):
    configs = configurations() if configs is None else configs
    grid = []
    for lemma, stop in dict.fromkeys(
        (c["lemmatize"], c["remove_stopwords"]) for c in configs
    ):
        tokenized = {
            d["doc_id"]: preprocess(d["text"], lemmatize=lemma, remove_stopwords=stop)
            for d in documents
        }
        query_tokens = {
            q["id"]: preprocess(q["query"], lemmatize=lemma, remove_stopwords=stop)
            for q in queries
        }
        for config in configs:
            if (config["lemmatize"], config["remove_stopwords"]) != (lemma, stop):
                continue
            index = BM25Index(tokenized, **config)
            rows = []
            for q in queries:
                matches = index.search_tokens(query_tokens[q["id"]], k=index.n)
                row = {key: q[key] for key in ("id", "topic", "expected")}
                row.update(
                    max_score=matches[0].score if matches else None,
                    retrieved_count=len(matches),
                )
                if q["expected"] == "answer":
                    row["metrics"] = retrieval_metrics(
                        [r.doc_id for r in matches], q["relevant"]
                    )
                else:
                    row["refusal_category"] = q["refusal_category"]
                rows.append(row)
            grid.append(
                {
                    "id": config_id(config),
                    "config": config,
                    "index_stats": {
                        "documents": index.n,
                        "terms": len(index.df),
                        "avgdl": index.avgdl,
                    },
                    **summary(rows),
                    "per_query": rows,
                }
            )
        print(f"Готов вариант lemmatize={lemma}, remove_stopwords={stop}", flush=True)
    return grid


def cross_validate(grid, queries, folds):
    all_ids = {q["id"] for q in queries}
    require(len(folds) >= 2 and all(folds), "Пустые folds")
    flat = [qid for fold in folds for qid in fold]
    require(
        len(flat) == len(set(flat)) and set(flat) == all_ids,
        "Folds должны покрывать каждый запрос ровно один раз",
    )
    records, out_of_fold = [], []
    for number, validation in enumerate(folds):
        validation_ids = set(validation)
        training_ids = all_ids - validation_ids
        winner = choose_configuration(grid, training_ids)
        train_rows = [r for r in winner["per_query"] if r["id"] in training_ids]
        calibration = calibrate_threshold(train_rows)
        val_rows = [dict(r) for r in winner["per_query"] if r["id"] in validation_ids]
        threshold = (
            calibration["selected"]["threshold"] if calibration["selected"] else None
        )
        for row in val_rows:
            row["fold"] = number
            row["config_id"] = winner["id"]
            row["threshold"] = threshold
            row["score_only_refusal"] = (
                row["max_score"] is None or row["max_score"] <= threshold
                if threshold is not None
                else None
            )
        records.append(
            {
                "fold": number,
                "training_ids": sorted(training_ids),
                "validation_ids": sorted(validation_ids),
                "selected_config": winner["config"],
                "training_metrics": summary(train_rows),
                "validation_metrics": summary(val_rows),
                "threshold_calibration_train_only": calibration,
                "threshold_validation": threshold_stats(val_rows, threshold)
                if threshold is not None
                else None,
            }
        )
        out_of_fold.extend(val_rows)

    def refusals(rows):
        require(bool(rows), "Пустая группа out-of-fold")
        if any(r["score_only_refusal"] is None for r in rows):
            return {"status": "infeasible_in_some_folds"}
        count = sum(r["score_only_refusal"] for r in rows)
        return {
            "queries": len(rows),
            "score_only_refusals": count,
            "rate": count / len(rows),
        }

    positives = [r for r in out_of_fold if r["expected"] == "answer"]
    negatives = [r for r in out_of_fold if r["expected"] == "refuse"]
    return {
        "role": "diagnostic_cross_validation_within_known_development_set",
        "folds": records,
        **summary(out_of_fold),
        "refusal_diagnostics": {
            "expected_refusals": refusals(negatives),
            "false_refusals_on_answerable": refusals(positives),
            "by_category": {
                c: refusals([r for r in negatives if r["refusal_category"] == c])
                for c in REFUSAL_CATEGORIES
            },
        },
        "per_query": sorted(out_of_fold, key=lambda r: r["id"]),
    }


def compare_metrics(current, baseline):
    return {
        group: {m: current[group][m] - baseline[group][m] for m in METRICS}
        for group in ("overall",)
    } | {
        "by_topic": {
            topic: {
                m: current["by_topic"][topic][m] - baseline["by_topic"][topic][m]
                for m in METRICS
            }
            for topic in TOPICS
        }
    }


def verify_baseline(current, archived, queries, annotations, corpus_hash):
    provenance = archived["reproducibility"]
    for name, expected in [
        ("corpus_sha256", corpus_hash),
        ("queries_sha256", json_hash(queries)),
        ("judgments_sha256", json_hash(annotations)),
    ]:
        require(
            provenance[name] == expected,
            f"Данные отличаются от первого прогона: {name}",
        )
    for config_key, value in BASELINE.items():
        require(
            archived["index"][config_key] == value,
            "Параметры архивного baseline отличаются",
        )
    comparisons = [("overall", current["overall"], archived["overall"])]
    comparisons += [
        (topic, current["by_topic"][topic], archived["by_topic"][topic])
        for topic in TOPICS
    ]
    for group, a, b in comparisons:
        require(
            all(math.isclose(a[m], b[m], rel_tol=0, abs_tol=1e-12) for m in METRICS),
            f"Не воспроизведены метрики baseline: {group}",
        )


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def run_tuning(output, profile_path, index_output):
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=ROOT, text=True
    ).strip()
    require(
        branch == "2-rag-benchmark-Murad",
        "Подбор выполняется только в ветке 2-rag-benchmark-Murad",
    )
    inputs = [
        ROOT / "data/corpus/documents.json",
        ROOT / "benchmark/queries.json",
        ROOT / "benchmark/judgments.json",
        ROOT / "benchmark/results/2026-11-20-baseline.json",
        ROOT / "data/index/bm25.json",
    ]
    outputs = [output, profile_path, index_output]
    require(
        len({p.resolve() for p in outputs}) == 3, "Выходные файлы должны различаться"
    )
    require(
        not {p.resolve() for p in outputs} & {p.resolve() for p in inputs},
        "Результат не должен перезаписывать исходные данные или baseline",
    )
    documents, queries, annotations, archived = [
        json.loads(p.read_text(encoding="utf-8")) for p in inputs[:4]
    ]
    baseline_index = BM25Index.from_documents(documents, **BASELINE)
    validate_dataset(queries, documents, annotations, baseline_index.corpus_sha256)
    folds = stratified_folds(queries)
    grid = calculate_grid(documents, queries)
    baseline = next(item for item in grid if item["config"] == BASELINE)
    verify_baseline(
        baseline, archived, queries, annotations, baseline_index.corpus_sha256
    )
    winner = choose_configuration(grid, {q["id"] for q in queries})
    calibration = calibrate_threshold(winner["per_query"])
    cv = cross_validate(grid, queries, folds)
    chosen_index = BM25Index.from_documents(documents, **winner["config"])
    require(
        calibration["selected"] is not None,
        "Не найден допустимый порог; профиль не публикуется",
    )
    threshold = calibration["selected"]["threshold"]
    chosen = evaluate(chosen_index, queries, threshold, annotations)
    chosen["refusal_diagnostics"].update(
        threshold_calibrated=True,
        calibration_scope="development_full_in_sample",
        production_policy=False,
    )
    provenance = {
        "python": platform.python_version(),
        "dependencies": {p: version(p) for p in ("pymorphy3", "pymorphy3-dicts-ru")},
        "corpus_sha256": baseline_index.corpus_sha256,
        "queries_sha256": json_hash(queries),
        "judgments_sha256": json_hash(annotations),
        "baseline_file_sha256": hashlib.sha256(inputs[3].read_bytes()).hexdigest(),
        "code_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                Path(__file__),
                ROOT / "benchmark/run.py",
                ROOT / "app/rag/bm25.py",
                ROOT / "app/rag/preprocess.py",
            ]
        },
    }
    report = {
        "version": 1,
        "status": "completed",
        "run_at": datetime.now(ZoneInfo("Europe/Moscow")).isoformat(),
        "dataset_role": "development_v1_not_independent_test",
        "qrels_exhaustive": False,
        "protocol": {
            "k1": K1_VALUES,
            "b": B_VALUES,
            "configurations": len(grid),
            "selection": ["recall_at_5", "recall_at_10", "mrr"],
            "tie_break": "rounded 12 digits, abs(k1-1.5), abs(b-.75), lemma, stop, lower k1, lower b",
            "max_false_refusal_rate": MAX_FALSE_REFUSAL_RATE,
            "fold_assignment": "sorted query ids within topic/refusal category; ordinal modulo 3",
        },
        "reproducibility": provenance,
        "baseline": {
            "config": BASELINE,
            **summary(baseline["per_query"]),
            "threshold_zero": threshold_stats(baseline["per_query"], 0),
        },
        "preprocessing_ablation_at_baseline_parameters": [
            {k: v for k, v in item.items() if k != "per_query"}
            for item in grid
            if item["config"]["k1"] == 1.5 and item["config"]["b"] == 0.75
        ],
        "selected": {
            "config": winner["config"],
            "scope": "full_development_in_sample",
            "threshold": threshold,
            "threshold_selection": calibration,
            "boundary_optimum": winner["config"]["k1"] in (K1_VALUES[0], K1_VALUES[-1])
            or winner["config"]["b"] in (B_VALUES[0], B_VALUES[-1]),
            **chosen,
        },
        "delta_vs_first_baseline": compare_metrics(chosen, baseline),
        "cross_validation": cv,
        "cross_validation_delta_vs_baseline": compare_metrics(cv, baseline),
        "grid": grid,
        "limitations": [
            "Подбор на известном development-наборе; рост full-set метрик не является оценкой на новом test set.",
            "3-fold проверка использует только train при выборе, но всё ещё относится к ранее изученному development-набору.",
            "Неполные qrels: непомеченные чанки считаются нерелевантными.",
            "Порог score не реализует политику запрещённых тем и не подтверждает безопасность ответов.",
            "Оптимум ограничен заранее заданной сеткой; граница сетки не доказывает глобальный оптимум.",
        ],
    }
    profile = {
        "version": 1,
        "status": "development_candidate_not_integrated",
        "retrieval": winner["config"],
        "refusal_threshold": threshold,
        "refusal_rule": "empty ranking or max_score <= threshold",
        "calibration_scope": "development_full_in_sample",
        "max_false_refusal_rate_constraint": MAX_FALSE_REFUSAL_RATE,
        "reproducibility": provenance,
        "index_path": str(index_output),
        "report_path": str(output),
        "not_a_safety_policy": True,
    }
    chosen_index.save(index_output)
    write_json(output, report)
    write_json(profile_path, profile)
    print(
        json.dumps(
            {
                "selected": winner["config"],
                "threshold": threshold,
                "overall": chosen["overall"],
                "delta": report["delta_vs_first_baseline"]["overall"],
                "refusal_selection": calibration["selected"],
                "cross_validation": cv["overall"],
                "report": str(output),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "benchmark/results/2026-11-20-tuning.json"
    )
    parser.add_argument(
        "--profile", type=Path, default=ROOT / "benchmark/selected_config.json"
    )
    parser.add_argument(
        "--index-output", type=Path, default=ROOT / "data/index/bm25.tuned.json"
    )
    args = parser.parse_args(argv)
    try:
        run_tuning(args.output, args.profile, args.index_output)
    except (OSError, ValueError, TypeError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
