"""Фиксация нового набора до поиска и оценка двух неизменных профилей."""

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from app.rag.bm25 import BM25Index
from benchmark.run import ROOT, evaluate, json_hash, require, validate_dataset
from benchmark.tune import BASELINE, compare_metrics

HOLDOUT = ROOT / "benchmark/holdout"
FROZEN_PATHS = (
    "benchmark/holdout/queries.json",
    "benchmark/holdout/judgments.json",
    "benchmark/holdout/annotation-review.json",
    "benchmark/queries.json",
    "benchmark/judgments.json",
    "benchmark/selected_config.json",
    "data/corpus/documents.json",
    "app/rag/preprocess.py",
    "app/rag/bm25.py",
    "benchmark/run.py",
    "benchmark/tune.py",
    "benchmark/holdout.py",
)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new_json(path, value):
    """Не перезаписывает первый протокол/результат даже при повторном запуске."""
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(encoded)


def check_branch():
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=ROOT, text=True
    ).strip()
    require(
        branch == "2-rag-benchmark-Murad",
        "Разрешена только ветка 2-rag-benchmark-Murad",
    )


def validate_holdout(queries, development, annotations, documents, corpus_hash):
    counts = validate_dataset(queries, documents, annotations, corpus_hash)
    require(
        counts["answerable"] == 24 and counts["refusals"] == 12,
        "Ожидается 24+12 вопросов",
    )
    require(
        set(counts["refusals_by_category"].values()) == {3},
        "Нужно по 3 отказа на категорию",
    )
    dev_ids = {q["id"] for q in development}
    normalize = lambda text: " ".join(text.casefold().split())
    dev_texts = {normalize(q["query"]) for q in development}
    require(not dev_ids & {q["id"] for q in queries}, "ID пересекаются с development")
    require(
        not dev_texts & {normalize(q["query"]) for q in queries},
        "Повтор вопроса development",
    )
    require(annotations.get("dataset_role") == "holdout_v1", "Неверная роль набора")
    require(
        annotations.get("ranking_seen_before_freeze") is False,
        "Разметка должна предшествовать выдаче",
    )
    require(bool(annotations.get("novelty_review")), "Нет проверки новых намерений")
    return counts


def inputs():
    queries = read_json(HOLDOUT / "queries.json")
    annotations = read_json(HOLDOUT / "judgments.json")
    documents = read_json(ROOT / "data/corpus/documents.json")
    profile = read_json(ROOT / "benchmark/selected_config.json")
    baseline = BM25Index.load(ROOT / "data/index/bm25.json")
    selected = BM25Index.load(ROOT / "data/index/bm25.tuned.json")
    for index, config in ((baseline, BASELINE), (selected, profile["retrieval"])):
        require(index.matches_documents(documents), "Индекс не соответствует корпусу")
        require(
            all(getattr(index, k) == v for k, v in config.items()),
            "Параметры индекса изменились",
        )
    require(
        profile["calibration_scope"] == "development_full_in_sample",
        "Порог должен быть из development",
    )
    counts = validate_holdout(
        queries,
        read_json(ROOT / "benchmark/queries.json"),
        annotations,
        documents,
        baseline.corpus_sha256,
    )
    return queries, annotations, profile, baseline, selected, counts


def freeze():
    check_branch()
    require(not (HOLDOUT / "frozen.json").exists(), "Набор уже зафиксирован")
    queries, annotations, profile, baseline, _, counts = inputs()
    review = read_json(HOLDOUT / "annotation-review.json")
    require(
        review.get("status") == "approved_before_ranking",
        "Не завершена взаимная проверка разметки",
    )
    lock = {
        "version": 1,
        "dataset_role": "holdout_v1",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "ranking_seen": False,
        "dataset": counts,
        "corpus_sha256": baseline.corpus_sha256,
        "queries_sha256": json_hash(queries),
        "judgments_sha256": json_hash(annotations),
        "baseline_config": BASELINE,
        "selected_config": profile["retrieval"],
        "baseline_threshold": 0.0,
        "selected_threshold": profile["refusal_threshold"],
        "threshold_origin": "development_full_in_sample_not_fit_on_holdout",
        "files_sha256": {p: file_hash(ROOT / p) for p in FROZEN_PATHS},
        "limitations": [
            "Новые вопросы относительно подбора, но тот же корпус и агентская разметка.",
            "Неполные qrels; это не внешний экспертный gold standard.",
            "После первого просмотра результатов набор не является скрытым.",
        ],
    }
    write_new_json(HOLDOUT / "frozen.json", lock)
    return lock


def verify_frozen(lock, root=ROOT):
    require(
        lock.get("version") == 1 and lock.get("dataset_role") == "holdout_v1",
        "Неверный протокол",
    )
    require(
        set(lock.get("files_sha256", {})) == set(FROZEN_PATHS),
        "Неполный список фиксированных файлов",
    )
    for path, expected in lock["files_sha256"].items():
        require(file_hash(root / path) == expected, f"После фиксации изменён {path}")
    profile = read_json(root / "benchmark/selected_config.json")
    require(
        lock["selected_config"] == profile["retrieval"], "Изменён профиль протокола"
    )
    require(
        lock["selected_threshold"] == profile["refusal_threshold"],
        "Изменён порог протокола",
    )
    require(
        lock["baseline_config"] == BASELINE and lock["baseline_threshold"] == 0,
        "Изменён baseline протокола",
    )


def run(output):
    check_branch()
    protected = {ROOT / p for p in FROZEN_PATHS} | {
        HOLDOUT / "frozen.json",
        ROOT / "data/index/bm25.json",
        ROOT / "data/index/bm25.tuned.json",
    }
    require(
        output.resolve() not in {p.resolve() for p in protected},
        "Выход совпадает с входным файлом",
    )
    require(
        not output.exists(),
        "Первый результат не перезаписывается; задайте другой --output",
    )
    lock = read_json(HOLDOUT / "frozen.json")
    verify_frozen(lock)
    queries, annotations, profile, baseline, selected, counts = inputs()
    require(
        json_hash(queries) == lock["queries_sha256"],
        "Запросы не совпадают с протоколом",
    )
    require(
        json_hash(annotations) == lock["judgments_sha256"],
        "Разметка не совпадает с протоколом",
    )
    results = {}
    for name, index, threshold in (
        ("baseline", baseline, lock["baseline_threshold"]),
        ("selected", selected, lock["selected_threshold"]),
    ):
        result = evaluate(index, queries, threshold, annotations)
        result["config"] = {k: getattr(index, k) for k in BASELINE}
        result["refusal_diagnostics"].update(
            threshold_calibrated=(name == "selected"),
            calibration_scope="development_only"
            if name == "selected"
            else "uncalibrated_baseline",
            fitted_on_holdout=False,
        )
        coverage = Counter(
            hit["judgment"]
            for row in result["per_query"]
            if row["expected"] == "answer"
            for hit in row["top_10"]
        )
        result["top_10_judgment_coverage"] = {
            "positions": sum(coverage.values()),
            **{
                label: coverage[label]
                for label in ("relevant", "non_relevant", "unjudged")
            },
        }
        results[name] = result
    report = {
        "version": 1,
        "status": "evaluated_not_quality_gate",
        "dataset_role": "holdout_v1",
        "run_at": datetime.now(timezone.utc).isoformat(),
        "frozen_at": lock["frozen_at"],
        "frozen_file_sha256": file_hash(HOLDOUT / "frozen.json"),
        "qrels_exhaustive": False,
        "dataset": counts,
        "protocol": lock,
        **results,
        "delta_selected_minus_baseline": compare_metrics(
            results["selected"], results["baseline"]
        ),
        "limitations": lock["limitations"]
        + [
            "Recall/MRR до порога; отказы — score-only диагностика, не политика приложения.",
            "Проверяется retrieval, не ответы LLM и не достоверность научных источников.",
            "Параметры и порог по этому набору не менялись.",
        ],
    }
    write_new_json(output, report)
    print(
        json.dumps(
            {
                name: {k: result[k] for k in ("overall", "refusal_diagnostics")}
                for name, result in results.items()
            },
            ensure_ascii=False,
        )
    )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--freeze", action="store_true", help="Зафиксировать данные без запуска поиска"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "benchmark/results/2026-11-20-holdout.json",
    )
    args = parser.parse_args(argv)
    try:
        if args.freeze:
            lock = freeze()
            print(f"Набор зафиксирован до поиска: {lock['frozen_at']}")
        else:
            run(args.output)
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
