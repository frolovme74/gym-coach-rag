"""Абляции перевода и поиска v3 на фиксированных qrels, без интеграции."""

import argparse
from datetime import datetime, timezone
from pathlib import Path

from app.rag.retrieval import ConceptBM25
from benchmark.audit_pool import augment_metrics, verify_lock
from benchmark.holdout import (
    check_branch,
    file_hash,
    read_json,
    verify_frozen,
    write_new_json,
)
from benchmark.run import ROOT, evaluate, require, validate_dataset
from benchmark.tune import compare_metrics

V3 = ROOT / "benchmark/retrieval-v3"
VARIANTS = {
    "plain": {"concepts": False, "deduplicate_query": False},
    "dedup": {"concepts": False, "deduplicate_query": True},
    "concepts": {"concepts": True, "deduplicate_query": False},
    "concepts_dedup": {"concepts": True, "deduplicate_query": True},
}
NEW_FILES = (
    "data/corpus-v3/documents.json",
    "data/corpus-v3/manifest.json",
    "data/corpus-v3/translation_patch.json",
    "data/corpus-v3/review.json",
    "benchmark/retrieval-v3/development-judgments.json",
    "benchmark/retrieval-v3/diagnostic-judgments.json",
    "benchmark/retrieval-v3/annotation-migrations.json",
    "benchmark/retrieval-v3/original-files.json",
    "app/rag/retrieval.py",
    "scripts/corpus_revision.py",
    "benchmark/compare_retrieval.py",
)


def choose_variant(results):
    # max сохраняет первый элемент при равенстве — более простой вариант.
    return max(
        VARIANTS,
        key=lambda name: tuple(
            results[name]["overall"][key]
            for key in ("recall_at_5", "recall_at_10", "mrr")
        ),
    )


def check_fixed_labels(before, after):
    require(before["queries"].keys() == after["queries"].keys(), "Изменён состав qrels")
    for qid in before["queries"]:
        old = {j["doc_id"]: j["relevant"] for j in before["queries"][qid]["judgments"]}
        new = {j["doc_id"]: j["relevant"] for j in after["queries"][qid]["judgments"]}
        require(old == new, f"Изменены булевы метки: {qid}")


def verify_originals():
    for path, expected in read_json(V3 / "original-files.json").items():
        require(file_hash(ROOT / path) == expected, f"Изменён исходный файл: {path}")
    verify_frozen(read_json(ROOT / "benchmark/holdout/frozen.json"))
    verify_lock(read_json(ROOT / "benchmark/holdout-v2/frozen.json"))


def load_inputs():
    verify_originals()
    old = read_json(ROOT / "data/corpus/documents.json")
    corrected = read_json(ROOT / "data/corpus-v3/documents.json")
    require(
        [d["doc_id"] for d in old] == [d["doc_id"] for d in corrected],
        "Изменены doc_id/порядок",
    )
    for a, b in zip(old, corrected):
        require(
            {k: v for k, v in a.items() if k not in ("text", "section")}
            == {k: v for k, v in b.items() if k not in ("text", "section")},
            "Изменены метаданные",
        )
    profile = read_json(ROOT / "benchmark/selected_config.json")
    datasets = {}
    for name, query_path, annotation_path in (
        ("development", "benchmark/queries.json", "benchmark/judgments.json"),
        (
            "diagnostic",
            "benchmark/holdout-v2/queries.json",
            "benchmark/holdout-v2/judgments.json",
        ),
    ):
        queries, original_ann = (
            read_json(ROOT / query_path),
            read_json(ROOT / annotation_path),
        )
        revised_ann = read_json(V3 / f"{name}-judgments.json")
        check_fixed_labels(original_ann, revised_ann)
        for docs, ann in ((old, original_ann), (corrected, revised_ann)):
            index = ConceptBM25.from_documents(
                docs, **profile["retrieval"], **VARIANTS["plain"]
            )
            validate_dataset(queries, docs, ann, index.corpus_sha256)
        datasets[name] = (queries, original_ann, revised_ann)
    return old, corrected, profile, datasets


def freeze():
    check_branch()
    require(not (V3 / "frozen.json").exists(), "v3 уже зафиксирована")
    load_inputs()
    review = read_json(ROOT / "data/corpus-v3/review.json")
    require(
        review["status"] == "approved_targeted_revision", "Переводы ещё не проверены"
    )
    result = {
        "version": 3,
        "role": "diagnostic_retrieval_experiment_after_error_analysis",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "variants": VARIANTS,
        "concept_weight": 3,
        "selection": "development Recall@5, Recall@10, MRR; ties simpler",
        "files_sha256": {p: file_hash(ROOT / p) for p in NEW_FILES},
    }
    write_new_json(V3 / "frozen.json", result)
    return result


def verify_experiment(lock):
    require(
        lock.get("concept_weight") == 3
        and lock.get("role") == "diagnostic_retrieval_experiment_after_error_analysis",
        "Изменён вес или назначение протокола v3",
    )
    require(
        lock.get("version") == 3 and lock.get("variants") == VARIANTS,
        "Неверный протокол v3",
    )
    require(set(lock.get("files_sha256", {})) == set(NEW_FILES), "Неполный протокол v3")
    for path, expected in lock["files_sha256"].items():
        require(file_hash(ROOT / path) == expected, f"Изменён файл v3: {path}")


def score(index, queries, annotations, threshold, *, current_scale=False):
    result = evaluate(index, queries, threshold, annotations)
    augment_metrics(result, require_complete=False)
    result["refusal_diagnostics"].update(
        threshold_calibrated=current_scale,
        calibration_scope=(
            "development_v2_original_scale"
            if current_scale
            else "legacy_v2_threshold_on_changed_score_scale_diagnostic_only"
        ),
        recalibrated_in_this_experiment=False,
    )
    result["precision_hit_scope"] = "known_qrels_only_unjudged_counted_as_not_relevant"
    return result


def run(output):
    check_branch()
    require(not output.exists(), "Отчёт не перезаписывается; задайте новый --output")
    lock = read_json(V3 / "frozen.json")
    verify_experiment(lock)
    old, corrected, profile, datasets = load_inputs()
    threshold = profile["refusal_threshold"]
    indexes = {}

    def index(corpus, variant):
        key = (corpus, variant)
        if key not in indexes:
            indexes[key] = ConceptBM25.from_documents(
                old if corpus == "original" else corrected,
                **profile["retrieval"],
                **VARIANTS[variant],
                concept_weight=3,
            )
        return indexes[key]

    dev_queries, dev_old, dev_new = datasets["development"]
    development = {
        name: score(index("corrected", name), dev_queries, dev_new, threshold)
        for name in VARIANTS
    }
    chosen = choose_variant(development)
    comparisons = {}
    for name, (queries, original_ann, new_ann) in datasets.items():
        cells = {}
        for corpus, variant, cell in (
            ("original", "plain", "current"),
            ("corrected", "plain", "translation_only"),
            ("original", chosen, "retrieval_only"),
            ("corrected", chosen, "combined"),
        ):
            cells[cell] = score(
                index(corpus, variant),
                queries,
                original_ann if corpus == "original" else new_ann,
                threshold,
                current_scale=(corpus == "original" and variant == "plain"),
            )
        comparisons[name] = {
            **cells,
            "delta_combined_minus_current": compare_metrics(
                cells["combined"], cells["current"]
            ),
            "delta_translation_only_minus_current": compare_metrics(
                cells["translation_only"], cells["current"]
            ),
            "delta_retrieval_only_minus_current": compare_metrics(
                cells["retrieval_only"], cells["current"]
            ),
        }
    first = read_json(ROOT / "benchmark/results/2026-11-20-holdout-v2.json")["selected"]
    current = comparisons["diagnostic"]["current"]
    for old_row, new_row in zip(first["per_query"], current["per_query"]):
        for key in (
            "id",
            "query",
            "top_10",
            "relevant_positions",
            "max_score",
            "score_only_refusal",
        ):
            require(
                old_row[key] == new_row[key], f"Текущая версия не воспроизвелась: {key}"
            )
    target_ids = ("ht02", "ht05", "test_re06")
    targets = {}
    for qid in target_ids:
        targets[qid] = {}
        for cell in ("current", "translation_only", "retrieval_only", "combined"):
            row = next(
                r
                for r in comparisons["diagnostic"][cell]["per_query"]
                if r["id"] == qid
            )
            ranks = [p for p in row["relevant_positions"].values() if p is not None]
            targets[qid][cell] = {
                "first_relevant_rank": min(ranks) if ranks else None,
                "metrics": row["metrics"],
                "top_10": row["top_10"],
            }
    report = {
        "version": 3,
        "role": lock["role"],
        "run_at": datetime.now(timezone.utc).isoformat(),
        "protocol": lock,
        "selected_variant": chosen,
        "selection_scope": "development_only; rule design informed by diagnostic errors",
        "legacy_threshold": threshold,
        "threshold_recalibrated": False,
        "development_variants": development,
        "comparisons": comparisons,
        "target_queries": targets,
        "qrels_exhaustive": False,
        "fixed_boolean_qrels": True,
        "annotation_migrations": read_json(V3 / "annotation-migrations.json"),
        "limitations": [
            "Это известные вопросы, а три промаха повлияли на дизайн словаря; независимое улучшение не доказано.",
            "Все варианты сравниваются на фиксированных булевых qrels. Изменение основания метки после перевода отмечено отдельно.",
            "Новая выдача содержит unjudged; метрики условны относительно старой разметки, полное покрытие нового top-10 не утверждается.",
            "Старый порог показан только как диагностика новой шкалы; требуется отдельная калибровка на development.",
            "Проверены выбранные переводы; весь корпус не вычитан. Интеграция и LLM не проверяются.",
        ],
    }
    write_new_json(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "benchmark/results/2026-11-20-retrieval-v3.json",
    )
    args = parser.parse_args()
    result = freeze() if args.freeze else run(args.output)
    print(result.get("selected_variant", "frozen"))


if __name__ == "__main__":
    main()
