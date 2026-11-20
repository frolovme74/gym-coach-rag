"""Повторная оценка неизменного поиска после полного аудита двух top-10."""

import argparse
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from benchmark.holdout import (
    HOLDOUT,
    check_branch,
    file_hash,
    inputs,
    read_json,
    verify_frozen,
    write_new_json,
)
from benchmark.run import ROOT, TOPICS, evaluate, require, validate_dataset
from benchmark.tune import compare_metrics

V2 = ROOT / "benchmark/holdout-v2"
FIRST_REPORT = ROOT / "benchmark/results/2026-11-20-holdout.json"
ROLE = "holdout_v2_post_retrieval_pool_audit"
EXTRA_METRICS = ("precision_at_5", "precision_at_10", "hit_at_5", "hit_at_10")
LOCK_PATHS = (
    "benchmark/holdout-v2/queries.json",
    "benchmark/holdout-v2/judgments.json",
    "benchmark/holdout-v2/pool.json",
    "benchmark/holdout-v2/annotation-review.json",
    "benchmark/audit_pool.py",
    "benchmark/holdout/frozen.json",
    "benchmark/results/2026-11-20-holdout.json",
    "benchmark/results/2026-11-20-holdout-independent-review.json",
)


def pool_from_report(report):
    """Объединение ранее неразмеченных пар; профили участвуют симметрично."""
    pairs = {}
    for name in ("baseline", "selected"):
        for row in report[name]["per_query"]:
            if row["expected"] != "answer":
                continue
            for hit in row["top_10"]:
                if hit["judgment"] == "unjudged":
                    pairs.setdefault((row["id"], hit["doc_id"]), []).append(name)
    return pairs


def validate_extension(
    old_queries, old_annotations, queries, annotations, pool, report
):
    require(annotations.get("dataset_role") == ROLE, "Неверная роль v2")
    require(annotations.get("annotation_version") == 2, "Нужна версия разметки 2")
    require(
        annotations.get("ranking_seen_before_freeze") is True, "Выдача уже известна"
    )
    require(annotations.get("qrels_exhaustive") is False, "Qrels не исчерпывающие")
    require(len(queries) == len(old_queries), "Изменён состав вопросов")
    old_pairs = {}
    for before, after in zip(old_queries, queries):
        require(
            {k: v for k, v in before.items() if k != "relevant"}
            == {k: v for k, v in after.items() if k != "relevant"},
            "Изменён вопрос или порядок вопросов",
        )
        require(set(before["relevant"]) <= set(after["relevant"]), "Удалён positive v1")
        qid = before["id"]
        previous = old_annotations["queries"][qid]
        current = annotations["queries"][qid]
        if before["expected"] == "refuse":
            require(current == previous and after == before, "Изменён запрос на отказ")
        require(
            set(previous["candidate_ids"]) <= set(current["candidate_ids"]),
            "Удалены кандидаты v1",
        )
        current_map = {j["doc_id"]: j for j in current["judgments"]}
        for judgment in previous["judgments"]:
            did = judgment["doc_id"]
            old_pairs[qid, did] = judgment
            require(current_map.get(did) == judgment, "Изменено решение v1")
    expected = pool_from_report(report)
    actual = {(r["query_id"], r["doc_id"]): r for r in pool}
    require(len(actual) == len(pool), "Повтор пары в pool")
    require(actual.keys() == expected.keys(), "Неполный или лишний pool")
    require(not actual.keys() & old_pairs.keys(), "Pool содержит разметку v1")
    new_pairs = {
        (qid, j["doc_id"]): j
        for qid, review in annotations["queries"].items()
        for j in review["judgments"]
        if (qid, j["doc_id"]) not in old_pairs
    }
    require(new_pairs.keys() == expected.keys(), "Неполные или лишние решения v2")
    for pair, row in actual.items():
        require(
            sorted(row["profiles"]) == expected[pair], "Изменено происхождение пары"
        )
        require(
            row.get("full_text_read") is True, "Не подтверждено чтение полного чанка"
        )
        for key in ("doc_id", "relevant", "evidence", "reason", "text_sha256"):
            require(row[key] == new_pairs[pair][key], "Pool расходится с judgments")
    return {
        "new_pairs": len(pool),
        "selected_unjudged": sum("selected" in r["profiles"] for r in pool),
        "baseline_unjudged": sum("baseline" in r["profiles"] for r in pool),
        "baseline_only_additional": sum(r["profiles"] == ["baseline"] for r in pool),
        "new_relevant": sum(r["relevant"] for r in pool),
        "new_non_relevant": sum(not r["relevant"] for r in pool),
        "preserved_v1_judgments": len(old_pairs),
    }


def verify_ranking(before, after):
    require(len(before["per_query"]) == len(after["per_query"]), "Изменено число выдач")
    for old, new in zip(before["per_query"], after["per_query"]):
        for key in (
            "id",
            "query",
            "topic",
            "expected",
            "retrieved_count",
            "max_score",
            "score_only_refusal",
        ):
            require(old[key] == new[key], f"Изменена выдача: {key}")
        require(
            [(h["doc_id"], h["score"]) for h in old["top_10"]]
            == [(h["doc_id"], h["score"]) for h in new["top_10"]],
            "Изменены top-10 или scores",
        )
        require(
            all(
                new["relevant_positions"].get(did) == rank
                for did, rank in old["relevant_positions"].items()
            ),
            "Изменились позиции старых positives",
        )


def augment_metrics(result, require_complete=True):
    rows = [r for r in result["per_query"] if r["expected"] == "answer"]
    coverage = Counter(h["judgment"] for r in rows for h in r["top_10"])
    if require_complete:
        require(coverage["unjudged"] == 0, "Остались unjudged в top-10")
    for row in rows:
        for k in (5, 10):
            hits = sum(h["judgment"] == "relevant" for h in row["top_10"][:k])
            row["metrics"][f"precision_at_{k}"] = hits / k
            row["metrics"][f"hit_at_{k}"] = float(hits > 0)
    for group, selected_rows in [(result["overall"], rows)] + [
        (result["by_topic"][topic], [r for r in rows if r["topic"] == topic])
        for topic in TOPICS
    ]:
        for key in EXTRA_METRICS:
            group[key] = math.fsum(r["metrics"][key] for r in selected_rows) / len(
                selected_rows
            )
    result["top_10_judgment_coverage"] = {
        "positions": sum(coverage.values()),
        **{
            label: coverage[label] for label in ("relevant", "non_relevant", "unjudged")
        },
    }


def load_inputs():
    verify_frozen(read_json(HOLDOUT / "frozen.json"))
    old_queries, old_annotations, profile, baseline, selected, _ = inputs()
    report = read_json(FIRST_REPORT)
    queries = read_json(V2 / "queries.json")
    annotations = read_json(V2 / "judgments.json")
    pool = read_json(V2 / "pool.json")
    documents = read_json(ROOT / "data/corpus/documents.json")
    counts = validate_dataset(queries, documents, annotations, baseline.corpus_sha256)
    audit = validate_extension(
        old_queries, old_annotations, queries, annotations, pool, report
    )
    review = read_json(V2 / "annotation-review.json")
    require(
        review.get("status") == "approved_before_v2_metrics", "Не завершена проверка"
    )
    return queries, annotations, profile, baseline, selected, report, counts, audit


def freeze():
    check_branch()
    require(not (V2 / "frozen.json").exists(), "v2 уже зафиксирована")
    *_, counts, audit = load_inputs()
    lock = {
        "version": 2,
        "dataset_role": ROLE,
        "ranking_seen": True,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "files_sha256": {p: file_hash(ROOT / p) for p in LOCK_PATHS},
        "dataset": counts,
        "audit": audit,
    }
    write_new_json(V2 / "frozen.json", lock)
    return lock


def verify_lock(lock, root=ROOT):
    require(
        lock.get("version") == 2 and lock.get("dataset_role") == ROLE,
        "Неверный lock v2",
    )
    require(set(lock.get("files_sha256", {})) == set(LOCK_PATHS), "Неполный lock v2")
    for path, expected in lock["files_sha256"].items():
        require(
            file_hash(root / path) == expected, f"Изменён фиксированный файл: {path}"
        )


def run(output):
    check_branch()
    protected = [ROOT / p for p in LOCK_PATHS] + [V2 / "frozen.json"]
    require(
        output.resolve() not in {p.resolve() for p in protected},
        "Выход совпадает с входом",
    )
    require(not output.exists(), "Отчёт не перезаписывается: задайте новый --output")
    lock = read_json(V2 / "frozen.json")
    verify_lock(lock)
    queries, annotations, profile, baseline, selected, first, counts, audit = (
        load_inputs()
    )
    results = {}
    for name, index, threshold in (
        ("baseline", baseline, 0.0),
        ("selected", selected, profile["refusal_threshold"]),
    ):
        result = evaluate(index, queries, threshold, annotations)
        verify_ranking(first[name], result)
        augment_metrics(result)
        result["config"] = first[name]["config"]
        result["refusal_diagnostics"] = first[name]["refusal_diagnostics"]
        results[name] = result
    report = {
        "version": 2,
        "dataset_role": ROLE,
        "status": "annotation_audit_not_retrieval_improvement",
        "run_at": datetime.now(timezone.utc).isoformat(),
        "frozen_file_sha256": file_hash(V2 / "frozen.json"),
        "protocol": lock,
        "dataset": counts,
        "audit": audit,
        "qrels_exhaustive": False,
        "rankings_scores_and_refusals_unchanged": True,
        **results,
        "delta_selected_minus_baseline": compare_metrics(
            results["selected"], results["baseline"]
        ),
        "delta_v2_minus_v1": {
            name: compare_metrics(results[name], first[name]) for name in results
        },
        "v1_metrics": {name: first[name]["overall"] for name in results},
        "limitations": [
            "Те же вопросы после просмотра выдачи: это не новый независимый holdout.",
            "Полностью размечены top-10 двух фиксированных профилей только для 24 предметных вопросов.",
            "За пределами объединённого пула qrels остаются неполными; Recall и MRR условны относительно разметки.",
            "Изменения v1→v2 отражают разметку, а не улучшение алгоритма.",
            "Два агента, не внешние предметные эксперты; спорные переводы корпуса остаются ограничением.",
            "Порог не подбирался повторно; score-only отказ не заменяет продуктовую политику.",
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
        default=ROOT / "benchmark/results/2026-11-20-holdout-v2.json",
    )
    args = parser.parse_args()
    result = freeze() if args.freeze else run(args.output)
    print(result["audit"])


if __name__ == "__main__":
    main()
