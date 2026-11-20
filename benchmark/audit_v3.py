"""Аудит разметки v3.1 и отдельная калибровка порога фиксированного v3."""

import argparse
import copy
from datetime import datetime, timezone
from pathlib import Path

from app.rag.retrieval import ConceptBM25
from benchmark.audit_pool import augment_metrics, verify_ranking
from benchmark.compare_retrieval import verify_experiment, verify_originals
from benchmark.holdout import check_branch, file_hash, read_json, write_new_json
from benchmark.run import ROOT, evaluate, json_hash, require, validate_dataset
from benchmark.tune import (
    calibrate_threshold,
    compare_metrics,
    stratified_folds,
    threshold_stats,
)

V31 = ROOT / "benchmark/retrieval-v3.1"
V3 = ROOT / "benchmark/retrieval-v3"
FIRST = ROOT / "benchmark/results/2026-11-20-retrieval-v3.json"
ROLE = "diagnostic_v3_1_post_retrieval_annotation_audit"
DISPUTED = ("test_re02", "doc_re_006_008")
LOCK_FILES = (
    "benchmark/retrieval-v3.1/queries.json",
    "benchmark/retrieval-v3.1/judgments.json",
    "benchmark/retrieval-v3.1/annotation-review.json",
    "benchmark/retrieval-v3.1/original-files.json",
    "benchmark/audit_v3.py",
)


def expected_pool(report):
    return {
        (row["id"], hit["doc_id"])
        for row in report["comparisons"]["diagnostic"]["combined"]["per_query"]
        if row["expected"] == "answer"
        for hit in row["top_10"]
        if hit["judgment"] == "unjudged"
    }


def validate_revision(old_queries, old_ann, queries, ann, review, report):
    require(ann.get("dataset_role") == ROLE, "Неверная роль разметки")
    require(ann.get("ranking_seen_before_freeze") is True, "Выдача уже известна")
    require(ann.get("qrels_exhaustive") is False, "Qrels не исчерпывающие")
    require(len(queries) == len(old_queries), "Изменён состав вопросов")
    pool = expected_pool(report)
    require(len(pool) == 8, "Ожидалось 8 новых пар")
    decisions = review["decisions"]
    decision_map = {(d["query_id"], d["doc_id"]): d for d in decisions}
    require(len(decision_map) == len(decisions), "Дубли решений")
    require(decision_map.keys() == pool | {DISPUTED}, "Неполный/лишний аудит")
    added = set()
    for old_q, q in zip(old_queries, queries):
        qid = q["id"]
        require(
            {k: v for k, v in old_q.items() if k != "relevant"}
            == {k: v for k, v in q.items() if k != "relevant"},
            "Изменён вопрос или порядок",
        )
        before = old_ann["queries"][qid]
        after = ann["queries"][qid]
        old_map = {j["doc_id"]: j for j in before["judgments"]}
        new_map = {j["doc_id"]: j for j in after["judgments"]}
        require(old_map.keys() <= new_map.keys(), "Удалено решение")
        for did, judgment in old_map.items():
            if (qid, did) == DISPUTED:
                require(
                    judgment["relevant"] is False, "Исходная спорная метка не negative"
                )
                require(
                    new_map[did]["relevant"] is True, "Не пересмотрена спорная метка"
                )
                require(
                    new_map[did].get("semantic_status") == "reassessed",
                    "Нет статуса пересмотра",
                )
                require(
                    decision_map[DISPUTED]["previous_judgment"] == judgment,
                    "Потеряна история решения",
                )
            else:
                require(judgment == new_map[did], "Изменено другое решение")
        new_ids = new_map.keys() - old_map.keys()
        added.update((qid, did) for did in new_ids)
        require(
            set(after["candidate_ids"]) == set(before["candidate_ids"]) | new_ids,
            "Изменены кандидаты",
        )
        if q["expected"] == "refuse":
            require(q == old_q and after == before, "Изменён отказной вопрос")
        for did in new_ids | ({DISPUTED[1]} if qid == DISPUTED[0] else set()):
            d = decision_map[qid, did]
            require(
                d.get("full_text_read") is True and d.get("cross_reviewed") is True,
                "Нет проверки полного текста",
            )
            require(d.get("cross_review_reason"), "Нет основания перекрёстной проверки")
            for key in ("doc_id", "relevant", "evidence", "reason", "text_sha256"):
                require(new_map[did][key] == d[key], "Решение расходится с аудитом")
    require(added == pool, "Добавлены не те пары")
    return {
        "new_pairs": len(pool),
        "new_positive": sum(decision_map[p]["relevant"] for p in pool),
        "new_negative": sum(not decision_map[p]["relevant"] for p in pool),
        "reassessed_pairs": 1,
    }


def verify_parents():
    verify_originals()
    verify_experiment(read_json(V3 / "frozen.json"))
    for path, digest in read_json(V31 / "original-files.json").items():
        require(file_hash(ROOT / path) == digest, f"Изменён родительский файл: {path}")


def load_inputs():
    verify_parents()
    report = read_json(FIRST)
    profile = read_json(V3 / "selected_config.json")
    docs = read_json(ROOT / profile["corpus"])
    index = ConceptBM25.from_documents(docs, **profile["retrieval"])
    require(index.corpus_sha256 == profile["corpus_sha256"], "Изменён корпус профиля")
    old_q = read_json(ROOT / "benchmark/holdout-v2/queries.json")
    old_ann = read_json(V3 / "diagnostic-judgments.json")
    queries, ann = read_json(V31 / "queries.json"), read_json(V31 / "judgments.json")
    review = read_json(V31 / "annotation-review.json")
    audit = validate_revision(old_q, old_ann, queries, ann, review, report)
    validate_dataset(queries, docs, ann, index.corpus_sha256)
    dev_q = read_json(ROOT / "benchmark/queries.json")
    dev_ann = read_json(V3 / "development-judgments.json")
    validate_dataset(dev_q, docs, dev_ann, index.corpus_sha256)
    return report, profile, index, queries, ann, dev_q, dev_ann, audit


def freeze():
    check_branch()
    load_inputs()
    lock = {
        "version": "3.1",
        "role": ROLE,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "threshold_selection": "development only; correct refusals max, false <=10%, ties fewer false then lower threshold",
        "files_sha256": {p: file_hash(ROOT / p) for p in LOCK_FILES},
    }
    write_new_json(V31 / "frozen.json", lock)
    return lock


def verify_lock(lock):
    require(
        lock.get("version") == "3.1" and lock.get("role") == ROLE, "Неверный протокол"
    )
    require(set(lock.get("files_sha256", {})) == set(LOCK_FILES), "Неполный протокол")
    for path, digest in lock["files_sha256"].items():
        require(file_hash(ROOT / path) == digest, f"Изменён вход v3.1: {path}")


def refusal_details(rows, threshold):
    result = threshold_stats(rows, threshold)
    refused = lambda r: r["max_score"] is None or r["max_score"] <= threshold
    result["false_refusal_ids"] = [
        r["id"] for r in rows if r["expected"] == "answer" and refused(r)
    ]
    result["missed_refusal_ids"] = [
        r["id"] for r in rows if r["expected"] == "refuse" and not refused(r)
    ]
    result["by_category"] = {
        category: {
            "total": len(selected),
            "correct_refusals": sum(refused(r) for r in selected),
        }
        for category in sorted(
            {r["refusal_category"] for r in rows if r["expected"] == "refuse"}
        )
        for selected in [[r for r in rows if r.get("refusal_category") == category]]
    }
    return result


def cross_validate_threshold(queries, rows):
    by_id = {r["id"]: r for r in rows}
    folds, heldout = [], []
    for ids in stratified_folds(queries):
        train = [r for r in rows if r["id"] not in ids]
        valid = [by_id[qid] for qid in ids]
        fit = calibrate_threshold(train)
        require(fit["status"] == "selected", "Нет допустимого порога train")
        threshold = fit["selected"]["threshold"]
        stats = refusal_details(valid, threshold)
        folds.append(
            {
                "training_ids": [r["id"] for r in train],
                "validation_ids": ids,
                "calibration": fit,
                "validation": stats,
            }
        )
        heldout.extend(
            {
                "id": r["id"],
                "expected": r["expected"],
                "threshold": threshold,
                "refused": r["max_score"] is None or r["max_score"] <= threshold,
            }
            for r in valid
        )
    require(
        len({r["id"] for r in heldout}) == len(rows), "Не все вопросы оценены один раз"
    )
    return {
        "scope": "threshold-only CV; retrieval previously selected and fixed",
        "folds": folds,
        "out_of_fold": {
            "answerable": sum(r["expected"] == "answer" for r in heldout),
            "refusals": sum(r["expected"] == "refuse" for r in heldout),
            "correct_refusals": sum(
                r["refused"] and r["expected"] == "refuse" for r in heldout
            ),
            "false_refusals": sum(
                r["refused"] and r["expected"] == "answer" for r in heldout
            ),
            "per_query": heldout,
        },
    }


def run(output, profile_output):
    check_branch()
    require(output.resolve() != profile_output.resolve(), "Пути результатов совпадают")
    require(
        not output.exists() and not profile_output.exists(),
        "Результаты не перезаписываются",
    )
    lock = read_json(V31 / "frozen.json")
    verify_lock(lock)
    before, profile, index, queries, ann, dev_q, dev_ann, audit = load_inputs()
    old_threshold = before["legacy_threshold"]
    result = evaluate(index, queries, old_threshold, ann)
    augment_metrics(result, require_complete=True)
    parent = before["comparisons"]["diagnostic"]["combined"]
    verify_ranking(parent, result)
    dev = evaluate(index, dev_q, old_threshold, dev_ann)
    augment_metrics(dev, require_complete=False)
    verify_ranking(before["comparisons"]["development"]["combined"], dev)
    calibration = calibrate_threshold(dev["per_query"])
    require(calibration["status"] == "selected", "Нет допустимого порога development")
    threshold = calibration["selected"]["threshold"]
    calibrated_profile = copy.deepcopy(profile)
    calibrated_profile.update(
        version="3.1",
        role="standalone_v3_development_calibrated",
        refusal_threshold=threshold,
        threshold_calibrated=True,
        calibration_scope="development_only_fixed_v3_retrieval",
        calibration_queries_sha256=json_hash(dev_q),
        calibration_report=str(output),
        annotation_version="3.1",
        integration_enabled=False,
    )
    report = {
        "version": "3.1",
        "role": ROLE,
        "run_at": datetime.now(timezone.utc).isoformat(),
        "protocol": lock,
        "parent_report_sha256": file_hash(FIRST),
        "audit": audit,
        "before": parent,
        "after": result,
        "delta_annotation_only": compare_metrics(parent, result),
        "ranking_and_scores_unchanged": True,
        "development": dev,
        "calibration": calibration,
        "cross_validation": cross_validate_threshold(dev_q, dev["per_query"]),
        "threshold_evaluation": {
            name: {
                "legacy": refusal_details(rows, old_threshold),
                "calibrated": refusal_details(rows, threshold),
            }
            for name, rows in (
                ("development", dev["per_query"]),
                ("diagnostic", result["per_query"]),
            )
        },
        "selected_profile": calibrated_profile,
        "limitations": [
            "Diagnostic questions and rankings already known; not independent holdout.",
            "Only selected v3 diagnostic top-10 fully judged; qrels outside pool incomplete.",
            "Threshold selected on small development; score-only refusal is not application policy.",
            "Retrieval metrics computed before threshold gating; calibration does not improve ranking.",
        ],
    }
    write_new_json(output, report)
    write_new_json(profile_output, calibrated_profile)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "benchmark/results/2026-11-20-retrieval-v3.1.json",
    )
    parser.add_argument(
        "--profile-output", type=Path, default=V31 / "selected_config.json"
    )
    args = parser.parse_args()
    if args.freeze:
        freeze()
        print("Протокол v3.1 зафиксирован")
    else:
        r = run(args.output, args.profile_output)
        print(
            {
                "metrics": r["after"]["overall"],
                "threshold": r["calibration"]["selected"]["threshold"],
            }
        )


if __name__ == "__main__":
    main()
