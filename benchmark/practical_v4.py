"""Оценка расширенного корпуса с проверенной разметкой текущего top-10."""

import argparse
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from app.rag.pipeline import ROOT, RAGPipeline
from app.rag.retrieval import ConceptBM25
from benchmark.run import evaluate, json_hash, text_hash, validate_dataset


def run(output):
    if output.exists():
        raise ValueError("Отчёт уже существует; задайте новый путь")
    folder = ROOT / "benchmark/practical-v4"
    pipeline = RAGPipeline.from_project()
    previous = ROOT / "benchmark/retrieval-v3.1"
    queries = json.loads((previous / "queries.json").read_text())
    old_annotations = json.loads((previous / "judgments.json").read_text())
    annotations = copy.deepcopy(old_annotations)
    decisions = json.loads((folder / "annotation-review.json").read_text())["decisions"]
    for decision in decisions:
        qid, did = decision["query_id"], decision["doc_id"]
        review = annotations["queries"][qid]
        if any(j["doc_id"] == did for j in review["judgments"]):
            raise ValueError(f"Повторное решение: {qid}/{did}")
        if did not in review["candidate_ids"]:
            review["candidate_ids"].append(did)
        review["judgments"].append(
            {k: v for k, v in decision.items() if k != "query_id"}
            | {"text_sha256": text_hash(pipeline.documents[did]["text"])}
        )
        if decision["relevant"]:
            next(q for q in queries if q["id"] == qid)["relevant"].append(did)
    annotations["corpus_sha256"] = pipeline.index.corpus_sha256
    annotations["queries_sha256"] = json_hash(queries)
    annotations["dataset_role"] = "practical_v4_audit_not_independent_holdout"
    counts = validate_dataset(
        queries,
        list(pipeline.documents.values()),
        annotations,
        pipeline.index.corpus_sha256,
    )
    old_index_path = ROOT / "data/index/bm25.v3.json"
    old_documents = json.loads((ROOT / "data/corpus-v3/documents.json").read_text())
    old_config = json.loads((previous / "selected_config.json").read_text())[
        "retrieval"
    ]
    old_index = (
        ConceptBM25.load(old_index_path)
        if old_index_path.exists()
        else ConceptBM25.from_documents(old_documents, **old_config)
    )
    if old_index.config != old_config or not old_index.matches_documents(old_documents):
        raise ValueError("Индекс v3 не соответствует сохранённой базе и профилю")
    validate_dataset(
        json.loads((previous / "queries.json").read_text()),
        old_documents,
        old_annotations,
        old_index.corpus_sha256,
    )
    before = evaluate(
        old_index,
        json.loads((previous / "queries.json").read_text()),
        13.909689866544728,
        old_annotations,
    )
    after = evaluate(pipeline.index, queries, pipeline.threshold, annotations)
    unjudged = [
        (q["id"], h["doc_id"])
        for q in after["per_query"]
        if q["expected"] == "answer"
        for h in q["top_10"]
        if h["judgment"] == "unjudged"
    ]
    if unjudged:
        raise ValueError(f"Нужно прочитать и разметить новые попадания: {unjudged}")
    report = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "dataset_role": "reused_diagnostic_questions_not_independent_holdout",
        "dataset": counts,
        "answerable_top10_unjudged": len(unjudged),
        "new_judgments": len(decisions),
        "qrels_exhaustive": False,
        "limitations": [
            "Полностью размечен текущий top-10 предметных вопросов, не весь корпус.",
            "Разметка агентская; набор уже использовался для разработки и не является независимым.",
            "Recall считается по известным релевантным документам; unjudged вне пула считаются нерелевантными.",
            "Score-only refusal не описывает итоговую политику приложения: используются фильтр тем и LLM.",
            "Retrieval-метрики не измеряют смысловую точность ответа или качество тренировочной программы.",
        ],
        "before_v3_1": before,
        "after_v4": after,
        "delta": {
            m: after["overall"][m] - before["overall"][m]
            for m in ("recall_at_5", "recall_at_10", "mrr")
        },
        "reproducibility": {
            "corpus_sha256": pipeline.index.corpus_sha256,
            "queries_sha256": json_hash(queries),
            "judgments_sha256": json_hash(annotations),
            "profile": json.loads((ROOT / "data/corpus-v4/profile.json").read_text()),
            "code_sha256": {
                name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                for name in [
                    "app/rag/bm25.py",
                    "app/rag/preprocess.py",
                    "app/rag/retrieval.py",
                    "app/rag/runtime_retrieval.py",
                    "benchmark/run.py",
                    "benchmark/practical_v4.py",
                ]
            },
        },
    }
    for name, data in [("queries.json", queries), ("judgments.json", annotations)]:
        target = folder / name
        if target.exists() and json.loads(target.read_text()) != data:
            raise ValueError(f"Зафиксированная разметка отличается: {target}")
        if not target.exists():
            target.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as out:
        json.dump(report, out, ensure_ascii=False, indent=2, allow_nan=False)
        out.write("\n")
    print(
        json.dumps(
            {
                "before": before["overall"],
                "after": after["overall"],
                "delta": report["delta"],
                "unjudged": unjudged,
            },
            ensure_ascii=False,
        )
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
