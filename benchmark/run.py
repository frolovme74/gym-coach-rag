"""Воспроизводимая оценка retrieval и отдельная диагностика отказов по score."""

import argparse
import hashlib
import json
import math
import platform
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from app.rag.bm25 import BM25Index

ROOT = Path(__file__).resolve().parents[1]
TOPICS = ("training", "nutrition", "supplements", "recovery")
REFUSAL_CATEGORIES = ("out_of_scope", "medical", "prohibited", "unsupported")
METRICS = ("recall_at_5", "recall_at_10", "mrr")


def json_hash(value):
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def validate_dataset(queries, documents, annotations, corpus_sha256):
    """Не разрешает запуск на устаревших qrels или неподтверждённых positives."""
    require(isinstance(queries, list) and queries, "Нужен непустой список запросов")
    require(isinstance(annotations, dict), "Нужен объект разметки")
    require(
        type(annotations.get("version")) is int and annotations["version"] == 1,
        "Неизвестная версия разметки",
    )
    require(
        annotations.get("corpus_sha256") == corpus_sha256,
        "Разметка относится к другому корпусу",
    )
    require(
        annotations.get("queries_sha256") == json_hash(queries),
        "Запросы изменились после разметки",
    )
    doc_map = {d["doc_id"]: d for d in documents}
    require(len(doc_map) == len(documents), "Повтор doc_id в корпусе")
    reviews = annotations.get("queries")
    require(isinstance(reviews, dict), "Нет решений по запросам")
    ids, texts, counts, refusal_counts = set(), set(), Counter(), Counter()
    for q in queries:
        require(isinstance(q, dict), "Запрос должен быть объектом")
        qid, query = q.get("id"), q.get("query")
        require(nonempty(qid) and nonempty(query), "Пустой id/query")
        require(qid not in ids, f"Повтор id {qid}")
        normalized = " ".join(query.casefold().split())
        require(normalized not in texts, f"Повтор текста запроса {qid}")
        ids.add(qid)
        texts.add(normalized)
        relevant = q.get("relevant")
        require(
            isinstance(relevant, list) and all(nonempty(d) for d in relevant),
            f"Некорректный relevant: {qid}",
        )
        require(len(set(relevant)) == len(relevant), f"Повтор relevant: {qid}")
        require(set(relevant) <= doc_map.keys(), f"Неизвестный relevant: {qid}")
        expected = q.get("expected")
        require(expected in ("answer", "refuse"), f"Некорректный expected: {qid}")
        if expected == "answer":
            require(q.get("topic") in TOPICS, f"Некорректная область: {qid}")
            require(len(relevant) >= 3, f"Нужно >=3 релевантных чанка: {qid}")
            counts[q["topic"]] += 1
        else:
            require(
                q.get("topic") is None and not relevant,
                f"У отказов topic=null и relevant=[]: {qid}",
            )
            require(
                q.get("refusal_category") in REFUSAL_CATEGORIES
                and nonempty(q.get("reason")),
                f"Нужны категория и причина отказа: {qid}",
            )
            refusal_counts[q["refusal_category"]] += 1
        review = reviews.get(qid)
        require(isinstance(review, dict), f"Нет обоснований: {qid}")
        candidates = review.get("candidate_ids", [])
        require(
            isinstance(candidates, list) and all(nonempty(d) for d in candidates),
            f"Некорректные кандидаты: {qid}",
        )
        require(
            len(set(candidates)) == len(candidates)
            and set(candidates) <= doc_map.keys(),
            f"Повтор/неизвестный кандидат: {qid}",
        )
        judgments = review.get("judgments")
        require(isinstance(judgments, list), f"Нет judgments: {qid}")
        seen, positives = set(), set()
        for judgment in judgments:
            require(isinstance(judgment, dict), f"Некорректное решение: {qid}")
            did = judgment.get("doc_id")
            require(
                nonempty(did) and did in candidates and did not in seen,
                f"Повтор или неизвестный doc_id в разметке: {qid}",
            )
            seen.add(did)
            require(
                type(judgment.get("relevant")) is bool,
                f"Метка должна быть bool: {qid}/{did}",
            )
            if judgment["relevant"]:
                positives.add(did)
            evidence = judgment.get("evidence")
            require(
                nonempty(evidence) and evidence in doc_map[did]["text"],
                f"Цитата отсутствует в чанке: {qid}/{did}",
            )
            require(
                judgment.get("text_sha256") == text_hash(doc_map[did]["text"]),
                f"Текст изменился после разметки: {qid}/{did}",
            )
            require(nonempty(judgment.get("reason")), f"Нет причины: {qid}/{did}")
        require(positives == set(relevant), f"Relevant расходится с решениями: {qid}")
    require(set(reviews) == ids, "Состав запросов и разметки различается")
    require(
        set(counts) == set(TOPICS) and len(set(counts.values())) == 1,
        "Предметные запросы должны поровну покрывать четыре области",
    )
    require(
        set(refusal_counts) == set(REFUSAL_CATEGORIES),
        "Нужны все четыре категории отказов",
    )
    return {
        "answerable": sum(counts.values()),
        "by_topic": dict(counts),
        "refusals": sum(refusal_counts.values()),
        "refusals_by_category": dict(refusal_counts),
    }


def retrieval_metrics(ranked_ids, relevant):
    """Macro-единица: один запрос. RR использует полный ranking, не top-10."""
    relevant = set(relevant)
    require(bool(relevant), "Recall/MRR не определены для запроса на отказ")
    require(len(set(ranked_ids)) == len(ranked_ids), "Дубли в ranking")
    first = next(
        (rank for rank, did in enumerate(ranked_ids, 1) if did in relevant), None
    )
    return {
        "recall_at_5": len(set(ranked_ids[:5]) & relevant) / len(relevant),
        "recall_at_10": len(set(ranked_ids[:10]) & relevant) / len(relevant),
        "mrr": 1 / first if first is not None else 0.0,
    }


def macro_average(rows):
    require(bool(rows), "Пустая группа метрик")
    return {
        "queries": len(rows),
        **{
            key: math.fsum(row["metrics"][key] for row in rows) / len(rows)
            for key in METRICS
        },
    }


def evaluate(index, queries, refusal_threshold=0.0, annotations=None):
    require(
        not isinstance(refusal_threshold, bool)
        and isinstance(refusal_threshold, (int, float))
        and math.isfinite(refusal_threshold)
        and refusal_threshold >= 0,
        "Порог должен быть конечным числом >=0",
    )
    details = []
    for q in queries:
        ranking = index.search(q["query"], k=index.n)
        ids = [r.doc_id for r in ranking]
        positions = {did: rank for rank, did in enumerate(ids, 1)}
        relevant = set(q["relevant"])
        reviewed = {
            j["doc_id"]: j["relevant"]
            for j in (annotations or {})
            .get("queries", {})
            .get(q["id"], {})
            .get("judgments", [])
        }
        max_score = ranking[0].score if ranking else None
        predicted_refusal = max_score is None or max_score <= refusal_threshold
        row = {
            "id": q["id"],
            "topic": q["topic"],
            "query": q["query"],
            "expected": q["expected"],
            "relevant_count": len(relevant),
            "retrieved_count": len(ranking),
            "max_score": max_score,
            "score_only_refusal": predicted_refusal,
            "top_10": [
                {
                    "doc_id": r.doc_id,
                    "score": r.score,
                    "judgment": "relevant"
                    if r.doc_id in relevant
                    else "non_relevant"
                    if reviewed.get(r.doc_id) is False
                    else "unjudged",
                }
                for r in ranking[:10]
            ],
            "relevant_positions": {did: positions.get(did) for did in sorted(relevant)},
        }
        if q["expected"] == "answer":
            row["metrics"] = retrieval_metrics(ids, relevant)
        else:
            row["refusal_category"] = q["refusal_category"]
        details.append(row)
    positive = [r for r in details if r["expected"] == "answer"]
    refusals = [r for r in details if r["expected"] == "refuse"]

    def refusal_summary(rows):
        return {
            "queries": len(rows),
            "score_only_refusals": sum(r["score_only_refusal"] for r in rows),
            "rate": sum(r["score_only_refusal"] for r in rows) / len(rows)
            if rows
            else None,
        }

    return {
        "overall": macro_average(positive),
        "by_topic": {
            topic: macro_average([r for r in positive if r["topic"] == topic])
            for topic in TOPICS
        },
        "refusal_diagnostics": {
            "mode": "score_only_proxy_not_application_policy",
            "threshold": refusal_threshold,
            "rule": "empty ranking or max_score <= threshold",
            "threshold_calibrated": False,
            "expected_refusals": refusal_summary(refusals),
            "by_category": {
                category: refusal_summary(
                    [r for r in refusals if r["refusal_category"] == category]
                )
                for category in REFUSAL_CATEGORIES
            },
            "false_refusals_on_answerable": refusal_summary(positive),
        },
        "per_query": details,
    }


def run(
    index_path, corpus_path, queries_path, judgments_path, output, refusal_threshold=0.0
):
    for path in (index_path, corpus_path, queries_path, judgments_path):
        require(
            output.resolve() != path.resolve(),
            "Файл результата совпадает с входными данными",
        )
    documents = json.loads(corpus_path.read_text(encoding="utf-8"))
    queries = json.loads(queries_path.read_text(encoding="utf-8"))
    annotations = json.loads(judgments_path.read_text(encoding="utf-8"))
    index = BM25Index.load(index_path)
    require(
        index.matches_documents(documents),
        "Индекс устарел: пересоберите из текущего корпуса",
    )
    counts = validate_dataset(queries, documents, annotations, index.corpus_sha256)
    report = {
        "version": 1,
        "status": "passed",
        "run_at": datetime.now(ZoneInfo("Europe/Moscow")).isoformat(),
        "dataset_role": "development_v1",
        "qrels_exhaustive": False,
        "limitations": [
            "Агентская разметка по тематическому пулу; возможны неотмеченные релевантные чанки.",
            "Непомеченные документы считаются нерелевантными при расчёте метрик.",
            "Проверяется retrieval, а не достоверность источников или качество ответа LLM.",
            "Диагностика отказов использует только score; интеграция и политика отказов не реализованы.",
        ],
        "dataset": counts,
        "index": {
            "documents": index.n,
            "terms": len(index.df),
            "avgdl": index.avgdl,
            "k1": index.k1,
            "b": index.b,
            "lemmatize": index.lemmatize,
            "remove_stopwords": index.remove_stopwords,
        },
        "reproducibility": {
            "python": platform.python_version(),
            "corpus_sha256": index.corpus_sha256,
            "queries_sha256": json_hash(queries),
            "judgments_sha256": json_hash(annotations),
            "code_sha256": {
                str(path.relative_to(ROOT)): hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                for path in (
                    ROOT / "app/rag/bm25.py",
                    ROOT / "app/rag/preprocess.py",
                    Path(__file__),
                )
            },
        },
        **evaluate(index, queries, refusal_threshold, annotations),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "overall": report["overall"],
                "by_topic": report["by_topic"],
                "refusals": report["refusal_diagnostics"],
                "output": str(output),
            },
            ensure_ascii=False,
        )
    )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=ROOT / "data/index/bm25.json")
    parser.add_argument(
        "--corpus", type=Path, default=ROOT / "data/corpus/documents.json"
    )
    parser.add_argument("--queries", type=Path, default=ROOT / "benchmark/queries.json")
    parser.add_argument(
        "--judgments", type=Path, default=ROOT / "benchmark/judgments.json"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "benchmark/results"
        / (
            datetime.now(ZoneInfo("Europe/Moscow")).strftime("%Y-%m-%d_%H%M%S")
            + "-baseline.json"
        ),
    )
    parser.add_argument("--refusal-threshold", type=float, default=0.0)
    args = parser.parse_args(argv)
    try:
        run(
            args.index,
            args.corpus,
            args.queries,
            args.judgments,
            args.output,
            args.refusal_threshold,
        )
    except (OSError, ValueError, TypeError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
