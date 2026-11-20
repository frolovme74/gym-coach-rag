"""Учебный benchmark и техническая проверка корпуса, без оценки его релевантности.

Запуск после сборки индекса: python -m benchmark.smoke_bm25.
Полный benchmark с ручной разметкой остаётся отдельным этапом.
"""

import argparse
import json
import math
import platform
from datetime import datetime
from pathlib import Path
from time import perf_counter
from zoneinfo import ZoneInfo

from app.rag.bm25 import BM25Index

ROOT = Path(__file__).resolve().parents[1]
QUERIES = [
    ("training", "Сколько подходов в неделю выполнять для роста мышц?"),
    ("nutrition", "Какое количество белка нужно при силовых тренировках?"),
    ("supplements", "Помогает ли креатин увеличивать силу?"),
    ("recovery", "Как недостаток сна влияет на восстановление после тренировок?"),
]


def run(index_path, output):
    fixture = json.loads((ROOT / "tests/fixtures/bm25_methodical.json").read_text())
    example = BM25Index(fixture["documents"], **fixture["parameters"])
    results = example.search_tokens(fixture["query"])
    assert [r.doc_id for r in results] == fixture["expected_ranking"]
    for result in results:
        assert math.isclose(
            result.score,
            fixture["expected_scores"][result.doc_id],
            rel_tol=1e-13,
            abs_tol=1e-14,
        )
    relevant = set(fixture["relevant"])
    ranking = [r.doc_id for r in results]
    educational_metrics = {
        f"recall_at_{k}": len(relevant.intersection(ranking[:k])) / len(relevant)
        for k in [1, 3, 5]
    }
    educational_metrics["mrr"] = next(
        (1 / rank for rank, doc_id in enumerate(ranking, 1) if doc_id in relevant), 0
    )
    documents = json.loads((ROOT / "data/corpus/documents.json").read_text())
    started = perf_counter()
    index = BM25Index.load(index_path)
    load_seconds = perf_counter() - started
    assert index.matches_documents(documents), "Индекс не соответствует корпусу"
    doc_ids = {d["doc_id"] for d in documents}
    assert index.n == len(documents)
    samples = []
    for topic, query in QUERIES:
        started = perf_counter()
        matches = index.search(query, k=5)
        seconds = perf_counter() - started
        assert matches, f"Нет лексических совпадений: {query}"
        assert all(
            r.doc_id in doc_ids and math.isfinite(r.score) and r.score > 0
            for r in matches
        )
        assert matches == sorted(matches, key=lambda r: (-r.score, r.doc_id))
        samples.append(
            {
                "topic": topic,
                "query": query,
                "seconds": seconds,
                "results": [{"doc_id": r.doc_id, "score": r.score} for r in matches],
            }
        )
    assert index.search("несуществующийтермззффхх") == []
    assert index.search("") == []
    report = {
        "status": "passed",
        "run_at": datetime.now(ZoneInfo("Europe/Moscow")).isoformat(),
        "python": platform.python_version(),
        "parameters": {"k1": index.k1, "b": index.b, "k": 5},
        "preprocess": {
            "lemmatize": index.lemmatize,
            "remove_stopwords": index.remove_stopwords,
        },
        "methodical_example": {
            "source": fixture["source"],
            "queries": 1,
            "metrics": educational_metrics,
            "scores": example.scores(fixture["query"]),
        },
        "corpus_smoke": {
            "documents": index.n,
            "terms": len(index.df),
            "postings": sum(len(p) for p in index.postings.values()),
            "avgdl": index.avgdl,
            "load_seconds": load_seconds,
            "corpus_sha256": index.corpus_sha256,
            "queries": samples,
            "empty_and_unknown_queries": "passed",
        },
        "corpus_relevance_benchmark": {
            "status": "not_run",
            "reason": "Набор запросов и ручная разметка ещё не подготовлены.",
            "limitation": "Метрики одного учебного примера не оценивают качество поиска по фитнес-корпусу.",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "documents": index.n,
                "terms": len(index.df),
                "report": str(output),
            },
            ensure_ascii=False,
        )
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=ROOT / "data/index/bm25.json")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "benchmark/results/2026-11-20-bm25-smoke.json",
    )
    args = parser.parse_args()
    run(args.index, args.output)


if __name__ == "__main__":
    main()
