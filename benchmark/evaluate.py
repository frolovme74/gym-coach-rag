"""Проверить текущий поиск; исторические корпуса не требуются."""

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from app.rag.pipeline import ROOT, PROFILE, RAGPipeline
from benchmark.run import (
    TOPICS,
    evaluate,
    json_hash,
    macro_average,
    retrieval_metrics,
    validate_dataset,
)

BASELINE = "benchmark/results/2026-11-21-practical-v5-final.json"


def targeted(index, queries):
    rows = []
    for q in queries:
        hits = index.search(q["query"], index.n)
        rows.append(
            {
                **q,
                "metrics": retrieval_metrics([h.doc_id for h in hits], q["relevant"]),
                "top10": [h.doc_id for h in hits[:10]],
            }
        )
    return dict(
        overall=macro_average(rows),
        by_topic={
            t: macro_average([r for r in rows if r["topic"] == t]) for t in TOPICS
        },
        per_query=rows,
    )


def run(output, root=ROOT):
    output = Path(output)
    if output.exists():
        raise ValueError("Старые отчёты не перезаписываются")
    folder = root / "benchmark"
    fixed = (folder / "targeted-queries.json").read_bytes()
    lock = json.loads((folder / "frozen-targeted.json").read_text())
    if hashlib.sha256(fixed).hexdigest() != lock["sha256"]:
        raise ValueError("Зафиксированные вопросы изменены")
    targets = json.loads(fixed)
    p = RAGPipeline.from_project(root=root)
    q = json.loads((folder / "queries.json").read_text())
    a = json.loads((folder / "judgments.json").read_text())
    counts = validate_dataset(q, list(p.documents.values()), a, p.index.corpus_sha256)
    if len({t["id"] for t in targets}) != len(targets):
        raise ValueError("Повтор вопроса")
    expected = {(s, t): 3 for s in ("development", "check") for t in TOPICS}
    if Counter((t["split"], t["topic"]) for t in targets) != expected:
        raise ValueError("Целевой набор должен быть сбалансирован")
    if any(
        not t["relevant"] or not set(t["relevant"]) <= set(p.documents) for t in targets
    ):
        raise ValueError("Неизвестный релевантный документ")
    current = evaluate(p.index, q, p.threshold, a)
    unknown = sum(
        h["judgment"] == "unjudged"
        for r in current["per_query"]
        if r["expected"] == "answer"
        for h in r["top_10"]
    )
    target_results = {
        split: {
            name: targeted(index, [t for t in targets if t["split"] == split])
            for name, index in (("raw_bm25", p.index.base), ("runtime", p.index))
        }
        for split in ("development", "check")
    }
    baseline = json.loads((root / BASELINE).read_text())
    comparison = {
        "baseline": BASELINE,
        "same_corpus": p.index.corpus_sha256
        == baseline["reproducibility"]["corpus_sha256"],
        "same_queries": json_hash(q) == baseline["reproducibility"]["queries_sha256"],
        "same_judgments": json_hash(a)
        == baseline["reproducibility"]["judgments_sha256"],
        "same_manifest": json.loads((root / PROFILE).read_text())["manifest_sha256"]
        == baseline["reproducibility"]["manifest_sha256"],
        "same_ranking_and_metrics": current == baseline["after_v5_updated_qrels"],
        "same_targeted_results": target_results == baseline["targeted"],
        "delta": {
            k: current["overall"][k] - baseline["after_v5_updated_qrels"]["overall"][k]
            for k in ("recall_at_5", "recall_at_10", "mrr")
        },
    }
    report = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if not unknown else "needs_annotation",
        "corpus_documents": len(p.documents),
        "dataset": counts,
        "qrels_exhaustive": False,
        "limitations": baseline["limitations"]
        + [
            "Диагностика score-only отказов не оценивает реальные отказы LLM.",
            "raw_bm25 — ConceptBM25 с расширением понятий; runtime добавляет приоритет справок.",
        ],
        "overall": current["overall"],
        "runtime": current,
        "raw_bm25": evaluate(p.index.base, q, p.threshold, a),
        "unjudged_top10": unknown,
        "targeted": target_results,
        "comparison": comparison,
        "reproducibility": {
            "corpus_sha256": p.index.corpus_sha256,
            "manifest_sha256": json.loads((root / PROFILE).read_text())[
                "manifest_sha256"
            ],
            "queries_sha256": json_hash(q),
            "judgments_sha256": json_hash(a),
            "targeted_sha256": lock["sha256"],
            "config": p.index.config,
            "code_sha256": {
                name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                for name in (
                    "benchmark/evaluate.py",
                    "benchmark/run.py",
                    "app/rag/bm25.py",
                    "app/rag/retrieval.py",
                    "app/rag/runtime_retrieval.py",
                    "app/rag/preprocess.py",
                )
            },
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    print(
        json.dumps(
            {
                "overall": report["overall"],
                "comparison": comparison,
                "unjudged_top10": unknown,
                "output": str(output),
            },
            ensure_ascii=False,
        )
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    try:
        run(parser.parse_args().output)
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
