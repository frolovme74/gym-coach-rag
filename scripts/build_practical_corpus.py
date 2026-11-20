"""Сборка единого v4: прежний корпус плюс редактируемые памятки проекта."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.rag.retrieval import ConceptBM25, RULESET, _digest  # noqa: E402


def main():
    target = ROOT / "data/corpus-v4"
    notes = json.loads((target / "practical.json").read_text())
    documents = json.loads((ROOT / "data/corpus-v3/documents.json").read_text())
    manifest = json.loads((ROOT / "data/corpus-v3/manifest.json").read_text())
    for row in notes:
        documents.append(
            {k: row[k] for k in ("doc_id", "topic", "source", "section", "text")}
        )
        manifest.append(
            {
                "doc_id": row["doc_id"],
                "source_id": row["doc_id"],
                "source_url": "",
                "kind": "project_guide",
                "created_with_ai": True,
                "review": "editorial_check_not_external_expert_review",
                "references": row.get("references", []),
                "answer_patterns": row.get("answer_patterns", []),
                "program": row.get("program", {}),
            }
        )
    if len({d["doc_id"] for d in documents}) != len(documents):
        raise ValueError("Повтор doc_id")
    config = json.loads(
        (ROOT / "benchmark/retrieval-v3.1/selected_config.json").read_text()
    )["retrieval"]
    index = ConceptBM25.from_documents(documents, **config)
    profile = {
        "version": "4",
        "corpus": "data/corpus-v4/documents.json",
        "index": "data/index/bm25.v4.json",
        "corpus_sha256": index.corpus_sha256,
        "rules_sha256": _digest(RULESET),
        "manifest_sha256": _digest(manifest),
        "retrieval": config,
        "refusal_threshold": 0.0,
        "threshold_calibrated": False,
        "refusal_policy": "positive_bm25_then_context_sufficiency_by_llm",
    }
    for name, value in (
        ("documents.json", documents),
        ("manifest.json", manifest),
        ("profile.json", profile),
    ):
        (target / name).write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n"
        )
    index.save(ROOT / profile["index"])
    print(f"v4: {len(documents)} документов, {len(notes)} новых памяток")


if __name__ == "__main__":
    main()
