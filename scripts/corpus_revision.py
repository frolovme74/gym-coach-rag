"""Применить проверенную редактуру как отдельную версию корпуса."""

import argparse
import copy
import json
from pathlib import Path

from benchmark.holdout import check_branch, read_json, write_new_json
from benchmark.run import ROOT, require, text_hash
from scripts.corpus_prepare import word_count


def revise(documents, manifest, patches):
    revised = copy.deepcopy(documents)
    provenance = copy.deepcopy(manifest)
    docs = {d["doc_id"]: d for d in revised}
    origins = {d["doc_id"]: d for d in provenance}
    require(len(docs) == len(revised), "Повтор doc_id в корпусе")
    seen = set()
    for patch in patches:
        did = patch["doc_id"]
        require(
            did in docs and did in origins and did not in seen,
            "Неизвестный или повторный patch",
        )
        seen.add(did)
        doc = docs[did]
        require(doc["text"] == patch["old_text"], f"Устаревший исходный текст: {did}")
        require(
            doc["section"] == patch.get("old_section", doc["section"]),
            f"Устаревший заголовок: {did}",
        )
        text = patch["new_text"]
        require(isinstance(text, str) and bool(text.strip()), f"Пустой перевод: {did}")
        require(
            bool(patch.get("reason")) and bool(patch.get("provenance")),
            f"Нет обоснования/оригинала: {did}",
        )
        doc["text"] = text
        doc["section"] = patch.get("new_section", doc["section"])
        origin = origins[did]
        # Старые unit hashes относятся к тексту до правки, не к новой редакции.
        if "units" in origin:
            origin["parent_units"] = origin.pop("units")
        origin["revision"] = {
            "version": 3,
            "parent_text_sha256": text_hash(patch["old_text"]),
            "text_sha256": text_hash(text),
            "reason": patch["reason"],
            "provenance": patch["provenance"],
            "review_scope": patch.get("review_scope", "targeted_semantic_revision"),
            "word_count": word_count(text),
        }
        origin["text_sha256"] = text_hash(text)
        origin["word_count"] = word_count(text)
        origin["whitespace_word_count"] = len(text.split())
        origin["parent_status"] = origin.get("status")
        origin["parent_full_semantic_review"] = origin.get("full_semantic_review")
        origin["status"] = "targeted_semantic_revision"
        origin["full_semantic_review"] = False
        if not 300 <= word_count(text) <= 600:
            origin["flags"] = list(origin.get("flags", [])) + [
                "revision_preserves_source_coverage_outside_target_word_count"
            ]
    return revised, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--patch", type=Path, default=ROOT / "data/corpus-v3/translation_patch.json"
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/corpus-v3")
    args = parser.parse_args()
    check_branch()
    documents = read_json(ROOT / "data/corpus/documents.json")
    manifest = read_json(ROOT / "data/corpus/manifest.json")
    patch = read_json(args.patch)
    patches = patch["patches"] if isinstance(patch, dict) else patch
    target_docs, target_manifest = (
        args.output_dir / "documents.json",
        args.output_dir / "manifest.json",
    )
    require(
        not target_docs.exists() and not target_manifest.exists(),
        "Версия уже существует; выберите новую папку",
    )
    revised, provenance = revise(documents, manifest, patches)
    write_new_json(target_docs, revised)
    write_new_json(target_manifest, provenance)
    print(
        json.dumps(
            {
                "documents": len(revised),
                "patched": len(patches),
                "output": str(target_docs),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
