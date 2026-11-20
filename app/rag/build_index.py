"""Сборка автономного индекса: python -m app.rag.build_index."""

import argparse
import json
import re
from pathlib import Path

from .bm25 import BM25Index

ROOT = Path(__file__).resolve().parents[2]


def build_project(root=ROOT):
    """Собрать текущий корпус без зависимости от исторических версий."""
    from .prompts import passages
    from .retrieval import ConceptBM25, RULESET, _digest

    profile_path = root / "data/corpus/profile.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    corpus_path = root / profile["corpus"]
    manifest_path = corpus_path.with_name("manifest.json")
    output = root / profile["index"]
    if output.resolve() in {
        p.resolve() for p in (profile_path, corpus_path, manifest_path)
    }:
        raise ValueError("Индекс не может перезаписать входные данные")
    documents = json.loads(corpus_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_id = {d["doc_id"]: d for d in documents}
    if len(by_id) != len(documents) or len({m["doc_id"] for m in manifest}) != len(
        manifest
    ):
        raise ValueError("Повтор doc_id")
    if set(by_id) != {m["doc_id"] for m in manifest}:
        raise ValueError("Manifest не соответствует корпусу")
    for d in documents:
        if d.get("topic") not in {"training", "nutrition", "supplements", "recovery"}:
            raise ValueError("Неизвестная область документа")
        if any(
            not isinstance(d.get(k), str) or not d[k].strip()
            for k in ("doc_id", "source", "section", "text")
        ):
            raise ValueError("Документ содержит пустые поля")
    for m in manifest:
        if not set(m.get("support_doc_ids", [])) <= set(by_id):
            raise ValueError("Неизвестный подтверждающий документ")
        for pattern in m.get("answer_patterns", []) + m.get(
            "answer_exclude_patterns", []
        ):
            re.compile(pattern)
        parts = passages(by_id[m["doc_id"]]["text"])
        if any(q not in parts for q in m.get("general_dose_passages", [])):
            raise ValueError("Общая дозировка должна ссылаться на точное предложение")
    index = ConceptBM25.from_documents(documents, **profile["retrieval"])
    profile.update(
        corpus_sha256=index.corpus_sha256,
        manifest_sha256=_digest(manifest),
        rules_sha256=_digest(RULESET),
    )
    index.save(output)
    temp = profile_path.with_suffix(".tmp")
    temp.write_text(
        json.dumps(profile, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temp.replace(profile_path)
    return {
        "documents": index.n,
        "corpus_sha256": index.corpus_sha256,
        "output": str(output),
        "retrieval": index.config,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Построить собственный индекс BM25")
    parser.add_argument(
        "--plain-bm25", action="store_true", help="Учебный BM25 без профиля приложения"
    )
    parser.add_argument(
        "--corpus", type=Path, default=ROOT / "data/corpus/documents.json"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "data/index/bm25.json")
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
    parser.add_argument("--no-lemmatize", action="store_true")
    parser.add_argument("--keep-stopwords", action="store_true")
    args = parser.parse_args(argv)
    if not args.plain_bm25:
        if (
            args.corpus != ROOT / "data/corpus/documents.json"
            or args.output != ROOT / "data/index/bm25.json"
            or args.k1 != 1.5
            or args.b != 0.75
            or args.no_lemmatize
            or args.keep_stopwords
        ):
            parser.error(
                "Параметры приложения задаются в data/corpus/profile.json; для учебной сборки укажите --plain-bm25"
            )
        try:
            print(json.dumps(build_project(), ensure_ascii=False))
        except (
            OSError,
            ValueError,
            TypeError,
            KeyError,
            RuntimeError,
            re.error,
        ) as error:
            parser.error(str(error))
        return
    if args.corpus.resolve() == args.output.resolve():
        parser.error("Корпус и файл индекса не могут совпадать")
    try:
        documents = json.loads(args.corpus.read_text(encoding="utf-8"))
        index = BM25Index.from_documents(
            documents,
            k1=args.k1,
            b=args.b,
            lemmatize=not args.no_lemmatize,
            remove_stopwords=not args.keep_stopwords,
        )
        index.save(args.output)
    except (OSError, ValueError, TypeError, RuntimeError) as error:
        parser.error(str(error))
    print(
        json.dumps(
            {
                "documents": index.n,
                "terms": len(index.df),
                "avgdl": index.avgdl,
                "k1": index.k1,
                "b": index.b,
                "corpus_sha256": index.corpus_sha256,
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
