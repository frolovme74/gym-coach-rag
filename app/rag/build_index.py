"""Сборка автономного индекса: python -m app.rag.build_index."""

import argparse
import json
from pathlib import Path

from .bm25 import BM25Index

ROOT = Path(__file__).resolve().parents[2]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Построить собственный индекс BM25")
    parser.add_argument(
        "--corpus", type=Path, default=ROOT / "data/corpus/documents.json"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "data/index/bm25.json")
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
    parser.add_argument("--no-lemmatize", action="store_true")
    parser.add_argument("--keep-stopwords", action="store_true")
    args = parser.parse_args(argv)
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
