"""Экспериментальный BM25: понятия спортивного словаря и повторы запроса.

Исходные тексты, числа и отрицания не меняются. Служебные термы добавляются
одинаковыми правилами в документы и запросы; вес понятия задаётся только
числом его повторов в запросе. Это не семантическая модель и не политика отказа.
"""

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

from .bm25 import BM25Index
from .preprocess import normalize_text, preprocess

FORMAT_VERSION = 1
VARIANTS = {
    "plain": (False, False),
    "dedup": (False, True),
    "concepts": (True, False),
    "concepts_dedup": (True, True),
}
# Правила работают в пределах одного предложения и окна до 32 слов.
# Корни учитывают русские окончания; общие слова «холод», «пара», «неделя»
# сами по себе не создают понятие.
RULES = {
    "periodization": (
        r"\bпериодизац\w*\b|\bperiodi[sz]ation\b",
        r"\b(?:меня|изменя|варьир)\w*\b.{0,80}\bнагруз\w*\b.{0,80}\bот недели к неделе\b",
    ),
    "superset": (
        r"\bсуперсет(?:а|у|ом|е|ы|ов|ам|ами|ах)?\b|\bsupersets?\b",
        r"\bпарн\w*\s+(?:силов\w*\s+)?(?:подход|упражнен)\w*\b",
    ),
    "cold_water_immersion": (
        r"\b(?:ледян|холодн)\w*\s+ванн\w*\b",
        r"\bванн\w*\s+(?:(?:со?|с)\s+)?льд\w*\b",
        r"\b(?:погружен|купан)\w*\b.{0,35}\bхолодн\w*\s+вод\w*\b",
        r"\bcold[- ]water immersion\b|\bice baths?\b|\bcwi\b",
    ),
    "post_exercise_cooldown": (
        r"\b(?:активн\w*\s+)?заминк\w*\b",
        r"\bактивн\w*\s+охлажден\w*\b",
        r"\bactive cool[- ]?down\b",
    ),
}
ALL_OF_RULES = {
    "periodization": (
        (
            r"\bчеред\w*\b",
            r"\bтяжел\w*\b",
            r"\bлегк\w*\b",
            r"\b(?:дн|день)\w*\b",
            r"\b(?:нагруз|трениров)\w*\b",
        ),
    ),
    "superset": (
        (
            r"\bупражнен\w*\b",
            r"\b(?:пар|парн|объедин)\w*\b",
            r"\b(?:противоположн|антагонист)\w*\b",
            r"\bмышц\w*\b",
        ),
    ),
}
RULESET = {
    "any": RULES,
    "all": ALL_OF_RULES,
    "window": 32,
    "stride": 16,
    "sentence_separator": r"[.!?;\n]+",
    "version": 1,
}
TOKENS = {key: f"__concept_{key}__" for key in RULES}
COMPILED = {key: tuple(re.compile(p) for p in values) for key, values in RULES.items()}


def _digest(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def _fingerprint(documents):
    rows = {}
    for document in documents:
        did, text = document.get("doc_id"), document.get("text")
        if not isinstance(did, str) or not did.strip() or did in rows:
            raise ValueError("Пустой или повторный doc_id")
        if not isinstance(text, str):
            raise TypeError("text должен быть строкой")
        rows[did] = text
    raw = json.dumps(sorted(rows.items()), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def detect_concepts(text):
    """Распознаёт темы, но не утверждение, отрицание или научный вывод."""
    normalized = normalize_text(text)
    found = set()
    for sentence in re.split(r"[.!?;\n]+", normalized):
        words = sentence.split()
        for start in range(0, len(words), 16):
            window = " ".join(words[start : start + 32])
            for key, patterns in COMPILED.items():
                if any(pattern.search(window) for pattern in patterns):
                    found.add(key)
            for key, alternatives in ALL_OF_RULES.items():
                if any(
                    all(re.search(pattern, window) for pattern in patterns)
                    for patterns in alternatives
                ):
                    found.add(key)
    return tuple(sorted(found))


def _settings(concepts, deduplicate_query, concept_weight, options):
    if type(concepts) is not bool or type(deduplicate_query) is not bool:
        raise TypeError("concepts и deduplicate_query должны быть bool")
    if type(concept_weight) is not int or concept_weight < 1:
        raise ValueError("concept_weight должен быть положительным целым")
    # Валидируем k1/b/параметры до обработки всего корпуса.
    BM25Index({}, **options)
    return dict(
        concepts=concepts,
        deduplicate_query=deduplicate_query,
        concept_weight=concept_weight,
        **options,
    )


class ConceptBM25:
    """Обёртка неизменного собственного BM25 с явными настройками эксперимента."""

    @classmethod
    def from_documents(
        cls,
        documents,
        *,
        concepts=True,
        deduplicate_query=False,
        concept_weight=3,
        k1=3.0,
        b=0.0,
        lemmatize=True,
        remove_stopwords=False,
    ):
        config = _settings(
            concepts,
            deduplicate_query,
            concept_weight,
            dict(k1=k1, b=b, lemmatize=lemmatize, remove_stopwords=remove_stopwords),
        )
        documents = list(documents)
        fingerprint = _fingerprint(documents)
        tokenized, doc_concepts = {}, {}
        for document in documents:
            did, text = document["doc_id"], document["text"]
            tokens = preprocess(
                text, lemmatize=lemmatize, remove_stopwords=remove_stopwords
            )
            matches = detect_concepts(text) if concepts else ()
            tokenized[did] = tokens + [TOKENS[key] for key in matches]
            doc_concepts[did] = matches
        index = BM25Index(
            tokenized,
            k1=k1,
            b=b,
            lemmatize=lemmatize,
            remove_stopwords=remove_stopwords,
        )
        return cls(index, config, fingerprint, doc_concepts)

    def __init__(self, index, config, corpus_sha256, document_concepts):
        self.index = index
        self.config = dict(config)
        self.corpus_sha256 = corpus_sha256
        self.document_concepts = dict(document_concepts)

    @property
    def n(self):
        return self.index.n

    @property
    def k1(self):
        return self.index.k1

    @property
    def b(self):
        return self.index.b

    @property
    def lemmatize(self):
        return self.index.lemmatize

    @property
    def remove_stopwords(self):
        return self.index.remove_stopwords

    def matches_documents(self, documents):
        return self.corpus_sha256 == _fingerprint(documents)

    def query_tokens(self, text):
        tokens = preprocess(
            text, lemmatize=self.lemmatize, remove_stopwords=self.remove_stopwords
        )
        if self.config["deduplicate_query"]:
            tokens = list(dict.fromkeys(tokens))
        if self.config["concepts"]:
            for key in detect_concepts(text):
                tokens.extend([TOKENS[key]] * self.config["concept_weight"])
        return tokens

    def search(self, text, k=5):
        return self.index.search_tokens(self.query_tokens(text), k=k)

    def save(self, path):
        """Отдельный формат: обычный BM25.load не примет файл этой обёртки."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            inner = Path(tmp) / "bm25.json"
            self.index.save(inner)
            payload = dict(
                version=FORMAT_VERSION,
                kind="concept_bm25",
                rules_sha256=_digest(RULESET),
                config=self.config,
                corpus_sha256=self.corpus_sha256,
                document_concepts=self.document_concepts,
                bm25=json.loads(inner.read_text()),
            )
        encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                delete=False,
            ) as stream:
                temporary = Path(stream.name)
                stream.write(encoded)
                stream.write("\n")
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if (
            payload.get("version") != FORMAT_VERSION
            or payload.get("kind") != "concept_bm25"
        ):
            raise ValueError("Неизвестный формат ConceptBM25")
        if payload.get("rules_sha256") != _digest(RULESET):
            raise ValueError("Правила понятий изменились; пересоберите индекс")
        config = payload["config"]
        expected = {
            "concepts",
            "deduplicate_query",
            "concept_weight",
            "k1",
            "b",
            "lemmatize",
            "remove_stopwords",
        }
        if set(config) != expected:
            raise ValueError("Неверные настройки ConceptBM25")
        options = {
            key: config[key] for key in ("k1", "b", "lemmatize", "remove_stopwords")
        }
        _settings(
            config["concepts"],
            config["deduplicate_query"],
            config["concept_weight"],
            options,
        )
        with tempfile.TemporaryDirectory() as tmp:
            inner = Path(tmp) / "bm25.json"
            inner.write_text(
                json.dumps(payload["bm25"], ensure_ascii=False), encoding="utf-8"
            )
            index = BM25Index.load(inner)
        if any(getattr(index, key) != value for key, value in options.items()):
            raise ValueError("Настройки обёртки и BM25 расходятся")
        fingerprint = payload.get("corpus_sha256")
        if not isinstance(fingerprint, str) or not re.fullmatch(
            r"[0-9a-f]{64}", fingerprint
        ):
            raise ValueError("Неверный отпечаток корпуса")
        doc_concepts = payload.get("document_concepts")
        if not isinstance(doc_concepts, dict) or set(doc_concepts) != set(
            index.doc_lengths
        ):
            raise ValueError("Неверный состав документов")
        for did, values in doc_concepts.items():
            if (
                not isinstance(values, list)
                or len(values) != len(set(values))
                or not set(values) <= RULES.keys()
            ):
                raise ValueError("Неверные понятия документа")
            expected_values = set(values) if config["concepts"] else set()
            if set(values) != expected_values:
                raise ValueError("Понятия запрещены конфигурацией")
            for key, token in TOKENS.items():
                frequency = dict(index.postings.get(token, ())).get(did, 0)
                if frequency != int(key in expected_values):
                    raise ValueError("Понятия расходятся с postings")
        return cls(
            index, config, fingerprint, {k: tuple(v) for k, v in doc_concepts.items()}
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--corpus", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--variant", choices=VARIANTS, default="concepts")
    search = commands.add_parser("search")
    search.add_argument("--index", type=Path, required=True)
    search.add_argument("--query", required=True)
    search.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    if args.command == "build":
        if args.output.resolve() == args.corpus.resolve():
            parser.error("Выход индекса не может совпадать с корпусом")
        concepts, deduplicate = VARIANTS[args.variant]
        index = ConceptBM25.from_documents(
            json.loads(args.corpus.read_text()),
            concepts=concepts,
            deduplicate_query=deduplicate,
        )
        index.save(args.output)
        print(f"Индекс: {index.n} документов; вариант {args.variant}; {args.output}")
    else:
        index = ConceptBM25.load(args.index)
        print(
            json.dumps(
                [
                    dict(doc_id=r.doc_id, score=r.score)
                    for r in index.search(args.query, args.k)
                ],
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
