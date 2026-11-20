"""Самостоятельный инвертированный индекс и BM25 без поисковых библиотек."""

import hashlib
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from types import MappingProxyType

from .preprocess import preprocess

FORMAT_VERSION = 1


@dataclass(frozen=True)
class SearchResult:
    doc_id: str
    score: float


def _identifier(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("doc_id должен быть непустой строкой")
    return value


def _tokens(values):
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise TypeError("Ожидается последовательность токенов, а не строка")
    for value in values:
        if not isinstance(value, str) or not value or any(c.isspace() for c in value):
            raise ValueError("Токен должен быть непустой строкой без пробелов")
        yield value


def _documents(documents):
    if isinstance(documents, (str, bytes, Mapping)):
        raise TypeError("Ожидается последовательность записей корпуса")
    rows = {}
    for document in documents:
        if not isinstance(document, Mapping):
            raise TypeError("Документ должен быть отображением с doc_id и text")
        doc_id = _identifier(document.get("doc_id"))
        if doc_id in rows:
            raise ValueError(f"Повтор doc_id: {doc_id}")
        text = document.get("text")
        if not isinstance(text, str):
            raise TypeError(f"text должен быть строкой: {doc_id}")
        rows[doc_id] = text
    return rows


def _fingerprint(rows):
    data = json.dumps(sorted(rows.items()), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


class BM25Index:
    """Неизменяемые postings; настройки запросов совпадают с настройками корпуса.

    Конструктор получает готовые токены. Для исходных текстов используйте
    from_documents; исходные тексты и метаданные остаются в корпусе.
    """

    def __init__(
        self,
        tokenized_documents,
        *,
        k1=1.5,
        b=0.75,
        lemmatize=True,
        remove_stopwords=True,
    ):
        self._configure(k1, b, lemmatize, remove_stopwords)
        if not isinstance(tokenized_documents, Mapping):
            raise TypeError("Ожидается отображение doc_id → токены")
        lengths, postings = {}, defaultdict(list)
        for doc_id, tokens in tokenized_documents.items():
            _identifier(doc_id)
            frequencies = Counter(_tokens(tokens))
            lengths[doc_id] = sum(frequencies.values())
            for term, frequency in frequencies.items():
                postings[term].append((doc_id, frequency))
        self._set_statistics(lengths, postings)
        self._corpus_sha256 = None

    def _configure(self, k1, b, lemmatize, remove_stopwords):
        for name, value in (("k1", k1), ("b", b)):
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} должен быть числом")
            if not math.isfinite(value):
                raise ValueError(f"{name} должен быть конечным")
        if k1 < 0 or not 0 <= b <= 1:
            raise ValueError("Требуется k1 >= 0 и 0 <= b <= 1")
        if type(lemmatize) is not bool or type(remove_stopwords) is not bool:
            raise TypeError("Настройки предобработки должны быть bool")
        self._k1, self._b = float(k1), float(b)
        self._lemmatize, self._remove_stopwords = lemmatize, remove_stopwords

    def _set_statistics(self, lengths, postings):
        self._doc_lengths = MappingProxyType(dict(sorted(lengths.items())))
        self._postings = MappingProxyType(
            {term: tuple(sorted(entries)) for term, entries in sorted(postings.items())}
        )
        self._df = MappingProxyType({t: len(p) for t, p in self._postings.items()})
        self._avgdl = sum(lengths.values()) / len(lengths) if lengths else 0.0
        self._idf = MappingProxyType(
            {
                t: math.log1p((self.n - frequency + 0.5) / (frequency + 0.5))
                for t, frequency in self._df.items()
            }
        )

    @property
    def n(self):
        return len(self._doc_lengths)

    @property
    def avgdl(self):
        return self._avgdl

    @property
    def doc_lengths(self):
        return self._doc_lengths

    @property
    def postings(self):
        return self._postings

    @property
    def df(self):
        return self._df

    @property
    def idf(self):
        return self._idf

    @property
    def k1(self):
        return self._k1

    @property
    def b(self):
        return self._b

    @property
    def lemmatize(self):
        return self._lemmatize

    @property
    def remove_stopwords(self):
        return self._remove_stopwords

    @property
    def corpus_sha256(self):
        return self._corpus_sha256

    @classmethod
    def from_documents(cls, documents, **options):
        rows = _documents(documents)
        # Проверяем параметры до запуска лемматизации.
        index = cls({}, **options)
        tokens = {
            doc_id: preprocess(
                text, lemmatize=index.lemmatize, remove_stopwords=index.remove_stopwords
            )
            for doc_id, text in rows.items()
        }
        index = cls(tokens, **options)
        index._corpus_sha256 = _fingerprint(rows)
        return index

    def matches_documents(self, documents):
        return self.corpus_sha256 == _fingerprint(_documents(documents))

    def scores(self, tokens):
        """Положительные оценки кандидатов; повторы запроса имеют вес QTF."""
        query = Counter(_tokens(tokens))
        if not self.avgdl:
            return {}
        scores = defaultdict(float)
        # Стабильный порядок сложения независимо от порядка термов запроса.
        for term, query_frequency in sorted(query.items()):
            for doc_id, tf in self.postings.get(term, ()):
                normalization = (
                    1 - self.b + self.b * self.doc_lengths[doc_id] / self.avgdl
                )
                if self.k1 >= 1:
                    # Деление на k1 предотвращает переполнение при большом k1.
                    saturation = tf * (
                        (1 + 1 / self.k1) / (tf / self.k1 + normalization)
                    )
                else:
                    saturation = tf * (self.k1 + 1) / (tf + self.k1 * normalization)
                scores[doc_id] += query_frequency * self.idf[term] * saturation
        return dict(scores)

    def search_tokens(self, tokens, k=5):
        if type(k) is not int:
            raise TypeError("k должен быть целым числом")
        if k < 0:
            raise ValueError("k не может быть отрицательным")
        ranked = sorted(
            self.scores(tokens).items(), key=lambda item: (-item[1], item[0])
        )
        return [SearchResult(doc_id, score) for doc_id, score in ranked[:k]]

    def search(self, text, k=5):
        tokens = preprocess(
            text, lemmatize=self.lemmatize, remove_stopwords=self.remove_stopwords
        )
        return self.search_tokens(tokens, k)

    def save(self, path):
        """Атомарная запись JSON; статистика не требует повторной лемматизации."""
        payload = {
            "version": FORMAT_VERSION,
            "parameters": {"k1": self.k1, "b": self.b},
            "preprocess": {
                "lemmatize": self.lemmatize,
                "remove_stopwords": self.remove_stopwords,
            },
            "corpus_sha256": self.corpus_sha256,
            "doc_lengths": dict(self.doc_lengths),
            "postings": dict(self.postings),
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as stream:
                temporary = Path(stream.name)
                json.dump(
                    payload,
                    stream,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                stream.write("\n")
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, path):
        """Проверяет версию, TF, принадлежность документов и суммы длин."""
        with Path(path).open(encoding="utf-8") as stream:
            data = json.load(stream)
        expected = {
            "version",
            "parameters",
            "preprocess",
            "corpus_sha256",
            "doc_lengths",
            "postings",
        }
        if not isinstance(data, dict) or set(data) != expected:
            raise ValueError("Некорректная схема индекса")
        if type(data["version"]) is not int or data["version"] != FORMAT_VERSION:
            raise ValueError("Несовместимая версия индекса; пересоберите его")
        parameters, settings = data["parameters"], data["preprocess"]
        if not isinstance(parameters, dict) or set(parameters) != {"k1", "b"}:
            raise ValueError("Некорректные параметры индекса")
        if not isinstance(settings, dict) or set(settings) != {
            "lemmatize",
            "remove_stopwords",
        }:
            raise ValueError("Некорректные настройки предобработки")
        index = cls({}, **parameters, **settings)
        fingerprint = data["corpus_sha256"]
        if fingerprint is not None and (
            not isinstance(fingerprint, str)
            or len(fingerprint) != 64
            or any(c not in "0123456789abcdef" for c in fingerprint)
        ):
            raise ValueError("Некорректный отпечаток корпуса")
        lengths, postings = data["doc_lengths"], data["postings"]
        if not isinstance(lengths, dict) or not isinstance(postings, dict):
            raise ValueError("Некорректная структура индекса")
        for doc_id, length in lengths.items():
            _identifier(doc_id)
            if type(length) is not int or length < 0:
                raise ValueError("Длина документа должна быть целым числом >= 0")
        totals = Counter()
        checked = {}
        for term, entries in postings.items():
            list(_tokens([term]))
            if not isinstance(entries, list) or not entries:
                raise ValueError("Postings должны быть непустым списком")
            seen, checked[term] = set(), []
            for entry in entries:
                if not isinstance(entry, list) or len(entry) != 2:
                    raise ValueError("Некорректная запись posting")
                doc_id, tf = entry
                _identifier(doc_id)
                if doc_id not in lengths or doc_id in seen:
                    raise ValueError("Неизвестный или повторный документ в postings")
                if type(tf) is not int or tf <= 0:
                    raise ValueError("TF должен быть положительным целым числом")
                seen.add(doc_id)
                totals[doc_id] += tf
                checked[term].append((doc_id, tf))
        if any(totals[d] != length for d, length in lengths.items()):
            raise ValueError("Суммы TF не совпадают с длинами документов")
        index._set_statistics(lengths, checked)
        index._corpus_sha256 = fingerprint
        return index
