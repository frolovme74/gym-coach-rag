"""Одинаковая предобработка русских документов и запросов для BM25.

Этот модуль не изменяет исходный текст для показа пользователю и не строит индекс.
"""

import re
import unicodedata
from functools import lru_cache

STOP_WORDS = frozenset(
    "и а но или в во на со к ко от для по при о об обо у из за как что это "
    "этот эта эти то же ли бы быть его ее их наш ваш который также чтобы".split()
)
PROTECTED_TOKENS = frozenset(
    "не ни нет без более менее до после г мг мкг кг мл л ккал кал см мм км "
    "м ч мин сек с сут rm bcaa eaa b12".split()
)
# В спортивном корпусе речь о пищевых белках, а не о животных «белках».
LEMMA_OVERRIDES = {"белки": "белок"}
CYRILLIC_WORD = re.compile(r"[а-я]+(?:-[а-я]+)*\Z")
TOKEN = re.compile(
    r"(?<![a-zа-я0-9])[-+]?\d+(?:[.,]\d+)?(?![a-zа-я0-9])"
    r"|[a-zа-я0-9]+(?:-[a-zа-я0-9]+)*"
)
ATTACHED_UNIT = re.compile(
    r"(?<![a-zа-я0-9])(\d+(?:[.,]\d+)?)(ккал|мкг|кг|мг|мл|см|мм|км|г|л|kg|mg|ml|g)\b"
)


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("text должен быть строкой")
    text = unicodedata.normalize("NFKC", text).lower().replace("ё", "е")
    text = text.translate(str.maketrans({"‐": "-", "‑": "-", "−": "-"}))
    # До разбиения диапазонов: иначе «3–500» превратилось бы в 3500.
    text = re.sub(
        r"\b(\d{1,3}(?:[ \u00a0\u202f]\d{3})+)(?!\d)",
        lambda match: re.sub(r"\s", "", match[1]),
        text,
    )
    text = re.sub(r"(?<=\d)\s*[–—-]\s*(?=\d)", " ", text)
    # В словах типографский дефис равнозначен обычному; тире между словами — граница.
    text = re.sub(r"(?<=[a-zа-я])[–—](?=[a-zа-я0-9])", "-", text)
    text = ATTACHED_UNIT.sub(r"\1 \2", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    return [
        m[0].replace(",", ".").lstrip("+") for m in TOKEN.finditer(normalize_text(text))
    ]


@lru_cache(maxsize=1)
def _analyzer():
    try:
        from pymorphy3 import MorphAnalyzer
    except ImportError as exc:
        raise RuntimeError(
            "Для лемматизации установите зависимости requirements-rag.txt "
            "или используйте lemmatize=False."
        ) from exc
    return MorphAnalyzer()


@lru_cache(maxsize=50000)
def _lemma(token: str) -> str:
    if token in PROTECTED_TOKENS or not CYRILLIC_WORD.fullmatch(token):
        return token
    if token in LEMMA_OVERRIDES:
        return LEMMA_OVERRIDES[token]
    if "-" in token:
        parts = token.split("-")
        return "-".join(
            part if i < len(parts) - 1 and part.endswith("о") else _lemma(part)
            for i, part in enumerate(parts)
        )
    return _analyzer().parse(token)[0].normal_form


def preprocess(
    text: str, *, lemmatize: bool = True, remove_stopwords: bool = True
) -> list[str]:
    result = []
    for token in tokenize(text):
        if remove_stopwords and token in STOP_WORDS and token not in PROTECTED_TOKENS:
            continue
        term = _lemma(token) if lemmatize else token
        if remove_stopwords and term in STOP_WORDS and term not in PROTECTED_TOKENS:
            continue
        result.append(term)
    return result
