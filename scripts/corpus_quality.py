"""Проверки сохранности перевода: числа, единицы и текст. Без моделей и загрузок."""

import re
from collections import Counter
from decimal import Decimal, InvalidOperation

NUMBER = re.compile(r"(?<!\w)[−–-]?\d+(?:[.,]\d+)*(?:[eE][+-]?\d+)?")


def canonical_number(value):
    normalized = value.replace(",", ".").replace("−", "-").replace("–", "-")
    try:
        return str(Decimal(normalized).normalize())
    except InvalidOperation:
        # Examples: section labels 2.1.3 and comma-separated lists.
        return "ambiguous:" + normalized


def numbers(text, language=None):
    # English comma groups and Russian spaces are thousands separators;
    # Russian commas (including 0,001) remain decimal separators.
    language = language or ("ru" if re.search(r"[А-Яа-яЁё]", text) else "en")
    if language == "en":
        text = re.sub(
            r"(?<![\d.,])\d{1,3}(?:,\d{3})+(?!\d)",
            lambda m: m[0].replace(",", ""),
            text,
        )
    else:
        text = re.sub(
            r"(?<![\d.,])\d{1,3}(?:[ \u00a0\u202f]\d{3})+(?!\d)",
            lambda m: re.sub(r"\s", "", m[0]),
            text,
        )
    text = re.sub(r"(?<=\d)[–—−-](?=\d)", " to ", text)
    text = re.sub(r"[−-]\s+(?=\d)", "-", text)
    return Counter(canonical_number(x) for x in NUMBER.findall(text))


def units(text):
    text = re.sub(r"\be\.\s*g\.", "for example", text, flags=re.I)
    # Статистическая g Хеджеса и русский год не являются граммами.
    text = re.sub(
        r"\b(?:Hedge'?s?|Хеджеса)\s+g\b|\b[gг]\s*(?=[=<>])", "effect ", text, flags=re.I
    )
    text = re.sub(r"\b((?:19|20)\d{2})\s*г\.(?=\s|$)", r"\1 year", text)
    text = re.sub(r"\b(?:19|20)\d0s\b", "decade", text)
    text = re.sub(r"\bkcals\b", "kcal", text, flags=re.I)
    for name, pattern in [
        ("mg", r"миллиграмм\w*|milligrams?"),
        ("kg", r"килограмм\w*|kilograms?"),
        ("g", r"грамм\w*|grams?"),
    ]:
        text = re.sub(r"\b(?:" + pattern + r")\b", name, text, flags=re.I)
    mapping = {
        "kg": "kg",
        "кг": "kg",
        "mg": "mg",
        "мг": "mg",
        "g": "g",
        "г": "g",
        "ml": "ml",
        "мл": "ml",
        "kcal": "kcal",
        "ккал": "kcal",
        "μg": "mcg",
        "µg": "mcg",
        "mcg": "mcg",
        "мкг": "mcg",
    }
    found = Counter(
        mapping[u.lower()]
        for u in re.findall(
            r"(?<![A-Za-zА-Яа-яЁё])(?:kg|mg|g|ml|kcal|μg|µg|mcg|кг|мг|г|мл|ккал|мкг)(?![A-Za-zА-Яа-яЁё])",
            text,
            re.I,
        )
    )
    for canonical, pattern in [
        ("hour", r"h|hr|hrs|hours?|ч|час\w*"),
        ("minute", r"min|mins|minutes?|мин|минут\w*"),
        ("second", r"s|sec|seconds?|(?-i:с)|сек|секунд\w*"),
    ]:
        found[canonical] += len(
            re.findall(
                r"\d[\s–—-]*(?:" + pattern + r")(?![A-Za-zА-Яа-яЁё])", text, re.I
            )
        )
    for canonical, terms in [
        ("per_day", r"days?|d|д|день|сутки|суток|сут"),
        ("per_week", r"weeks?|wk|неделю|недели|неделя|нед"),
    ]:
        found[canonical] += len(
            re.findall(
                r"(?:/\s*|\bper\s+|\bevery\s+|\beach\s+|\ba\s+|\bв\s+|\bза\s+)(?:"
                + terms
                + r")(?![A-Za-zА-Яа-яЁё])",
                text,
                re.I,
            )
        )
    found["per_week"] += len(re.findall(r"\b(?:weekly|еженедельн\w*)\b", text, re.I))
    return +found


def quality_flags(original, translated):
    flags = []
    if any(
        k.startswith("ambiguous:")
        for k in (*numbers(original, "en"), *numbers(translated, "ru"))
    ):
        flags.append("check_numeric_notation")
    if numbers(original, "en") != numbers(translated, "ru"):
        flags.append("numbers_changed")
    if units(original) != units(translated):
        flags.append("units_changed")
    for symbol, wording in [
        ("≥", r"≥|>=|не менее|как минимум|по меньшей мере"),
        ("≤", r"≤|<=|не более|максимум"),
    ]:
        if symbol in original and not re.search(wording, translated, re.I):
            flags.append("check_comparison_bound")
    if not translated.strip():
        flags.append("empty_translation")
    if "�" in translated or "<unk>" in translated:
        flags.append("unknown_token")
    if len(translated) < len(original) * 0.45 or len(translated) > len(original) * 2.4:
        flags.append("length_ratio")
    if len(re.findall(r"[А-Яа-яЁё]", translated)) < len(
        re.findall(r"[A-Za-z]", translated)
    ):
        flags.append("mostly_not_russian")
    words = translated.lower().split()
    ngrams = Counter(tuple(words[i : i + 5]) for i in range(max(0, len(words) - 4)))
    if any(n > 3 for n in ngrams.values()):
        flags.append("repetition")
    if bool(
        re.search(r"\b(no|not|without|neither|unlikely|inconclusive)\b", original, re.I)
    ) and not re.search(
        r"\b(не|нет|без|ни|нельзя|отсутств\w*|недостат\w*|неяс\w*|неоднознач\w*|маловероят\w*)\b",
        translated,
        re.I,
    ):
        flags.append("check_negation")
    return flags
