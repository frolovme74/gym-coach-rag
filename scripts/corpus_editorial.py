"""Воспроизводимая редактура по английскому контексту. Переводчики не запускаются.

Правила исправляют только перечисленные термины и не служат доказательством
полной смысловой проверки абзаца. Все применения записываются в журнал.
"""

import re

METHOD_SECTION = re.compile(
    r"^(?:\d+[. ]*)?(?:methods?|materials and methods|statistical analys|data analys|"
    r"risk of bias|search strategy|study selection|quality assessment|methodological quality)(?:\b|$)",
    re.I,
)


def method_section(section):
    # В JATS некоторых позиций ISSN содержательные главы вложены в Methods.
    # Сохраняем их: принадлежность XML-родителю не определяет смысл раздела.
    parts = section.split(" / ")
    if (
        len(parts) > 1
        and parts[0] == "Methods"
        and re.match(
            r"History of research|Mechanisms for|Effects of|Sex-specific effects|"
            r"Training status and|Optimal protocols of|Side-effects associated|"
            r"Placebo effects associated|Interaction of|Position of the International",
            parts[1],
            re.I,
        )
    ):
        return False
    return bool(METHOD_SECTION.search(section.split(" / ")[0]))


def edit_terms(en, text):
    changes = []

    def sub(name, pattern, replacement, flags=re.I):
        nonlocal text

        def replace(match):
            result = (
                replacement(match)
                if callable(replacement)
                else match.expand(replacement)
            )
            if match[0][:1].isupper() and not match[0].isupper():
                result = result[:1].upper() + result[1:]
            if result != match[0]:
                changes.append({"rule": name, "before": match[0], "after": result})
            return result

        text = re.sub(pattern, replace, text, flags=flags)

    if re.search(r"\bperformance\b", en, re.I):
        sub("performance", r"производительност", "работоспособност")
    if re.search(r"\bROM\b|range of motion", en, re.I):
        sub("range_of_motion", r"\bПЗУ\b", "ROM", flags=0)
    if re.search(r"\bCGs?\b", en):
        # CG обозначает одежду или контрольную группу; не превращаем в килограммы.
        sub("source_abbreviation_CG", r"\bКГ\b", "CG", flags=0)
    if re.search(r"Hedge", en, re.I):
        sub("hedges_g", r"\b[gг]\s+(?:Хеджеса|Hedges)", "g Хеджеса")
    if re.search(r"\btraining\b|\btrained\b", en, re.I) and not re.search(
        r"learn|educat|teaching|instruction", en, re.I
    ):
        sub("trained", r"\bнеобученн", "нетренированн")
        sub("trained", r"\bобученн", "тренированн")
        sub(
            "trained_resistance",
            r"тренированн(\w*)\s+(?:сопротивлению|резистентности)",
            r"тренированн\1 в силовых упражнениях",
        )
        for old, new in [
            ("статуса", "уровня"),
            ("статусу", "уровню"),
            ("статусом", "уровнем"),
            ("статусе", "уровне"),
            ("статус", "уровень"),
        ]:
            sub("training_status", rf"\b{old} обучения", new + " тренированности")
        # Согласование при замене среднего рода «обучение» на женский «тренировка».
        adjectives = r"(?:силов|интенсивн|одновременн|обычн|традиционн|регулярн|аэробн|физическ|длительн|комплексн|совместн|резистентн|разгрузочн|дополнительн|сопротивляющ)"
        for old, new in [
            ("ое", "ая"),
            ("ого", "ой"),
            ("ому", "ой"),
            ("ым", "ой"),
            ("ом", "ой"),
        ]:
            sub(
                "training_agreement",
                rf"\b({adjectives}){old}(?=\s+обучени)",
                rf"\1{new}",
            )
        for old, new in [
            ("обучением", "тренировкой"),
            ("обучению", "тренировке"),
            ("обучении", "тренировке"),
            ("обучения", "тренировки"),
            ("обучение", "тренировка"),
        ]:
            sub("training", rf"\b{old}\b", new)
        sub(
            "resistance_training",
            r"тренировк(\w*)\s+(?:сопротивления|резистентности|устойчивости|сопротивлению)",
            r"тренировк\1 с отягощениями",
        )
        if re.search(r"machine.based", en, re.I):
            for old, new in [
                ("машинного обучения", "занятий на тренажёрах"),
                ("машинным обучением", "занятиями на тренажёрах"),
                ("машинное обучение", "занятия на тренажёрах"),
                ("машинной тренировки", "тренировки на тренажёрах"),
                ("машинная тренировка", "тренировка на тренажёрах"),
                ("машинной тренировкой", "тренировкой на тренажёрах"),
            ]:
                sub("machine_training", old, new)
    if re.search(r"\bsets?\b", en, re.I):
        for old, new in [
            ("наборы капель", "дроп-сеты"),
            ("наборов капель", "дроп-сетов"),
            ("наборами капель", "дроп-сетами"),
            ("набора капель", "дроп-сета"),
            ("набор капель", "дроп-сет"),
        ]:
            sub("drop_sets", old, new)
        if not re.search(
            r"\bset of (?:studies|criteria|data|guidelines|recommendations)|data\s*sets?",
            en,
            re.I,
        ):
            # Набор массы/веса/данных не является упражнением.
            sub(
                "exercise_sets",
                r"\b(?:набор|комплект)(ы|ов|ам|ами|ах|а|у|ом)?\b(?!\s+(?:данн|вес|масс|критер|участ|доказ|поиск|питател|рекомендац))",
                r"подход\1",
            )
    if re.search(r"muscul(?:ar|e) failure|momentary failure", en, re.I):
        for old, new in [
            ("при мышечной недостаточности", "при мышечном отказе"),
            ("мышечная недостаточность", "мышечный отказ"),
            ("мышечной недостаточности", "мышечного отказа"),
            ("мышечную недостаточность", "мышечный отказ"),
            ("мышечной недостаточностью", "мышечным отказом"),
        ]:
            sub("muscle_failure", old, new)
    if re.search(r"multi.joint", en, re.I):
        sub("multi_joint", r"многослой", "многосустав")
    if re.search(r"position stand", en, re.I):
        sub("position_stand", r"позиционный стенд", "официальная позиция")
        sub("position_stand", r"позиционного стенда", "официальной позиции")
    # Восстанавливаем привычную запись частоты приёма, не меняя количества.
    sub("per_day", r"/\s*д\b", "/сут")
    # Удаляются только скобки с перечнем авторов, без чисел или содержательных данных.
    for m in list(re.finditer(r"\([^()]*\)", text))[::-1]:
        fragment = m[0]
        if re.search(r"et al|и др\.", fragment) and not re.search(
            r"\d|=|%|\b(?:when|after|before|при|после|группа)\b", fragment, re.I
        ):
            changes.append(
                {"rule": "inline_author_list", "before": fragment, "after": ""}
            )
            text = text[: m.start()] + text[m.end() :]
    text = re.sub(r"\(\s*[,;]?\s*\)", "", text)
    text = re.sub(r"\s+([,.])", r"\1", text)
    text = re.sub(r" {2,}", " ", text).strip()
    return text, changes
