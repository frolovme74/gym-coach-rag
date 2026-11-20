"""Совместимая команда редакторской сборки без вызовов моделей перевода."""

import argparse
import re


def terminology_flags(en, ru):
    """Эвристики отмечают риск, а не заменяют сопоставление смысла."""
    rules = [
        (
            r"train",
            r"обучен|обуча|подготовк[аи] сопротивления|резистентност",
            "check_training_term",
        ),
        (r"\bsets?\b", r"набор|комплект", "check_set_term"),
        (r"failure", r"сбо[йя]|неисправност|недостаточност", "check_failure_term"),
        (r"joint", r"многослой", "check_joint_term"),
        (r"\brows?\b", r"ряд[ыаов]*", "check_exercise_name"),
        (r"position stand", r"стенд|позиционн", "check_position_stand"),
        (r"performance", r"производительност", "check_performance_wording"),
        (r"secondary to", r"вторичн", "check_causal_statement"),
    ]
    return [
        flag
        for a, b, flag in rules
        if re.search(a, en, re.I) and re.search(b, ru, re.I)
    ]


def main():
    from corpus_curate import curate
    from corpus_build import build

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    curate()
    if args.build:
        build()


if __name__ == "__main__":
    main()
