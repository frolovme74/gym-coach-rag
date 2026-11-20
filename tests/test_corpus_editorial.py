"""Регрессии найденных ошибок и сохранности корпуса при редактуре."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from corpus_build import merge_short_sections, partition, sentence_units, digest
from corpus_curate import checked_record
from corpus_editorial import edit_terms, method_section
from corpus_quality import quality_flags, units
from app.rag.preprocess import tokenize


class EditorialTest(unittest.TestCase):
    def test_range_is_not_joined_as_thousands(self):
        self.assertEqual(
            tokenize("3–500 мг; 3 500 мг"), ["3", "500", "мг", "3500", "мг"]
        )

    def test_typical_units_and_frequencies_are_equivalent(self):
        for en, ru in [
            ("24-h period", "24-часовой период"),
            ("3 g/kg/d", "3 г/кг/д"),
            ("200 kcals", "200 ккал"),
            ("3 days/week", "3 дня/неделя"),
        ]:
            with self.subTest(en=en):
                self.assertEqual(units(en), units(ru))

    def test_decades_and_years_are_not_seconds_or_grams(self):
        self.assertEqual(units("the 1930s and 2020 г."), {})
        self.assertEqual(units("40 г."), {"g": 1})

    def test_real_unit_and_negation_losses_remain_flagged(self):
        self.assertIn(
            "units_changed", quality_flags("A 5 s sprint.", "Спринт из 5 повторений.")
        )
        self.assertIn(
            "check_negation",
            quality_flags("There was no benefit.", "Обнаружена польза."),
        )
        self.assertIn("numbers_changed", quality_flags("400–800 ml/h.", "700 мл/ч."))

    def test_terminology_preserves_learning_and_weight_gain(self):
        text, _ = edit_terms(
            "Motor learning improves with training.", "Обучение движениям улучшается."
        )
        self.assertEqual(text, "Обучение движениям улучшается.")
        text, _ = edit_terms(
            "Sets may affect weight gain.", "Набор массы зависит от наборов упражнений."
        )
        self.assertIn("Набор массы", text)
        self.assertIn("подходов", text)

    def test_data_sets_are_not_workout_sets(self):
        text, _ = edit_terms(
            "A data set was analysed.", "Проанализирован набор данных."
        )
        self.assertEqual(text, "Проанализирован набор данных.")

    def test_cg_is_not_mislabelled_as_kilograms(self):
        text, _ = edit_terms("CG improved recovery.", "КГ улучшила восстановление.")
        self.assertEqual(text, "CG улучшила восстановление.")

    def test_stale_editorial_edits_are_rejected(self):
        record = {
            "en_sha256": digest("Original."),
            "machine_ru_sha256": digest("Перевод."),
        }
        checked_record(record, "Original.", "Перевод.")
        with self.assertRaises(ValueError):
            checked_record(record, "Changed.", "Перевод.")
        with self.assertRaises(ValueError):
            checked_record(record, "Original.", "Изменено.")

    def test_research_methods_are_separated_from_training_methods(self):
        self.assertTrue(method_section("2. Materials and Methods / Analysis"))
        self.assertFalse(method_section("Advanced Time-Saving Training Methods"))
        self.assertFalse(
            method_section(
                "Methods / Effects of sodium bicarbonate on training adaptations"
            )
        )
        self.assertFalse(
            method_section(
                "Methods / Optimal protocols of sodium bicarbonate supplementation / Sodium bicarbonate dose"
            )
        )

    def test_short_sections_merge_without_padding_or_loss(self):
        def paragraph(pid, n, section):
            return {
                "paragraph_id": pid,
                "section_en": section,
                "section_ru": section,
                "text_ru": " ".join(" ".join(["слово"] * 50) + "." for _ in range(n)),
            }

        p1, p2 = paragraph("p1", 2, "Introduction"), paragraph("p2", 5, "Discussion")
        groups, log = merge_short_sections(
            [
                {
                    "section_en": "Introduction",
                    "section_ru": "Введение",
                    "paragraphs": [p1],
                },
                {
                    "section_en": "Discussion",
                    "section_ru": "Обсуждение",
                    "paragraphs": [p2],
                },
            ]
        )
        self.assertEqual(len(log), 1)
        self.assertEqual(groups[0]["paragraphs"], [p1, p2])
        units = sentence_units(p1) + sentence_units(p2)
        self.assertEqual([u for c in partition(units) for u in c], units)

    def test_subminimum_document_is_reported_not_padded(self):
        p = {
            "paragraph_id": "p",
            "section_en": "Conclusion",
            "section_ru": "Выводы",
            "text_ru": "Краткий вывод.",
        }
        g, _ = merge_short_sections(
            [{"section_en": "Conclusion", "section_ru": "Выводы", "paragraphs": [p]}]
        )
        self.assertIsNone(partition(sentence_units(g[0]["paragraphs"][0])))


if __name__ == "__main__":
    unittest.main()
