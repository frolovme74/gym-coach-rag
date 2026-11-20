"""Проверки потерь текста, очистки и числовой точности перевода."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from corpus_build import partition
from corpus_prepare import extract_article, word_count
from corpus_quality import numbers, quality_flags


class CorpusPreparationTest(unittest.TestCase):
    def test_reference_number_removed_but_numeric_dose_preserved(self):
        xml = b"""<article><front><article-meta><permissions><copyright-statement>Authors</copyright-statement></permissions></article-meta></front><body><sec><title>Protein</title><p>Use 1.6 g/kg in this research <xref ref-type="bibr">42</xref> while preserving the comparison.</p></sec></body><back><ref-list><ref>Bibliography</ref></ref-list></back></article>"""
        data = extract_article(xml, "NU-test")
        text = data["paragraphs"][0]["text_en"]
        self.assertIn("1.6 g/kg", text)
        self.assertIn("while preserving", text)
        self.assertNotIn("42", text)
        self.assertNotIn("Bibliography", text)

    def test_noise_and_out_of_scope_are_logged(self):
        xml = b"""<article><body><sec><title>Funding</title><p>This study was funded by a grant.</p></sec><sec><title>Discussion</title><p>This section discusses doping in competitive sport.</p><p>Training frequency may affect strength in healthy adults.</p></sec></body></article>"""
        data = extract_article(xml, "TR-test")
        self.assertEqual(len(data["paragraphs"]), 1)
        self.assertEqual(len(data["excluded"]), 2)

    def test_table_not_flattened_into_paragraph(self):
        xml = b"""<article><body><sec><title>Results</title><table-wrap><caption><p>Do not mix these numbers into prose.</p></caption></table-wrap><p>The narrative conclusion remains in the extracted text.</p></sec></body></article>"""
        self.assertEqual(len(extract_article(xml, "TR-test")["paragraphs"]), 1)

    def test_list_items_have_word_boundaries(self):
        xml = b"""<article><body><p><list><list-item><p>The first complete finding.</p></list-item><list-item><p>The second complete finding.</p></list-item></list></p></body></article>"""
        text = extract_article(xml, "TR-test")["paragraphs"][0]["text_en"]
        self.assertIn("finding. The second", text)

    def test_decimal_separator_and_range(self):
        self.assertEqual(
            numbers("1.6 g/kg and 3–5 sets"), numbers("1,6 г/кг и 3–5 подходов")
        )

    def test_numeric_dose_change_is_flagged(self):
        self.assertIn(
            "numbers_changed",
            quality_flags(
                "Take 3 g in this protocol.", "В этом протоколе принимают 5 г."
            ),
        )

    def test_thousands_and_small_decimal_are_distinct(self):
        self.assertEqual(numbers("200,000 downloads"), numbers("200 000 скачиваний"))
        self.assertEqual(numbers("0.001 g"), numbers("0,001 г"))
        self.assertNotEqual(numbers("1.000 g"), numbers("1 000 г"))

    def test_hyphenated_duration_is_preserved(self):
        self.assertNotIn(
            "units_changed",
            quality_flags("A 20–90-min nap.", "Сон длительностью 20–90 минут."),
        )

    def test_signed_numbers_and_spelled_units(self):
        self.assertEqual(numbers("from − 3.0 to 15.9%"), numbers("от −3,0 до 15,9%"))
        self.assertNotIn(
            "units_changed",
            quality_flags(
                "3 mg/kg, e.g. in adults.",
                "3 миллиграмма на килограмм, например у взрослых.",
            ),
        )

    def test_unit_change_is_flagged(self):
        self.assertIn(
            "units_changed",
            quality_flags("A dose of 3 mg was studied.", "Изучали дозу 3 г."),
        )

    def test_dose_frequency_change_is_flagged(self):
        self.assertIn("units_changed", quality_flags("3 g/kg/day.", "3 г/кг/неделю."))

    def test_ambiguous_numeric_notation_does_not_crash(self):
        self.assertIn(
            "check_numeric_notation",
            quality_flags("Software 4.0.4 was used.", "Использована программа 4.0.4."),
        )

    def test_negation_loss_is_flagged(self):
        self.assertIn(
            "check_negation",
            quality_flags("There was no benefit.", "Было обнаружено преимущество."),
        )

    def test_empty_translation_is_flagged(self):
        self.assertIn(
            "empty_translation", quality_flags("A complete source sentence.", "")
        )

    def test_chunk_boundaries_do_not_drop_short_tail(self):
        units = [
            {"text": " ".join(["слово"] * 90) + ".", "paragraph_id": str(i)}
            for i in range(11)
        ]
        chunks = partition(units)
        self.assertIsNotNone(chunks)
        self.assertEqual([u for c in chunks for u in c], units)
        self.assertTrue(
            all(300 <= sum(word_count(u["text"]) for u in c) <= 600 for c in chunks)
        )

    def test_short_section_is_not_padded_or_duplicated(self):
        self.assertIsNone(
            partition([{"text": " ".join(["слово"] * 120), "paragraph_id": "p1"}])
        )

    def test_numbers_cannot_inflate_chunk_to_minimum(self):
        # Decimal parts may count as separate lexical tokens, but not words.
        self.assertIsNone(
            partition([{"text": " ".join(["1.6"] * 200), "paragraph_id": "p1"}])
        )


if __name__ == "__main__":
    unittest.main()
