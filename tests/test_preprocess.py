import subprocess
import sys
import unittest
from pathlib import Path

from app.rag.preprocess import normalize_text, preprocess, tokenize


class PreprocessTest(unittest.TestCase):
    def test_case_yo_unicode_and_space(self):
        self.assertEqual(normalize_text("  ОБЪЁМ\u00a0\n ВЕСОВ  "), "объем весов")
        self.assertEqual(
            tokenize("В12 B１２, BCAA омега‑3"), ["в12", "b12", "bcaa", "омега-3"]
        )

    def test_decimals_ranges_and_negative_numbers(self):
        self.assertEqual(
            tokenize("1,6–2.2 г/кг; −0,5; 3-5; +2,5"),
            ["1.6", "2.2", "г", "кг", "-0.5", "3", "5", "2.5"],
        )

    def test_numbers_not_lost_and_units_split(self):
        self.assertEqual(
            tokenize("2 000 мг, 3г, 1,6г/кг, B12"),
            ["2000", "мг", "3", "г", "1.6", "г", "кг", "b12"],
        )

    def test_empty_and_wrong_types(self):
        for text in ["", " \n ", "!…—"]:
            self.assertEqual(preprocess(text), [])
        for value in [None, 12, [], b"text"]:
            with self.assertRaises(TypeError):
                preprocess(value)

    def test_morphology_connects_query_and_document(self):
        self.assertEqual(
            preprocess("мышечными подходами тренировок"),
            ["мышечный", "подход", "тренировка"],
        )
        self.assertEqual(preprocess("белки белка белков"), ["белок"] * 3)

    def test_order_and_tf_multiplicity_preserved(self):
        self.assertEqual(
            preprocess("Подход подхода подходами отдых"),
            ["подход", "подход", "подход", "отдых"],
        )

    def test_terms_units_and_negations_survive(self):
        tokens = preprocess(
            "не ни нет без более менее до после 3 г 2 мг с BCAA EAA B12 омега-3"
        )
        self.assertEqual(
            tokens,
            "не ни нет без более менее до после 3 г 2 мг с bcaa eaa b12 омега-3".split(),
        )

    def test_ablation_options_are_independent(self):
        self.assertEqual(preprocess("И подходами", lemmatize=False), ["подходами"])
        self.assertEqual(
            preprocess("И подходами", remove_stopwords=False), ["и", "подход"]
        )
        self.assertEqual(
            preprocess("И подходами", lemmatize=False, remove_stopwords=False),
            ["и", "подходами"],
        )

    def test_cyrillic_hyphenated_words(self):
        self.assertEqual(preprocess("мышечно-сухожильными"), ["мышечно-сухожильный"])

    def test_no_lemma_mode_does_not_import_morphology(self):
        code = "from app.rag.preprocess import preprocess; import sys; assert preprocess('подходами BCAA', lemmatize=False)==['подходами','bcaa']; assert 'pymorphy3' not in sys.modules"
        subprocess.run(
            [sys.executable, "-c", code],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
        )


if __name__ == "__main__":
    unittest.main()
