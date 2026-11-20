"""Редактура не перезаписывает исходник и требует точной версии текста."""

import copy
import unittest

from scripts.corpus_revision import revise


class CorpusRevisionTests(unittest.TestCase):
    def fixture(self):
        docs = [
            {
                "doc_id": "d1",
                "topic": "recovery",
                "source": "Review",
                "section": "Cooling",
                "text": "Активное охлаждение.",
            }
        ]
        manifest = [{"doc_id": "d1", "units": [{"text_sha256": "old"}]}]
        patches = [
            {
                "doc_id": "d1",
                "old_text": "Активное охлаждение.",
                "new_text": "Активная заминка.",
                "new_section": "Заминка",
                "reason": "cool-down means low-intensity movement",
                "provenance": [{"text_en": "active cool-down", "paragraph_id": "p1"}],
            }
        ]
        return docs, manifest, patches

    def test_parent_is_untouched_and_old_units_are_explicit(self):
        args = self.fixture()
        before = copy.deepcopy(args)
        docs, manifest = revise(*args)
        self.assertEqual(args, before)
        self.assertEqual(docs[0]["text"], "Активная заминка.")
        self.assertEqual(docs[0]["topic"], "recovery")
        self.assertNotIn("units", manifest[0])
        self.assertEqual(
            manifest[0]["text_sha256"], manifest[0]["revision"]["text_sha256"]
        )
        self.assertFalse(manifest[0]["full_semantic_review"])
        self.assertEqual(manifest[0]["parent_units"], before[1][0]["units"])
        self.assertNotEqual(
            manifest[0]["revision"]["text_sha256"],
            manifest[0]["revision"]["parent_text_sha256"],
        )

    def test_stale_or_duplicate_edits_are_rejected(self):
        args = self.fixture()
        args[2][0]["old_text"] = "другая версия"
        with self.assertRaisesRegex(ValueError, "Устаревший"):
            revise(*args)
        args = self.fixture()
        args[2].append(copy.deepcopy(args[2][0]))
        with self.assertRaisesRegex(ValueError, "повторный"):
            revise(*args)

    def test_empty_translation_or_missing_source_rejected(self):
        for field, value in [("new_text", " "), ("provenance", [])]:
            args = self.fixture()
            args[2][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                revise(*args)


if __name__ == "__main__":
    unittest.main()
