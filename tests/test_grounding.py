"""Контракт источников и дозировки из конкретных проверенных предложений."""

import unittest
from unittest.mock import Mock
from app.llm import LLMError, LLMResponse
from app.rag.pipeline import RAGPipeline, Evidence
from app.rag.prompts import passages


class V5Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = RAGPipeline.from_project(client=Mock())

    def test_creatine_question_selects_usage_card(self):
        sources = self.p.retrieve("как пить креатин?")
        self.assertEqual("guide_creatine_use", sources[0]["doc_id"])
        self.assertGreater(sources[0]["score"], 0)
        self.assertTrue(sources[0]["references"])

    def test_general_dose_requires_exact_reviewed_evidence(self):
        source = self.p.source("guide_creatine_use", 5)
        quote = source["_general_dose_passages"][0]
        n = passages(source["fragment"]).index(quote) + 1
        claim = dict(
            text="Для здоровых взрослых в справке описаны 3–5 г в день.",
            evidence=[{"source": 1, "passages": [n]}],
        )
        e = Evidence([source])
        self.assertIn("[1]", e.claims([claim]))
        self.assertNotIn("_general_dose_passages", list(e.used.values())[0])
        for bad in ["30 г в день", "3–5 мг в день"]:
            with self.assertRaises(LLMError):
                Evidence([source]).claims([{**claim, "text": bad}])
        unreviewed = {**source, "_general_dose_passages": []}
        with self.assertRaises(LLMError):
            Evidence([unreviewed]).claims([claim])

    def test_program_has_no_invented_bm25_score(self):
        self.p._client.generate.return_value = LLMResponse(
            '{"found":false}', "test", {}
        )
        value = self.p.program(
            dict(goal="endurance", level="intermediate", days=5, equipment="dumbbells")
        )
        self.assertTrue(value["found"])
        for s in value["sources"]:
            self.assertIsNone(s["score"])
            self.assertEqual("parameters", s["selection"])
            self.assertNotIn("_general_dose_passages", s)
            self.assertTrue(all(q in s["fragment"] for q in s["quotes"]))

    def test_general_caffeine_card_does_not_override_habituation_question(self):
        self.assertNotIn(
            "guide_caffeine",
            self.p.index.preferred_ids(
                "Перестанет ли кофеин помогать из-за привыкания?"
            ),
        )
