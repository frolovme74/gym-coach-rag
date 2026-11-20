"""Приоритет памяток, границы поиска и защита согласованности метаданных."""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app.llm import LLMError
from app.rag.pipeline import RAGPipeline
from app.rag.runtime_retrieval import RuntimeRetrieval


class RuntimeRetrievalTest(unittest.TestCase):
    def setUp(self):
        self.base = Mock(n=3)
        self.base.search.return_value = [
            SimpleNamespace(doc_id=d, score=s)
            for d, s in [("article", 20), ("guide", 5)]
        ]
        self.index = RuntimeRetrieval(
            self.base,
            [
                {"doc_id": "guide", "answer_patterns": ["креатин"]},
                {"doc_id": "absent", "answer_patterns": ["креатин"]},
            ],
        )

    def test_priority_preserves_scores_and_never_invents_hits(self):
        hits = self.index.search("Нужен ли креатин?", 3)
        self.assertEqual(["guide", "article"], [h.doc_id for h in hits])
        self.assertEqual([5, 20], [h.score for h in hits])
        self.assertEqual(["article"], [h.doc_id for h in self.index.search("Сон", 1)])

    def test_k_contract(self):
        self.assertEqual([], self.index.search("креатин", 0))
        for k in [True, 2.5, "2"]:
            with self.assertRaises(TypeError):
                self.index.search("креатин", k)
        with self.assertRaises(ValueError):
            self.index.search("креатин", -1)

    def test_short_faq_context_is_focused_on_matching_card(self):
        p = RAGPipeline.from_project(client=Mock())
        for query, did in [
            ("Нужен ли креатин?", "guide_creatine"),
            ("Сколько белка нужно в день?", "guide_protein"),
            ("Сколько нужно спать?", "guide_sleep"),
        ]:
            self.assertEqual([did], [s["doc_id"] for s in p.retrieve(query)])
        self.assertEqual(5, len(p.retrieve("Как периодизация влияет на рост силы?")))

    def test_changed_manifest_fails_before_generation(self):
        from pathlib import Path

        read = Path.read_text

        def changed(path, *args, **kwargs):
            value = read(path, *args, **kwargs)
            if path.name == "manifest.json" and path.parent.name == "corpus-v4":
                data = json.loads(value)
                data[-1]["program"] = {"tampered": True}
                return json.dumps(data)
            return value

        client = Mock()
        with patch.object(Path, "read_text", changed):
            with self.assertRaises(LLMError) as ctx:
                RAGPipeline.from_project(client=client)
        self.assertEqual("config", ctx.exception.code)
        client.generate.assert_not_called()
