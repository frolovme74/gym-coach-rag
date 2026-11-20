"""Покрытие параметров программы, оборудования и происхождения данных."""

import itertools
import json
import unittest
from unittest.mock import Mock

from app.llm import LLMError, LLMResponse
from app.rag.pipeline import EQUIPMENT, GOALS, LEVELS, RAGPipeline
from app.rag.planner import prepare, valid_days


class PlannerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = Mock()
        cls.pipeline = RAGPipeline.from_project(client=cls.client)

    def setUp(self):
        self.client.reset_mock(side_effect=True)
        self.client.generate.return_value = LLMResponse('{"found":false}', "test", {})

    def test_all_300_combinations_are_complete_and_grounded(self):
        for goal, level, days, equipment in itertools.product(
            GOALS, LEVELS, range(2, 6), EQUIPMENT
        ):
            params = dict(goal=goal, level=level, days=days, equipment=equipment)
            with self.subTest(**params):
                result = self.pipeline.program(params)
                self.assertTrue(result["found"])
                self.assertEqual(days, len(result["days"]))
                catalog, rule, rule_doc, _, draft, _ = prepare(self.pipeline, params)
                self.assertTrue(result["note"].startswith(LEVELS[level]))
                self.assertIn(f"{rule['sets']} × {rule['reps']}", result["note"])
                self.assertIn(rule["rest"], result["note"])
                self.assertTrue(valid_days({"days": draft}, catalog, params))
                self.assertTrue(
                    all(equipment in e["equipment"] for e in catalog.values())
                )
                allowed_names = {e["name"] for e in catalog.values()}
                for i, day in enumerate(result["days"]):
                    names = [e["name"] for e in day["exercises"]]
                    self.assertEqual(len(names), len(set(names)))
                    self.assertGreaterEqual(len(names), 4)
                    self.assertTrue(set(names) <= allowed_names)
                    for ex in day["exercises"]:
                        self.assertEqual(
                            1 if days == 5 and i == 4 else rule["sets"], ex["sets"]
                        )
                        self.assertEqual(rule["reps"], ex["reps"])
                        evidence = " ".join(
                            result["sources"][n - 1]["fragment"]
                            for n in ex["citations"]
                        )
                        self.assertIn(ex["name"], evidence)
                for source in result["sources"]:
                    doc_id = source["doc_id"]
                    card = self.pipeline.program_cards.get(doc_id)
                    if card is None:
                        card = self.pipeline.documents[doc_id]
                    self.assertEqual(source["fragment"], card["text"])
                    self.assertTrue(source["quotes"])
                    self.assertTrue(
                        all(q in source["fragment"] for q in source["quotes"])
                    )
                    if doc_id == rule_doc and "guidance" in rule:
                        self.assertIn(rule["guidance"], source["quotes"])
                        self.assertIn(rule["guidance"], result["note"])

    def test_load_changes_with_goal_and_level(self):
        for goal in GOALS:
            with self.subTest(goal=goal):
                profiles = []
                for level in LEVELS:
                    params = dict(goal=goal, level=level, days=3, equipment="gym")
                    _, rule, _, _, _, _ = prepare(self.pipeline, params)
                    profiles.append((rule["sets"], rule["reps"], rule["rest"]))
                self.assertEqual(3, len(set(profiles)))

    def test_program_extension_does_not_enter_question_index(self):
        self.assertTrue(self.pipeline.program_cards)
        self.assertEqual(len(self.pipeline.documents), self.pipeline.index.n)
        self.assertFalse(
            self.pipeline.documents.keys() & self.pipeline.program_cards.keys()
        )

    def test_main_exercises_change_with_level_for_every_equipment(self):
        for equipment in EQUIPMENT:
            with self.subTest(equipment=equipment):
                names_by_level = []
                for level in LEVELS:
                    params = dict(goal="muscle", level=level, days=3, equipment=equipment)
                    catalog, _, _, _, _, _ = prepare(self.pipeline, params)
                    names_by_level.append({
                        ex["name"] for ex in catalog.values()
                        if ex["pattern"] in {"knee", "push"}
                    })
                for first, second in itertools.combinations(names_by_level, 2):
                    self.assertTrue(first.isdisjoint(second), first & second)

    def test_model_can_select_valid_alternative(self):
        params = dict(goal="muscle", level="beginner", days=2, equipment="gym")
        catalog, _, _, _, draft, _ = prepare(self.pipeline, params)
        draft[0]["exercise_ids"][1] = "db_rdl"
        self.assertTrue(valid_days({"days": draft}, catalog, params))
        self.client.generate.return_value = LLMResponse(
            json.dumps({"found": True, "days": draft}), "test", {}
        )
        result = self.pipeline.program(params)
        self.assertEqual(
            "Румынская тяга с гантелями", result["days"][0]["exercises"][1]["name"]
        )

    def test_unknown_or_wrong_equipment_is_not_published(self):
        params = dict(goal="strength", level="beginner", days=2, equipment="dumbbells")
        for eid in ["gym_legpress", "invented_exercise"]:
            self.client.generate.return_value = LLMResponse(
                json.dumps({"found": True, "days": [{"exercise_ids": [eid] * 5}] * 2}),
                "test",
                {},
            )
            result = self.pipeline.program(params)
            self.assertTrue(result["found"])
            self.assertNotIn(
                "Жим ногами в тренажёре",
                [e["name"] for d in result["days"] for e in d["exercises"]],
            )

    def test_provider_failure_is_not_hidden_by_template(self):
        self.client.generate.side_effect = LLMError("auth", "Ошибка авторизации")
        with self.assertRaises(LLMError):
            self.pipeline.program(
                dict(goal="muscle", level="beginner", days=3, equipment="gym")
            )

    def test_short_questions_reach_retrieval(self):
        for question in [
            "Нужен ли креатин?",
            "Сколько белка нужно в день?",
            "Сколько нужно спать?",
            "Сколько отдыхать между подходами?",
        ]:
            self.assertTrue(self.pipeline.retrieve(question), question)
