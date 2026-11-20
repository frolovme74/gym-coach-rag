"""Покрытие параметров программы, оборудования и происхождения данных."""

import itertools
import json
import unittest
from unittest.mock import Mock

from app.llm import LLMError, LLMResponse
from app.rag.pipeline import EQUIPMENT, GOALS, LEVELS, RAGPipeline
from app.rag.planner import ACCESSORY_PATTERNS, FORMATS, prepare, schedule_for, valid_days


class PlannerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = Mock()
        cls.pipeline = RAGPipeline.from_project(client=cls.client)

    def setUp(self):
        self.client.reset_mock(side_effect=True)
        self.client.generate.return_value = LLMResponse('{"found":false}', "test", {})

    def test_all_600_combinations_are_distinct_complete_and_grounded(self):
        signatures = {}
        for goal, level, days, equipment, format_name in itertools.product(
            GOALS, LEVELS, range(2, 6), EQUIPMENT, FORMATS
        ):
            params = dict(goal=goal, level=level, days=days, equipment=equipment,
                          format=format_name)
            with self.subTest(**params):
                result = self.pipeline.program(params)
                self.assertTrue(result["found"])
                self.assertEqual(days, len(result["days"]))
                signature = tuple(
                    tuple(sorted((ex["name"], ex["sets"], ex["reps"])
                                 for ex in day["exercises"]))
                    for day in result["days"]
                )
                self.assertNotIn(signature, signatures, signatures.get(signature))
                signatures[signature] = params
                catalog, rule, rule_doc, _, draft, _ = prepare(self.pipeline, params)
                self.assertTrue(result["note"].startswith(LEVELS[level]))
                self.assertIn(f"{rule['sets']} × {rule['reps']}", result["note"])
                self.assertIn("2–3 минуты", result["note"])
                self.assertIn("3–5 минут", result["note"])
                self.assertNotIn("1–2 минуты", result["note"])
                self.assertIn(FORMATS[format_name], result["note"])
                self.assertTrue(valid_days({"days": draft}, catalog, params))
                self.assertTrue(
                    all(equipment in e["equipment"] for e in catalog.values())
                )
                patterns = {e["name"]: e["pattern"] for e in catalog.values()}
                for day, (_, kind, light) in zip(result["days"], schedule_for(params)):
                    names = [e["name"] for e in day["exercises"]]
                    self.assertEqual(len(names), len(set(names)))
                    self.assertGreaterEqual(len(names), 3 if kind in {"push", "pull"} else 4)
                    self.assertTrue(set(names) <= patterns.keys())
                    for ex in day["exercises"]:
                        self.assertEqual(1 if light else rule["sets"], ex["sets"])
                        accessory = patterns[ex["name"]] in ACCESSORY_PATTERNS
                        self.assertEqual(rule["accessory_reps" if accessory else "reps"], ex["reps"])
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
                    self.assertNotIn("1–2 минуты", source["fragment"])
                    self.assertNotIn("60–90 секунд", source["fragment"])
                    self.assertTrue(source["quotes"])
                    self.assertTrue(
                        all(q in source["fragment"] for q in source["quotes"])
                    )
                    if doc_id == rule_doc and "guidance" in rule:
                        self.assertIn(rule["guidance"], source["quotes"])
                        self.assertIn(rule["guidance"], result["note"])
                    if doc_id == rule["rest_doc"]:
                        self.assertTrue(any("3–5 минут" in q for q in source["quotes"]))
                self.assertIn(rule["rest_doc"], {source["doc_id"] for source in result["sources"]})
        self.assertEqual(600, len(signatures))

    def test_legacy_schedule_remains_available_without_format(self):
        params = dict(goal="muscle", level="beginner", days=5, equipment="gym")
        result = self.pipeline.program(params)
        self.assertTrue(result["found"])
        self.assertIn("верх тела", result["days"][0]["name"])
        self.assertIn("всё тело", result["days"][-1]["name"])
        self.assertTrue(all(ex["sets"] == 1 for ex in result["days"][-1]["exercises"]))

    def test_split_rejects_movement_from_another_day(self):
        params = dict(goal="muscle", level="advanced", days=3, equipment="gym", format="split")
        catalog, _, _, _, draft, _ = prepare(self.pipeline, params)
        self.assertTrue(valid_days({"days": draft}, catalog, params))
        draft[0]["exercise_ids"].append("db_split_pause")
        self.assertFalse(valid_days({"days": draft}, catalog, params))

    def test_load_changes_with_goal_and_level(self):
        for goal in GOALS:
            with self.subTest(goal=goal):
                profiles = []
                for level in LEVELS:
                    params = dict(goal=goal, level=level, days=3, equipment="gym")
                    _, rule, _, _, _, _ = prepare(self.pipeline, params)
                    profiles.append((rule["sets"], rule["reps"]))
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
