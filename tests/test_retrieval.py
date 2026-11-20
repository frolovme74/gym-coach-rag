"""Обобщённые примеры понятий, отсутствие ложного смешения и формат индекса."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from app.rag.bm25 import BM25Index
from app.rag.retrieval import ConceptBM25, TOKENS, detect_concepts


class RetrievalTests(unittest.TestCase):
    def documents(self):
        return [
            {
                "doc_id": "water",
                "text": "Погружение в холодную воду влияет на восстановление.",
            },
            {
                "doc_id": "walk",
                "text": "Активная заминка — лёгкая ходьба после тренировки.",
            },
            {
                "doc_id": "supplement",
                "text": "Принимать добавку каждый день может быть полезно для тренировки.",
            },
        ]

    def test_cold_water_aliases_and_cooldown_are_separate(self):
        for text in (
            "ледяные ванны",
            "ванна со льдом",
            "Купание в холодной воде",
            "Cold-water immersion (CWI)",
        ):
            self.assertEqual(detect_concepts(text), ("cold_water_immersion",))
        for text in ("активное охлаждение", "Активная заминка", "active cool-down"):
            self.assertEqual(detect_concepts(text), ("post_exercise_cooldown",))
        self.assertEqual(detect_concepts("Замёрз на прогулке в холодную погоду."), ())

    def test_periodization_natural_language_and_named_forms(self):
        for text in (
            "Волнообразная периодизация нагрузок.",
            "Linear periodization differs from daily undulating periodization.",
            "Хочу чередовать тяжёлый день и лёгкий день тренировок.",
            "Менять нагрузку от недели к неделе полезно?",
        ):
            self.assertIn("periodization", detect_concepts(text))
        self.assertNotIn(
            "periodization",
            detect_concepts("Каждую неделю постепенно добавлять вес к штанге."),
        )
        self.assertNotIn(
            "periodization",
            detect_concepts(
                "Лёгкая нагрузка полезна. Через неделю тяжёлый день. Чередовать продукты."
            ),
        )

    def test_superset_alias_requires_training_context(self):
        for text in (
            "Суперсеты позволяют сократить время.",
            "Парные силовые упражнения.",
            "Объединять упражнения на мышцы-антагонисты в пары.",
            "Упражнения на противоположные мышцы объединяют в пары.",
        ):
            self.assertIn("superset", detect_concepts(text))
        for text in (
            "Противоположные мнения объединяют людей в пары.",
            "Пара упражнений по математике.",
            "Суперсеточный метод вычислений.",
        ):
            self.assertNotIn("superset", detect_concepts(text))

    def test_aliases_can_coexist_but_do_not_imply_relevance(self):
        matches = detect_concepts("Ледяные ванны не лучше, чем активная заминка.")
        self.assertEqual(matches, ("cold_water_immersion", "post_exercise_cooldown"))
        index = ConceptBM25.from_documents(self.documents(), lemmatize=False)
        tokens = index.query_tokens(
            "Не принимать холодную ванну после тренировки 10 минут"
        )
        self.assertIn("не", tokens)
        self.assertIn("10", tokens)
        self.assertIn("после", tokens)

    def test_plain_mode_matches_old_bm25(self):
        documents = self.documents()
        for b in (0, 0.75):
            old = BM25Index.from_documents(
                documents, k1=3, b=b, lemmatize=False, remove_stopwords=False
            )
            new = ConceptBM25.from_documents(
                documents, concepts=False, k1=3, b=b, lemmatize=False
            )
            self.assertEqual(old.corpus_sha256, new.corpus_sha256)
            self.assertEqual(
                old.search("тренировки тренировки вода", k=10),
                new.search("тренировки тренировки вода", k=10),
            )

    def test_concepts_bridge_aliases_without_rewriting_documents(self):
        documents = self.documents()
        before = copy.deepcopy(documents)
        index = ConceptBM25.from_documents(documents, lemmatize=False)
        hits = index.search("ледяная ванна", 3)
        self.assertEqual([h.doc_id for h in hits], ["water"])
        self.assertEqual(documents, before)
        self.assertEqual(index.document_concepts["walk"], ("post_exercise_cooldown",))

    def test_dedup_only_changes_query_frequency(self):
        index = ConceptBM25.from_documents(
            self.documents(), deduplicate_query=True, lemmatize=False
        )
        tokens = index.query_tokens("ванна ванна холодная ванна")
        self.assertEqual(tokens.count("ванна"), 1)
        self.assertEqual(tokens.count(TOKENS["cold_water_immersion"]), 3)
        self.assertEqual(
            dict(index.index.postings[TOKENS["cold_water_immersion"]])["water"], 1
        )

    def test_default_lemmatization_retrieves_three_concepts(self):
        documents = [
            {
                "doc_id": "schedule",
                "text": "Линейная и волнообразная периодизация могут давать сходные результаты.",
            },
            {
                "doc_id": "pairs",
                "text": "Суперсеты выполняются с ограниченным отдыхом и позволяют экономить время.",
            },
            {
                "doc_id": "water",
                "text": "Погружение в холодную воду после занятий может менять адаптацию.",
            },
            {
                "doc_id": "noise",
                "text": "Силовые занятия влияют на рост мышц. Тяжёлые и лёгкие нагрузки изучают в тренировках.",
            },
        ]
        index = ConceptBM25.from_documents(documents)
        for query, expected in (
            ("Как чередовать тяжёлые и лёгкие дни тренировок?", "schedule"),
            ("Парные упражнения на мышцы-антагонисты без отдыха", "pairs"),
            ("Нужны ли ледяные ванны после силовых занятий?", "water"),
        ):
            with self.subTest(query=query):
                self.assertEqual(index.search(query, k=1)[0].doc_id, expected)
        tokens = index.query_tokens("Не увеличивать 10 кг после тренировок")
        self.assertIn("тренировка", tokens)
        self.assertTrue({"не", "10", "кг", "после"} <= set(tokens))

    def test_metadata_and_ids_do_not_add_search_terms(self):
        documents = [
            {
                "doc_id": "суперсет",
                "source": "периодизация",
                "section": "ледяные ванны",
                "topic": "training",
                "text": "Обычный текст",
            }
        ]
        index = ConceptBM25.from_documents(documents, lemmatize=False)
        self.assertEqual(index.document_concepts["суперсет"], ())
        self.assertEqual(index.search("суперсет"), [])

    def test_empty_unknown_and_invalid_k(self):
        index = ConceptBM25.from_documents([], lemmatize=False)
        self.assertEqual(index.search("ванна"), [])
        self.assertEqual(index.search(""), [])
        with self.assertRaises(TypeError):
            index.search("ванна", k=True)
        with self.assertRaises(ValueError):
            index.search("ванна", k=-1)
        index = ConceptBM25.from_documents(self.documents(), lemmatize=False)
        self.assertEqual(index.search("несуществующийтерм"), [])

    def test_options_and_duplicate_ids_validated(self):
        for options in (
            {"concepts": 1},
            {"deduplicate_query": 1},
            {"concept_weight": 0},
            {"concept_weight": True},
            {"k1": -1},
            {"b": 2},
        ):
            with (
                self.subTest(options=options),
                self.assertRaises((ValueError, TypeError)),
            ):
                ConceptBM25.from_documents([], **options)
        with self.assertRaises(ValueError):
            ConceptBM25.from_documents([self.documents()[0]] * 2)

    def test_roundtrip_preserves_scores_settings_and_raw_hash(self):
        index = ConceptBM25.from_documents(
            self.documents(), deduplicate_query=True, lemmatize=False
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "index.json"
            index.save(path)
            loaded = ConceptBM25.load(path)
            self.assertEqual(index.config, loaded.config)
            self.assertEqual(index.document_concepts, loaded.document_concepts)
            self.assertEqual(
                index.search("ледяные ванны после тренировки"),
                loaded.search("ледяные ванны после тренировки"),
            )
            self.assertTrue(loaded.matches_documents(self.documents()))
            changed = self.documents()
            changed[0]["text"] += " Изменение."
            self.assertFalse(loaded.matches_documents(changed))
            with self.assertRaises(ValueError):
                BM25Index.load(path)

    def test_changed_rules_or_inconsistent_metadata_rejected(self):
        index = ConceptBM25.from_documents(self.documents(), lemmatize=False)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "index.json"
            index.save(path)
            original = json.loads(path.read_text())
            for change in ("rules", "config", "concepts"):
                payload = copy.deepcopy(original)
                if change == "rules":
                    payload["rules_sha256"] = "0" * 64
                elif change == "config":
                    payload["config"]["k1"] = 1.5
                else:
                    payload["document_concepts"]["water"] = []
                path.write_text(json.dumps(payload))
                with self.subTest(change=change), self.assertRaises(ValueError):
                    ConceptBM25.load(path)


if __name__ == "__main__":
    unittest.main()
