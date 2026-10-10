"""Тесты контракта API из spec/tech/api.md."""

import json
import threading
import unittest
import urllib.error
import urllib.request

from app.server import create_server

TOPICS = {"training", "nutrition", "supplements", "recovery"}


class ApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = create_server("127.0.0.1", 0)
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def request(self, path, payload=None, raw=None):
        data = raw if raw is not None else (json.dumps(payload).encode() if payload is not None else None)
        request = urllib.request.Request(self.base + path, data=data)
        try:
            with urllib.request.urlopen(request) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as error:
            with error:
                return error.code, error.read()

    def json(self, path, payload=None, raw=None):
        status, body = self.request(path, payload, raw)
        return status, json.loads(body)

    def assert_sources(self, sources):
        self.assertTrue(sources)
        for source in sources:
            self.assertEqual({"doc_id", "topic", "title", "fragment", "score"}, set(source))
            self.assertIn(source["topic"], TOPICS)

    def test_health(self):
        status, data = self.json("/api/health")
        self.assertEqual(200, status)
        self.assertEqual("ok", data["status"])

    def test_ask_returns_answer_with_sources(self):
        status, data = self.json("/api/ask", {"question": "Сколько отдыхать между подходами?"})
        self.assertEqual(200, status)
        self.assertTrue(data["found"])
        self.assertIn("[1]", data["answer"])
        self.assert_sources(data["sources"])

    def test_ask_covers_every_topic(self):
        questions = {
            "training": "Как делать разминку?",
            "nutrition": "Сколько белка нужно в день?",
            "supplements": "Нужен ли креатин?",
            "recovery": "Сколько нужно спать?",
        }
        self.assertEqual(TOPICS, set(questions))
        for topic, question in questions.items():
            with self.subTest(topic=topic):
                status, data = self.json("/api/ask", {"question": question})
                self.assertEqual(200, status)
                self.assertTrue(data["found"])
                self.assert_sources(data["sources"])
                self.assertEqual({topic}, {source["topic"] for source in data["sources"]})

    def test_ask_outside_corpus_has_no_sources(self):
        status, data = self.json("/api/ask", {"question": "Какая завтра погода?"})
        self.assertEqual(200, status)
        self.assertFalse(data["found"])
        self.assertEqual([], data["sources"])
        self.assertTrue(data["answer"])

    def test_ask_rejects_bad_input(self):
        for payload in ({}, {"question": "   "}, {"question": 5}, {"question": "а" * 501}):
            with self.subTest(payload=payload):
                status, data = self.json("/api/ask", payload)
                self.assertEqual(400, status)
                self.assertIn("error", data)

    def test_ask_rejects_invalid_json(self):
        for raw in (b"not json", b"[1, 2]"):
            with self.subTest(raw=raw):
                status, data = self.json("/api/ask", raw=raw)
                self.assertEqual(400, status)
                self.assertIn("error", data)

    def test_program_returns_requested_days(self):
        for days in range(2, 6):
            with self.subTest(days=days):
                status, data = self.json(
                    "/api/program",
                    {"goal": "muscle", "level": "beginner", "days": days, "equipment": "gym"},
                )
                self.assertEqual(200, status)
                self.assertEqual(days, len(data["days"]))
                for day in data["days"]:
                    self.assertTrue(day["exercises"])
                    for exercise in day["exercises"]:
                        self.assertEqual({"name", "sets", "reps"}, set(exercise))
                self.assert_sources(data["sources"])

    def test_program_rejects_bad_input(self):
        valid = {"goal": "muscle", "level": "beginner", "days": 3, "equipment": "gym"}
        for field, value in (("goal", "fly"), ("level", "pro"), ("days", 7), ("days", "3"), ("equipment", "none")):
            with self.subTest(field=field, value=value):
                status, data = self.json("/api/program", {**valid, field: value})
                self.assertEqual(400, status)
                self.assertIn("error", data)

    def test_static_index(self):
        status, body = self.request("/")
        self.assertEqual(200, status)
        self.assertIn("Gym Coach", body.decode("utf-8"))

    def test_unknown_paths(self):
        for path, payload in (("/api/nope", None), ("/api/nope", {}), ("/missing.css", None), ("/../server.py", None)):
            with self.subTest(path=path):
                status, _ = self.request(path, payload)
                self.assertEqual(404, status)


if __name__ == "__main__":
    unittest.main()
