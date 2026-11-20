"""Текущая сборка автономна, не переписывает тексты и проверяет метаданные."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from app.rag.build_index import build_project
from app.rag.pipeline import RAGPipeline
from app.rag.retrieval import ConceptBM25


class ProjectBuildTests(unittest.TestCase):
    def setup_project(self, root):
        folder = root / "data/corpus"
        folder.mkdir(parents=True)
        docs = [
            {
                "doc_id": "one",
                "topic": "training",
                "source": "Источник",
                "section": "Раздел",
                "text": "Отдых между подходами.",
            }
        ]
        manifest = [{"doc_id": "one", "answer_patterns": ["отдых"]}]
        config = ConceptBM25.from_documents(docs, lemmatize=False).config
        values = {
            "documents": docs,
            "manifest": manifest,
            "profile": {
                "corpus": "data/corpus/documents.json",
                "index": "data/index/bm25.json",
                "retrieval": config,
                "refusal_threshold": 0,
            },
        }
        for name, value in values.items():
            (folder / f"{name}.json").write_text(json.dumps(value))
        return folder, manifest

    def test_build_load_and_repeat_without_historical_versions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder, _ = self.setup_project(root)
            originals = {
                n: (folder / n).read_bytes()
                for n in ("documents.json", "manifest.json")
            }
            result = build_project(root)
            pipeline = RAGPipeline.from_project(root=root, client=Mock())
            self.assertEqual(result["documents"], 1)
            self.assertEqual(pipeline.retrieve("Отдых")[0]["doc_id"], "one")
            index = (root / "data/index/bm25.json").read_bytes()
            self.assertEqual(build_project(root), result)
            self.assertEqual((root / "data/index/bm25.json").read_bytes(), index)
            for name, data in originals.items():
                self.assertEqual((folder / name).read_bytes(), data)

    def test_invalid_metadata_rejected_before_output(self):
        for change in ("duplicate", "missing", "support", "dose", "regex"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                folder, manifest = self.setup_project(root)
                if change == "duplicate":
                    manifest *= 2
                elif change == "missing":
                    manifest[0]["doc_id"] = "other"
                elif change == "support":
                    manifest[0]["support_doc_ids"] = ["absent"]
                elif change == "dose":
                    manifest[0]["general_dose_passages"] = ["Отсутствующая цитата."]
                else:
                    manifest[0]["answer_patterns"] = ["["]
                (folder / "manifest.json").write_text(json.dumps(manifest))
                original = (folder / "profile.json").read_bytes()
                import re

                with self.assertRaises((ValueError, re.error)):
                    build_project(root)
                self.assertFalse((root / "data/index/bm25.json").exists())
                self.assertEqual((folder / "profile.json").read_bytes(), original)

    def test_index_cannot_overwrite_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder, _ = self.setup_project(root)
            profile_path = folder / "profile.json"
            profile = json.loads(profile_path.read_text())
            profile["index"] = profile["corpus"]
            profile_path.write_text(json.dumps(profile))
            original = (folder / "documents.json").read_bytes()
            with self.assertRaises(ValueError):
                build_project(root)
            self.assertEqual((folder / "documents.json").read_bytes(), original)
