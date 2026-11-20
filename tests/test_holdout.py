"""Защита неизменного holdout и происхождения параметров."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmark.holdout import (
    BASELINE,
    FROZEN_PATHS,
    ROOT,
    file_hash,
    run,
    validate_holdout,
    verify_frozen,
    write_new_json,
)
from benchmark.run import json_hash


class HoldoutTests(unittest.TestCase):
    def fixture(self):
        queries = json.loads((ROOT / "benchmark/queries.json").read_text())
        annotations = json.loads((ROOT / "benchmark/judgments.json").read_text())
        documents = json.loads((ROOT / "data/corpus/documents.json").read_text())
        development = copy.deepcopy(queries)
        old = annotations["queries"]
        annotations["queries"] = {}
        for q in queries:
            previous = q["id"]
            q["id"] = "holdout_" + previous
            q["query"] = "Новый контекст: " + q["query"]
            annotations["queries"][q["id"]] = old[previous]
        annotations.update(
            queries_sha256=json_hash(queries),
            dataset_role="holdout_v1",
            ranking_seen_before_freeze=False,
            novelty_review="Synthetic fixture only; actual novelty is reviewed by humans/agents.",
        )
        return (
            queries,
            development,
            annotations,
            documents,
            annotations["corpus_sha256"],
        )

    def test_balanced_dataset_and_disjoint_ids(self):
        args = self.fixture()
        self.assertEqual(validate_holdout(*args)["answerable"], 24)
        args[1][0]["id"] = args[0][0]["id"]
        with self.assertRaisesRegex(ValueError, "ID пересекаются"):
            validate_holdout(*args)

    def test_normalized_duplicate_is_rejected(self):
        args = self.fixture()
        args[1][0]["query"] = "  " + args[0][0]["query"].upper() + "  "
        with self.assertRaisesRegex(ValueError, "Повтор вопроса"):
            validate_holdout(*args)

    def test_role_and_review_before_ranking_required(self):
        for field, value in [
            ("dataset_role", "development_v1"),
            ("ranking_seen_before_freeze", True),
            ("novelty_review", ""),
        ]:
            args = self.fixture()
            args[2][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_holdout(*args)

    def frozen_fixture(self, root):
        for name in FROZEN_PATHS:
            p = root / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("{}")
        profile = {
            "retrieval": {
                "k1": 3,
                "b": 0,
                "lemmatize": True,
                "remove_stopwords": False,
            },
            "refusal_threshold": 13.9,
        }
        (root / "benchmark/selected_config.json").write_text(json.dumps(profile))
        return dict(
            version=1,
            dataset_role="holdout_v1",
            files_sha256={p: file_hash(root / p) for p in FROZEN_PATHS},
            baseline_config=BASELINE,
            baseline_threshold=0,
            selected_config=profile["retrieval"],
            selected_threshold=13.9,
        )

    def test_frozen_files_and_threshold_cannot_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lock = self.frozen_fixture(root)
            verify_frozen(lock, root)
            bad = copy.deepcopy(lock)
            bad["selected_threshold"] += 1
            with self.assertRaisesRegex(ValueError, "порог протокола"):
                verify_frozen(bad, root)
            (root / "benchmark/holdout/queries.json").write_text("changed")
            with self.assertRaisesRegex(ValueError, "После фиксации изменён"):
                verify_frozen(lock, root)

    def test_incomplete_fingerprint_list_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lock = self.frozen_fixture(root)
            del lock["files_sha256"][FROZEN_PATHS[0]]
            with self.assertRaisesRegex(ValueError, "Неполный список"):
                verify_frozen(lock, root)

    def test_existing_artifact_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "report.json"
            write_new_json(p, {"first": True})
            original = p.read_bytes()
            with self.assertRaises(FileExistsError):
                write_new_json(p, {"first": False})
            self.assertEqual(p.read_bytes(), original)

    def test_run_protects_inputs_before_search(self):
        with (
            patch("benchmark.holdout.check_branch"),
            patch("benchmark.holdout.evaluate") as evaluate,
        ):
            for path in (
                "benchmark/holdout/frozen.json",
                "benchmark/selected_config.json",
                "data/index/bm25.tuned.json",
            ):
                with (
                    self.subTest(path=path),
                    self.assertRaisesRegex(ValueError, "Выход совпадает"),
                ):
                    run(ROOT / path)
            evaluate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
