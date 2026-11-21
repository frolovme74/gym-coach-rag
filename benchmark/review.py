"""Живой прогон 15 вопросов с сохранением поиска, контекста и ответов."""

import argparse
import hashlib
import json
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from app.llm import GigaChatClient, LLMError
from app.rag.pipeline import FORBIDDEN, PROFILE, ROOT, RAGPipeline
from app.server import create_server
from benchmark.run import json_hash


class RecordedClient:
    def __init__(self, client):
        self.client, self.calls = client, []

    def generate(self, messages, **kwargs):
        row = {"messages": messages}
        self.calls.append(row)
        try:
            response = self.client.generate(messages, **kwargs)
        except LLMError as exc:
            row["error"] = {"code": exc.code, "message": str(exc)}
            raise
        row.update(model=response.model, usage=response.usage, response=response.text)
        return response


def document_position(hits, relevant):
    rank = next((n for n, h in enumerate(hits, 1) if h.doc_id in relevant), None)
    return {
        "rank": rank,
        "label": "1" if rank == 1 else "2–3" if rank in (2, 3) else "нет",
        "not_in_top3": rank not in (1, 2, 3),
    }


def load_suite(folder, documents):
    lock = json.loads((folder / "frozen.json").read_text())
    for name, expected in lock["sha256"].items():
        if hashlib.sha256((folder / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Вопросы или эталоны изменены после фиксации")
    questions = json.loads((folder / "questions.json").read_text())
    gold = json.loads((folder / "gold.json").read_text())
    if questions != [{"id": q["id"], "query": q["query"]} for q in gold]:
        raise ValueError("Вопросы не соответствуют эталонам")
    for q in gold:
        if not set(q["relevant"]) <= documents.keys():
            raise ValueError("Эталон ссылается на отсутствующий документ")
        for e in q["evidence"]:
            text = documents[e["doc_id"]]["text"]
            if (
                e["evidence"] not in text
                or hashlib.sha256(text.encode()).hexdigest() != e["text_sha256"]
            ):
                raise ValueError("Эталонная цитата устарела")
    return gold, lock


def run(output, env_file=None):
    if output.exists():
        raise ValueError("Существующий прогон не перезаписывается")
    # Profile validation and reference checks happen before contacting the provider.
    client = RecordedClient(GigaChatClient.from_env(env_file))
    pipeline = RAGPipeline.from_project(client=client)
    questions, lock = load_suite(ROOT / "benchmark/review-15", pipeline.documents)
    report = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "kind": "internal_live_review_not_independent",
        "semantic_grades": "separate_review.json",
        "profile": json.loads((ROOT / PROFILE).read_text()),
        "suite_lock": lock,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "code_sha256": {
            p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
            for p in (
                "benchmark/review.py",
                "app/rag/pipeline.py",
                "app/rag/prompts.py",
                "app/rag/runtime_retrieval.py",
                "app/rag/retrieval.py",
                "app/rag/bm25.py",
                "app/rag/preprocess.py",
                "app/llm.py",
                "app/server.py",
            )
        },
        "cases": [],
    }
    server = create_server(port=0, pipeline=pipeline)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        for q in questions:
            raw = pipeline.index.base.search(q["query"], pipeline.index.n)
            ranked = pipeline.index.search(q["query"], pipeline.index.n)
            prefilter = bool(FORBIDDEN.search(q["query"].replace("ё", "е")))
            context = [] if prefilter else pipeline.retrieve(q["query"])
            before = len(client.calls)
            start = time.monotonic()
            req = Request(
                f"http://127.0.0.1:{server.server_address[1]}/api/ask",
                data=json.dumps({"question": q["query"]}).encode(),
                headers={"Content-Type": "application/json"},
            )
            try:
                response = urlopen(req, timeout=180)
            except HTTPError as exc:
                response = exc
            with response:
                payload, status = json.loads(response.read()), response.status
            report["cases"].append(
                {
                    "id": q["id"],
                    "query": q["query"],
                    "status": status,
                    "seconds": round(time.monotonic() - start, 2),
                    "response": payload,
                    "concept_bm25": {
                        "top3": [vars(h) for h in raw[:3]],
                        "document": document_position(raw, q["relevant"])
                        if q["relevant"]
                        else None,
                    },
                    "runtime": {
                        "top3": [vars(h) for h in ranked[:3]],
                        "document": document_position(ranked, q["relevant"])
                        if q["relevant"]
                        else None,
                    },
                    "blocked_before_retrieval": prefilter,
                    "context_candidates": context,
                    "sent_to_llm": bool(client.calls[before:]),
                    "llm_calls": client.calls[before:],
                }
            )
            print(q["id"], status, "found=", payload.get("found"), flush=True)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
        report["completed"] = len(report["cases"]) == len(questions)
        report["cases_sha256"] = json_hash(report["cases"])
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            f.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", required=True)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.output, args.env_file)


if __name__ == "__main__":
    main()
