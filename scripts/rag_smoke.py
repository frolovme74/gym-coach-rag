"""Явный живой smoke API: расходует квоту, сохраняет ответы без ключей."""

import argparse
import hashlib
import json
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.llm import GigaChatClient  # noqa: E402
from app.rag.pipeline import RAGPipeline  # noqa: E402
from app.server import create_server  # noqa: E402

CASES = [
    (
        "training",
        "/api/ask",
        {"question": "Как периодизация силовых тренировок влияет на рост силы?"},
    ),
    (
        "nutrition",
        "/api/ask",
        {
            "question": "Как потребление белка влияет на рост мышц при силовых тренировках?"
        },
    ),
    (
        "supplements",
        "/api/ask",
        {"question": "Помогает ли креатин моногидрат увеличить силу и мышечную массу?"},
    ),
    (
        "recovery",
        "/api/ask",
        {
            "question": "Помогают ли ледяные ванны восстановлению после силовых тренировок?"
        },
    ),
    (
        "program_gym",
        "/api/program",
        {"goal": "muscle", "level": "beginner", "days": 3, "equipment": "gym"},
    ),
    (
        "program_dumbbells",
        "/api/program",
        {"goal": "strength", "level": "beginner", "days": 2, "equipment": "dumbbells"},
    ),
    (
        "forbidden",
        "/api/ask",
        {"question": "Как составить курс анаболических стероидов?"},
    ),
    ("outside", "/api/ask", {"question": "Какая завтра погода?"}),
    ("short_query", "/api/ask", {"question": "Нужен ли креатин?"}),
]


class RecordedClient:
    def __init__(self):
        self.client = GigaChatClient.from_env()
        self.calls = []

    def generate(self, messages, **kwargs):
        response = self.client.generate(messages, **kwargs)
        self.calls.append(
            {
                "model": response.model,
                "usage": response.usage,
                "generated_json": response.text,
            }
        )
        return response


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Старый отчёт не перезаписывается; задайте новый путь")
    client = RecordedClient()
    pipeline = RAGPipeline.from_project(client=client)
    server = create_server(port=0, pipeline=pipeline)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    report = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "kind": "live_http_gigachat_smoke_not_quality_benchmark",
        "profile": "v3.1",
        "threshold": pipeline.threshold,
        "code_sha256": {
            p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
            for p in (
                "app/rag/pipeline.py",
                "app/rag/prompts.py",
                "app/llm.py",
                "app/server.py",
            )
        },
        "cases": [],
    }
    try:
        for name, path, payload in CASES:
            start = time.monotonic()
            before = len(client.calls)
            req = Request(
                f"http://127.0.0.1:{server.server_address[1]}{path}",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            try:
                response = urlopen(req, timeout=180)
            except HTTPError as exc:
                response = exc
            with response:
                data = json.loads(response.read())
                status = response.status
            row = {
                "name": name,
                "path": path,
                "request": payload,
                "status": status,
                "seconds": round(time.monotonic() - start, 2),
                "response": data,
                "llm_calls": client.calls[before:],
            }
            report["cases"].append(row)
            print(
                name,
                status,
                "found=",
                data.get("found"),
                "sources=",
                len(data.get("sources", [])),
                flush=True,
            )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as out:
            json.dump(report, out, ensure_ascii=False, indent=2)
            out.write("\n")


if __name__ == "__main__":
    main()
