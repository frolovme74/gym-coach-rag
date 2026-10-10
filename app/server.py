"""HTTP-сервер: раздаёт интерфейс и отвечает на запросы API (spec/tech/api.md)."""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import stub

STATIC_DIR = (Path(__file__).parent / "static").resolve()
CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
}
MAX_BODY_BYTES = 16 * 1024

# На контрольной точке 2 заглушка заменяется пайплайном RAG с тем же контрактом.
POST_ROUTES = {
    "/api/ask": stub.ask,
    "/api/program": stub.program,
}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/api/health":
            self._send_json(200, {"status": "ok", "mode": stub.MODE})
        elif path.startswith("/api/"):
            self._send_json(404, {"error": "Не найдено"})
        else:
            self._send_static(path)

    def do_POST(self):
        handler = POST_ROUTES.get(self.path.split("?", 1)[0])
        if handler is None:
            self._send_json(404, {"error": "Не найдено"})
            return
        try:
            payload = self._read_json()
            result = handler(payload)
        except ValueError as error:
            self._send_json(400, {"error": str(error)})
            return
        self._send_json(200, result)

    def _read_json(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
        except ValueError:
            raise ValueError("Некорректный заголовок Content-Length")
        if length > MAX_BODY_BYTES:
            raise ValueError("Слишком большой запрос")
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise ValueError("Тело запроса должно быть JSON")
        if not isinstance(payload, dict):
            raise ValueError("Тело запроса должно быть JSON-объектом")
        return payload

    def _send_static(self, path):
        file = (STATIC_DIR / (path.lstrip("/") or "index.html")).resolve()
        if STATIC_DIR not in file.parents or not file.is_file():
            self._send_json(404, {"error": "Не найдено"})
            return
        content_type = CONTENT_TYPES.get(file.suffix, "application/octet-stream")
        self._send(200, content_type, file.read_bytes())

    def _send_json(self, status, data):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self._send(status, "application/json; charset=utf-8", body)

    def _send(self, status, content_type, body):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass


def create_server(host="127.0.0.1", port=8000):
    return ThreadingHTTPServer((host, port), Handler)
