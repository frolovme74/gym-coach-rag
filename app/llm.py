"""Клиент GigaChat Freemium без сторонних зависимостей."""

import argparse
import json
import math
import os
import re
import ssl
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.client import HTTPException
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, HTTPSHandler, Request, build_opener

ROOT = Path(__file__).resolve().parents[1]
AUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
API_URL = "https://api.giga.chat/v1"
MAX_RESPONSE_BYTES = 2 * 1024 * 1024


class LLMError(RuntimeError):
    """Безопасная для отображения ошибка без текста запроса и секретов."""

    def __init__(self, code, message, status=None):
        super().__init__(message)
        self.code = code
        self.status = status


def _env_values(path):
    """Подмножество dotenv: KEY=value, кавычки и отдельные строки комментариев."""
    values = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return values
    except (OSError, UnicodeError):
        raise LLMError("config", "Не удалось прочитать .env.") from None
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        key, sep, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if not sep or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise LLMError("config", "В .env ожидаются строки KEY=value.")
        if value.startswith(('"', "'")):
            if len(value) < 2 or value[-1] != value[0]:
                raise LLMError("config", "В .env не закрыты кавычки.")
            value = value[1:-1]
        values[key] = value
    return values


@dataclass(frozen=True)
class GigaChatConfig:
    credentials: str = field(repr=False)
    model: str = "GigaChat-2-Pro"
    scope: str = "GIGACHAT_API_PERS"
    ca_bundle_file: str | None = None
    timeout: float = 30.0
    max_tokens: int = 1024
    temperature: float = 0.2

    def __post_init__(self):
        if (
            not isinstance(self.credentials, str)
            or not self.credentials
            or any(c.isspace() for c in self.credentials)
        ):
            raise LLMError(
                "config",
                "Заполните GIGACHAT_CREDENTIALS ключом Authorization key без префикса Basic.",
            )
        if self.scope != "GIGACHAT_API_PERS":
            raise LLMError(
                "config", "Этот клиент настроен только на личный GIGACHAT_API_PERS."
            )
        if not isinstance(self.model, str) or not re.fullmatch(
            r"[A-Za-z0-9_.-]+", self.model
        ):
            raise LLMError("config", "Некорректное имя модели GIGACHAT_MODEL.")
        for value, low, high, name in (
            (self.timeout, 1, 120, "GIGACHAT_TIMEOUT"),
            (self.temperature, 0.0001, 2, "GIGACHAT_TEMPERATURE"),
        ):
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or not low <= value <= high
            ):
                raise LLMError(
                    "config", f"{name} должен быть числом от {low} до {high}."
                )
        if type(self.max_tokens) is not int or not 1 <= self.max_tokens <= 8192:
            raise LLMError(
                "config", "GIGACHAT_MAX_TOKENS должен быть целым от 1 до 8192."
            )

    @classmethod
    def from_env(cls, env_file=None):
        path = Path(env_file) if env_file is not None else ROOT / ".env"
        values = {**_env_values(path), **os.environ}
        if values.get("LLM_PROVIDER", "gigachat") != "gigachat":
            raise LLMError("config", "Поддерживается LLM_PROVIDER=gigachat.")
        ca = values.get("GIGACHAT_CA_BUNDLE_FILE", "").strip()
        if ca:
            ca_path = Path(ca).expanduser()
            ca = str(ca_path if ca_path.is_absolute() else path.parent / ca_path)
        try:
            return cls(
                credentials=values.get("GIGACHAT_CREDENTIALS", "").strip(),
                model=values.get("GIGACHAT_MODEL", "GigaChat-2-Pro").strip(),
                scope=values.get("GIGACHAT_SCOPE", "GIGACHAT_API_PERS").strip(),
                ca_bundle_file=ca or None,
                timeout=float(values.get("GIGACHAT_TIMEOUT", "30")),
                max_tokens=int(values.get("GIGACHAT_MAX_TOKENS", "1024")),
                temperature=float(values.get("GIGACHAT_TEMPERATURE", "0.2")),
            )
        except ValueError:
            raise LLMError(
                "config", "Проверьте числовые настройки GigaChat в .env."
            ) from None


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    usage: dict[str, int]


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _make_opener(config):
    try:
        context = ssl.create_default_context()
        if config.ca_bundle_file:
            context.load_verify_locations(cafile=config.ca_bundle_file)
    except (OSError, ValueError):
        raise LLMError(
            "tls",
            "Не удалось загрузить доверенные сертификаты; проверьте GIGACHAT_CA_BUNDLE_FILE.",
        ) from None
    return build_opener(HTTPSHandler(context=context), _NoRedirect())


class GigaChatClient:
    def __init__(self, config):
        self.config = config
        self._opener = _make_opener(config)
        self._token = None
        self._expires_at = 0.0
        self._lock = threading.Lock()

    @classmethod
    def from_env(cls, env_file=None):
        return cls(GigaChatConfig.from_env(env_file))

    def _request(self, url, *, headers, data=None):
        # Один повтор только для явных 429/503; таймаут POST не повторяется.
        for attempt in range(2):
            request = Request(
                url, data=data, headers={"Accept": "application/json", **headers}
            )
            try:
                with self._opener.open(
                    request, timeout=self.config.timeout
                ) as response:
                    raw = response.read(MAX_RESPONSE_BYTES + 1)
            except HTTPError as exc:
                status = exc.code
                retry_after = (
                    exc.headers.get("Retry-After", "1") if exc.headers else "1"
                )
                exc.close()
                if status in (429, 503) and attempt == 0:
                    try:
                        delay = float(retry_after)
                        if not math.isfinite(delay):
                            raise ValueError
                    except ValueError:
                        delay = 1.0
                    # Не нарушаем длинный Retry-After ранним повтором.
                    if 0 <= delay <= 5:
                        time.sleep(delay)
                        continue
                code, message = {
                    401: (
                        "auth",
                        "GigaChat отклонил авторизацию. Проверьте ключ и scope.",
                    ),
                    402: (
                        "quota",
                        "Недостаточно токенов GigaChat. Проверьте бесплатную квоту в Studio.",
                    ),
                    403: (
                        "forbidden",
                        "GigaChat запретил доступ. Проверьте доступ к модели и проекту.",
                    ),
                    429: (
                        "rate_limit",
                        "Превышен лимит GigaChat. Повторите запрос позже.",
                    ),
                }.get(status, ("http", f"Ошибка GigaChat HTTP {status}."))
                raise LLMError(code, message, status) from None
            except (TimeoutError, URLError, OSError, HTTPException) as exc:
                reason = getattr(exc, "reason", exc)
                if isinstance(reason, ssl.SSLError):
                    raise LLMError(
                        "tls",
                        "Ошибка проверки TLS. Укажите сертификат НУЦ в GIGACHAT_CA_BUNDLE_FILE.",
                    ) from None
                if isinstance(reason, TimeoutError):
                    raise LLMError(
                        "timeout",
                        "GigaChat не ответил вовремя; автоматический повтор не выполнен.",
                    ) from None
                raise LLMError(
                    "network", "Не удалось соединиться с GigaChat."
                ) from None
            if len(raw) > MAX_RESPONSE_BYTES:
                raise LLMError("response", "Слишком большой ответ GigaChat.")
            try:
                value = json.loads(raw)
            except (ValueError, UnicodeError):
                raise LLMError(
                    "response", "GigaChat вернул некорректный JSON."
                ) from None
            if not isinstance(value, dict):
                raise LLMError("response", "Неожиданный формат ответа GigaChat.")
            return value
        raise AssertionError("Недостижимая ветка")

    def _access_token(self):
        if self._token and time.time() + 60 < self._expires_at:
            return self._token
        payload = self._request(
            AUTH_URL,
            headers={
                "Authorization": "Basic " + self.config.credentials,
                "RqUID": str(uuid.uuid4()),
                "Content-Type": "application/x-www-form-urlencoded",
            },
            data=urlencode({"scope": self.config.scope}).encode("ascii"),
        )
        token, expires = payload.get("access_token"), payload.get("expires_at")
        if (
            not isinstance(token, str)
            or not token
            or any(c.isspace() for c in token)
            or type(expires) not in (int, float)
            or not math.isfinite(expires)
            or expires / 1000 <= time.time() + 60
        ):
            raise LLMError(
                "response", "Некорректный токен или срок действия в ответе GigaChat."
            )
        self._token, self._expires_at = token, expires / 1000
        return token

    def _authorized(self, path, payload=None):
        data = (
            None
            if payload is None
            else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        )
        for attempt in range(2):
            token = self._access_token()
            try:
                return self._request(
                    API_URL + path,
                    data=data,
                    headers={
                        "Authorization": "Bearer " + token,
                        "Content-Type": "application/json",
                    },
                )
            except LLMError as exc:
                if exc.status != 401 or attempt:
                    raise
                self._token, self._expires_at = None, 0.0
        raise AssertionError("Недостижимая ветка")

    def list_models(self):
        with self._lock:
            result = self._authorized("/models")
        models = result.get("data")
        if not isinstance(models, list) or any(
            not isinstance(m, dict) or not isinstance(m.get("id"), str) for m in models
        ):
            raise LLMError("response", "Некорректный список моделей GigaChat.")
        return [m["id"] for m in models]

    def generate(self, messages, *, max_tokens=None):
        if max_tokens is not None and (
            type(max_tokens) is not int or not 1 <= max_tokens <= 8192
        ):
            raise LLMError("input", "max_tokens должен быть целым от 1 до 8192.")
        if not isinstance(messages, list) or not messages:
            raise LLMError("input", "Нужен непустой список сообщений.")
        clean = []
        for message in messages:
            if (
                not isinstance(message, dict)
                or message.get("role") not in ("system", "user", "assistant")
                or not isinstance(message.get("content"), str)
                or not message["content"].strip()
            ):
                raise LLMError(
                    "input", "Сообщение должно содержать role и непустой content."
                )
            clean.append({"role": message["role"], "content": message["content"]})
        payload = {
            "model": self.config.model,
            "messages": clean,
            "stream": False,
            "max_tokens": self.config.max_tokens if max_tokens is None else max_tokens,
            "temperature": self.config.temperature,
        }
        with self._lock:
            result = self._authorized("/chat/completions", payload)
        try:
            choice = result["choices"][0]
            finish = choice["finish_reason"]
            text = choice["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise LLMError(
                "response", "В ответе GigaChat нет текстового сообщения."
            ) from None
        if finish == "length":
            raise LLMError(
                "truncated",
                "Ответ обрезан лимитом токенов; увеличьте GIGACHAT_MAX_TOKENS.",
            )
        if finish == "blacklist":
            raise LLMError("blocked", "GigaChat отклонил запрос по своим ограничениям.")
        if finish != "stop" or not isinstance(text, str) or not text.strip():
            raise LLMError(
                "response", "GigaChat не вернул законченный текстовый ответ."
            )
        usage = result.get("usage", {})
        if not isinstance(usage, dict) or any(
            type(v) is not int or v < 0 for v in usage.values()
        ):
            raise LLMError("response", "Некорректная статистика токенов GigaChat.")
        model = result.get("model", self.config.model)
        if not isinstance(model, str):
            raise LLMError("response", "Некорректное имя модели в ответе GigaChat.")
        return LLMResponse(text.strip(), model, usage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--check",
        action="store_true",
        help="Проверить авторизацию и модель без генерации",
    )
    mode.add_argument(
        "--smoke", action="store_true", help="Один пробный ответ, расходует квоту"
    )
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    args = parser.parse_args()
    try:
        client = GigaChatClient.from_env(args.env_file)
        if args.check:
            models = client.list_models()
            if client.config.model not in models:
                raise LLMError(
                    "model",
                    "Авторизация работает, но GIGACHAT_MODEL отсутствует в списке доступных моделей.",
                )
            print(
                "Авторизация работает, выбранная модель доступна. Генерация не выполнялась."
            )
        else:
            result = client.generate(
                [
                    {
                        "role": "user",
                        "content": "Ответь одной короткой фразой по-русски: соединение работает.",
                    }
                ]
            )
            print(result.text)
            print("Токены:", json.dumps(result.usage, ensure_ascii=False))
    except LLMError as exc:
        parser.exit(1, f"Ошибка [{exc.code}]: {exc}\n")


if __name__ == "__main__":
    main()
