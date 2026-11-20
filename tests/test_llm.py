"""Контракт GigaChat без ключей, внешней сети и расхода бесплатной квоты."""

import io
import json
import os
import ssl
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs

from app.llm import (
    API_URL,
    AUTH_URL,
    GigaChatClient,
    GigaChatConfig,
    LLMError,
    _NoRedirect,
    _make_opener,
)

KEY = "test-key-not-real"
TOKEN = "test-access-token"
MESSAGES = [
    {"role": "system", "content": "Только по контексту."},
    {"role": "user", "content": "Контекст [1]: пример. Вопрос?"},
]


def auth(expiry=2000000, token=TOKEN):
    return {"access_token": token, "expires_at": expiry}


def completion(finish="stop", content="Ответ [1]."):
    return {
        "choices": [{"finish_reason": finish, "message": {"content": content}}],
        "model": "GigaChat-2-Pro",
        "usage": {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17},
    }


def http_error(code, retry="0"):
    return HTTPError(
        API_URL,
        code,
        "secret provider body " + KEY,
        {"Retry-After": retry},
        io.BytesIO(KEY.encode()),
    )


class FakeOpener:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def open(self, request, timeout):
        self.requests.append((request, timeout))
        item = self.responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return io.BytesIO(
            item if isinstance(item, bytes) else json.dumps(item).encode()
        )


class LLMTest(unittest.TestCase):
    def setUp(self):
        self.clock = patch("app.llm.time.time", return_value=1000)
        self.clock.start()
        self.addCleanup(self.clock.stop)

    def client(self, responses):
        opener = FakeOpener(responses)
        with patch("app.llm._make_opener", return_value=opener):
            client = GigaChatClient(GigaChatConfig(KEY))
        return client, opener

    def test_oauth_contract_and_message_payload(self):
        client, opener = self.client([auth(), completion()])
        answer = client.generate(MESSAGES)
        self.assertEqual(answer.text, "Ответ [1].")
        self.assertEqual(answer.usage["total_tokens"], 17)
        oauth, timeout = opener.requests[0]
        self.assertEqual(oauth.full_url, AUTH_URL)
        self.assertEqual(oauth.get_header("Authorization"), "Basic " + KEY)
        self.assertEqual(
            parse_qs(oauth.data.decode()), {"scope": ["GIGACHAT_API_PERS"]}
        )
        import uuid

        self.assertEqual(uuid.UUID(oauth.get_header("Rquid")).version, 4)
        request, _ = opener.requests[1]
        self.assertEqual(request.full_url, API_URL + "/chat/completions")
        self.assertEqual(request.get_header("Authorization"), "Bearer " + TOKEN)
        body = json.loads(request.data)
        self.assertEqual(body["messages"], MESSAGES)
        self.assertFalse(body["stream"])
        self.assertEqual(body["max_tokens"], 1024)
        self.assertEqual(timeout, 30)

    def test_per_request_token_budget_does_not_change_default(self):
        client, opener = self.client([auth(), completion(), completion()])
        client.generate(MESSAGES, max_tokens=3000)
        client.generate(MESSAGES)
        self.assertEqual(3000, json.loads(opener.requests[1][0].data)["max_tokens"])
        self.assertEqual(1024, json.loads(opener.requests[2][0].data)["max_tokens"])
        for bad in (True, 0, 9000, "3000"):
            with self.assertRaises(LLMError):
                client.generate(MESSAGES, max_tokens=bad)
        self.assertEqual(3, len(opener.requests))

    def test_token_cached_then_refreshed_before_millisecond_expiry(self):
        client, opener = self.client(
            [auth(1100000), completion(), completion(), auth(3000000), completion()]
        )
        client.generate(MESSAGES)
        client.generate(MESSAGES)
        with patch("app.llm.time.time", return_value=1080):
            client.generate(MESSAGES)
        self.assertEqual(sum(r.full_url == AUTH_URL for r, _ in opener.requests), 2)

    def test_model_401_refreshes_token_once(self):
        client, opener = self.client(
            [auth(), http_error(401), auth(token="new-token"), completion()]
        )
        self.assertEqual(client.generate(MESSAGES).text, "Ответ [1].")
        self.assertEqual(
            opener.requests[-1][0].get_header("Authorization"), "Bearer new-token"
        )
        client, opener = self.client([auth(), http_error(401), auth(), http_error(401)])
        with self.assertRaises(LLMError):
            client.generate(MESSAGES)
        self.assertEqual(len(opener.requests), 4)

    def test_bad_oauth_credentials_not_retried_or_exposed(self):
        client, opener = self.client([http_error(401)])
        with self.assertRaises(LLMError) as error:
            client.generate(MESSAGES)
        self.assertEqual(error.exception.code, "auth")
        self.assertNotIn(KEY, str(error.exception))
        self.assertNotIn(KEY, repr(client.config))
        self.assertEqual(len(opener.requests), 1)

    def test_429_and_503_have_only_one_retry(self):
        for code in (429, 503):
            with self.subTest(code=code), patch("app.llm.time.sleep") as sleep:
                client, opener = self.client(
                    [auth(), http_error(code), http_error(code)]
                )
                with self.assertRaises(LLMError):
                    client.generate(MESSAGES)
                self.assertEqual(len(opener.requests), 3)
                sleep.assert_called_once_with(0)

    def test_long_retry_after_is_not_shortened(self):
        client, opener = self.client([auth(), http_error(429, "120")])
        with patch("app.llm.time.sleep") as sleep, self.assertRaises(LLMError):
            client.generate(MESSAGES)
        sleep.assert_not_called()
        self.assertEqual(len(opener.requests), 2)

    def test_timeout_not_retried_and_tls_has_safe_error(self):
        for cause, code in [
            (TimeoutError(KEY), "timeout"),
            (ssl.SSLError(KEY), "tls"),
            (OSError(KEY), "network"),
        ]:
            client, opener = self.client([auth(), URLError(cause)])
            with self.subTest(code=code), self.assertRaises(LLMError) as error:
                client.generate(MESSAGES)
            self.assertEqual(error.exception.code, code)
            self.assertNotIn(KEY, str(error.exception))
            self.assertEqual(len(opener.requests), 2)

    def test_quota_and_forbidden_do_not_trigger_paid_fallback(self):
        for status, code in [(402, "quota"), (403, "forbidden")]:
            client, opener = self.client([auth(), http_error(status)])
            with self.subTest(status=status), self.assertRaises(LLMError) as error:
                client.generate(MESSAGES)
            self.assertEqual(error.exception.code, code)
            self.assertEqual(len(opener.requests), 2)
            self.assertEqual(
                json.loads(opener.requests[-1][0].data)["model"], "GigaChat-3-Ultra"
            )

    def test_invalid_expiration_rejected(self):
        for expiry in (2000, True, "2000000", float("nan")):
            client, _ = self.client([auth(expiry)])
            with self.subTest(expiry=expiry), self.assertRaises(LLMError):
                client.generate(MESSAGES)

    def test_truncated_and_blocked_are_not_success(self):
        for finish, code in [
            ("length", "truncated"),
            ("blacklist", "blocked"),
            ("function_call", "response"),
        ]:
            client, _ = self.client([auth(), completion(finish)])
            with self.assertRaises(LLMError) as error:
                client.generate(MESSAGES)
            self.assertEqual(error.exception.code, code)

    def test_malformed_and_empty_responses_rejected(self):
        for response in [
            b"<html>error</html>",
            [],
            {},
            {"choices": []},
            completion(content=" "),
        ]:
            client, _ = self.client([auth(), response])
            with self.subTest(response=response), self.assertRaises(LLMError):
                client.generate(MESSAGES)

    def test_invalid_input_does_not_call_network(self):
        client, opener = self.client([])
        for messages in [
            [],
            "hello",
            [{"role": "tool", "content": "x"}],
            [{"role": "user", "content": ""}],
        ]:
            with self.subTest(messages=messages), self.assertRaises(LLMError):
                client.generate(messages)
        self.assertEqual(opener.requests, [])

    def test_model_check_does_not_generate(self):
        client, opener = self.client([auth(), {"data": [{"id": "GigaChat-2-Pro"}]}])
        self.assertEqual(client.list_models(), ["GigaChat-2-Pro"])
        self.assertTrue(opener.requests[-1][0].full_url.endswith("/models"))
        self.assertIsNone(opener.requests[-1][0].data)

    def test_dotenv_quotes_equals_and_process_override(self):
        with (
            tempfile.TemporaryDirectory() as folder,
            patch.dict(os.environ, {}, clear=True),
        ):
            path = Path(folder) / ".env"
            path.write_text(
                '# comment\nGIGACHAT_CREDENTIALS="abc=="\nGIGACHAT_MODEL=GigaChat-2\nGIGACHAT_CA_BUNDLE_FILE=certs/root.pem\n'
            )
            with patch.dict(os.environ, {"GIGACHAT_MODEL": "GigaChat-2-Pro"}):
                config = GigaChatConfig.from_env(path)
            self.assertEqual(config.credentials, "abc==")
            self.assertEqual(config.model, "GigaChat-2-Pro")
            self.assertEqual(
                config.ca_bundle_file, str(Path(folder) / "certs/root.pem")
            )
            self.assertNotIn("GIGACHAT_CREDENTIALS", os.environ)

    def test_env_is_not_executed_or_interpolated(self):
        with (
            tempfile.TemporaryDirectory() as folder,
            patch.dict(os.environ, {}, clear=True),
        ):
            path = Path(folder) / ".env"
            path.write_text("GIGACHAT_CREDENTIALS=${NOT_EXPANDED}\n")
            self.assertEqual(
                GigaChatConfig.from_env(path).credentials, "${NOT_EXPANDED}"
            )
            path.write_text("bad line with secret\n")
            with self.assertRaises(LLMError) as error:
                GigaChatConfig.from_env(path)
            self.assertNotIn("secret", str(error.exception))

    def test_missing_key_invalid_scope_and_numbers(self):
        with (
            tempfile.TemporaryDirectory() as folder,
            patch.dict(os.environ, {}, clear=True),
        ):
            with self.assertRaises(LLMError):
                GigaChatConfig.from_env(Path(folder) / ".env")
        for kwargs in [
            {"scope": "GIGACHAT_API_B2B"},
            {"timeout": float("inf")},
            {"max_tokens": 0},
            {"temperature": -1},
        ]:
            with self.subTest(kwargs=kwargs), self.assertRaises(LLMError):
                GigaChatConfig(KEY, **kwargs)

    def test_tls_context_checks_hosts_and_loads_extra_ca(self):
        context = ssl.create_default_context()
        with (
            patch("app.llm.ssl.create_default_context", return_value=context),
            patch.object(context, "load_verify_locations") as load,
        ):
            _make_opener(GigaChatConfig(KEY, ca_bundle_file="test-ca.pem"))
            load.assert_called_once_with(cafile="test-ca.pem")
        self.assertTrue(context.check_hostname)
        self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)
        self.assertIsNone(
            _NoRedirect().redirect_request(
                Mock(), None, 302, "", {}, "https://example.com"
            )
        )


if __name__ == "__main__":
    unittest.main()
