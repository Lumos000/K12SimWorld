from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from src.llm_client import LLMClient


class LLMClientSecurityTests(unittest.TestCase):
    def test_gemini_key_is_sent_in_header_not_url(self) -> None:
        response = Mock()
        response.status_code = 200
        response.text = '{"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}'
        response.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": "ok"}]}}]
        }
        response.elapsed.total_seconds.return_value = 0.1

        client = LLMClient(
            "gemini-2.5-pro",
            gemini_api_key="secret-key",
            gemini_api_base="https://generativelanguage.googleapis.com",
            gemini_api_mode="google",
        )
        with patch("src.llm_client.requests.post", return_value=response) as post:
            text, _ = client.call([], "hello", 1, 1)

        self.assertEqual(text, "ok")
        url = post.call_args.args[0]
        headers = post.call_args.kwargs["headers"]
        self.assertNotIn("secret-key", url)
        self.assertEqual(headers["x-goog-api-key"], "secret-key")

    def test_anthropic_missing_key_fails_instead_of_generating_fallback(self) -> None:
        client = LLMClient("claude-sonnet-4-20250514", anthropic_api_key="")
        client.anthropic_api_key = ""
        with self.assertRaisesRegex(RuntimeError, "ANTHROPIC_API_KEY"):
            client.call([], "hello", 1, 1)

    def test_qwen_native_api_path_fails_before_sending_request(self) -> None:
        client = LLMClient(
            "qwen3-vl-plus",
            qwen_api_key="secret-key",
            qwen_base_url="https://workspace.cn-beijing.maas.aliyuncs.com/api/v1",
        )
        with patch("src.llm_client.requests.post") as post:
            with self.assertRaisesRegex(RuntimeError, "/compatible-mode/v1"):
                client.call([], "hello", 1, 1)
        post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
