from __future__ import annotations

import unittest

from src.api_config import (
    DASHSCOPE_DEFAULT_BASE_URL,
    HF_DEFAULT_BASE_URL,
    inspect_api_config,
    qwen_chat_completions_url,
)


class APIConfigTests(unittest.TestCase):
    def test_openai_does_not_expose_key(self) -> None:
        secret = "sk-super-secret"
        status = inspect_api_config("openai", {"OPENAI_API_KEY": secret})
        self.assertTrue(status.configured)
        self.assertEqual(status.key_source, "OPENAI_API_KEY")
        self.assertNotIn(secret, "\n".join(status.safe_lines()))

    def test_qwen_accepts_standard_dashscope_names(self) -> None:
        status = inspect_api_config("qwen", {"DASHSCOPE_API_KEY": "key"})
        self.assertTrue(status.configured)
        self.assertEqual(status.base_url, DASHSCOPE_DEFAULT_BASE_URL)

    def test_qwen_rejects_native_api_path(self) -> None:
        status = inspect_api_config(
            "qwen",
            {
                "DASHSCOPE_API_KEY": "key",
                "DASHSCOPE_API_BASE": "https://workspace.cn-beijing.maas.aliyuncs.com/api/v1",
            },
        )
        self.assertFalse(status.configured)
        self.assertIn("/compatible-mode/v1", status.error)

    def test_qwen_builds_workspace_chat_endpoint(self) -> None:
        endpoint = qwen_chat_completions_url(
            "https://workspace.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
        )
        self.assertEqual(
            endpoint,
            "https://workspace.cn-beijing.maas.aliyuncs.com/compatible-mode/v1/chat/completions",
        )

    def test_huggingface_uses_official_token_alias_and_router(self) -> None:
        status = inspect_api_config("huggingface", {"HF_TOKEN": "token"})
        self.assertTrue(status.configured)
        self.assertEqual(status.base_url, HF_DEFAULT_BASE_URL)

    def test_missing_key_is_not_ready(self) -> None:
        status = inspect_api_config("gemini", {})
        self.assertFalse(status.configured)
        self.assertIn("GEMINI_API_KEY", status.error)

    def test_openai_compatible_gemini_mode(self) -> None:
        status = inspect_api_config(
            "gemini",
            {
                "GEMINI_API_KEY": "secret",
                "GEMINI_API_BASE": "https://example.test/v1",
                "GEMINI_API_MODE": "openai",
            },
        )
        self.assertTrue(status.configured)
        self.assertIn("OpenAI-compatible", status.note)

    def test_invalid_gemini_mode_is_not_ready(self) -> None:
        status = inspect_api_config(
            "gemini",
            {"GEMINI_API_KEY": "secret", "GEMINI_API_MODE": "invalid"},
        )
        self.assertFalse(status.configured)
        self.assertIn("GEMINI_API_MODE", status.error)


if __name__ == "__main__":
    unittest.main()
