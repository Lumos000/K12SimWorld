"""Credential discovery and preflight checks for model providers.

This module deliberately reports only variable names and endpoint URLs.  It
must never print credential values.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple
from urllib.parse import urlparse


DASHSCOPE_DEFAULT_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
GEMINI_DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com"
HF_DEFAULT_BASE_URL = "https://router.huggingface.co/v1"


def _first_nonempty(environ: Mapping[str, str], names: Sequence[str]) -> Tuple[str, str]:
    for name in names:
        value = (environ.get(name) or "").strip()
        if value:
            return name, value
    return "", ""


def qwen_chat_completions_url(base_url: str) -> str:
    """Build a Qwen OpenAI-compatible chat endpoint or raise a useful error."""
    base = (base_url or "").strip().rstrip("/")
    parsed = urlparse(base)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("DashScope Base URL must be an absolute http(s) URL.")

    path = parsed.path.rstrip("/")
    if path.endswith("/api/v1"):
        raise ValueError(
            "DASHSCOPE_API_BASE ends with /api/v1, which is the native DashScope API. "
            "This project uses OpenAI-compatible Chat Completions; replace /api/v1 with "
            "/compatible-mode/v1 on the same workspace host."
        )
    if path.endswith("/chat/completions"):
        return base
    if path.endswith("/v1"):
        return f"{base}/chat/completions"
    if path.endswith("/compatible-mode"):
        return f"{base}/v1/chat/completions"
    raise ValueError(
        "Unsupported DashScope Base URL path. Expected a URL ending in "
        "/compatible-mode/v1 (or a complete /chat/completions endpoint)."
    )


@dataclass(frozen=True)
class APIConfigStatus:
    provider: str
    configured: bool
    key_source: str
    accepted_key_variables: Tuple[str, ...]
    base_url: str = ""
    base_url_source: str = ""
    error: str = ""
    note: str = ""

    def safe_lines(self) -> list[str]:
        """Return a human-readable report without exposing a secret value."""
        lines = [f"Provider: {self.provider}"]
        if self.key_source:
            lines.append(f"API key: configured via {self.key_source} (value hidden)")
        else:
            lines.append(
                "API key: missing; set one of " + ", ".join(self.accepted_key_variables)
            )
        if self.base_url:
            source = f" via {self.base_url_source}" if self.base_url_source else " (default)"
            lines.append(f"Base URL: {self.base_url}{source}")
        if self.note:
            lines.append(f"Note: {self.note}")
        if self.error:
            lines.append(f"Error: {self.error}")
        lines.append(f"Status: {'ready' if self.configured else 'not ready'}")
        return lines


def inspect_api_config(
    provider: str,
    environ: Optional[Mapping[str, str]] = None,
) -> APIConfigStatus:
    """Inspect one provider's environment configuration without making a request."""
    env = environ if environ is not None else os.environ
    normalized = (provider or "").strip().lower()

    if normalized == "openai":
        key_names = ("OPENAI_API_KEY",)
        key_source, _ = _first_nonempty(env, key_names)
        base_source, base_url = _first_nonempty(env, ("OPENAI_BASE_URL", "OPENAI_API_BASE"))
        return APIConfigStatus(
            provider=normalized,
            configured=bool(key_source),
            key_source=key_source,
            accepted_key_variables=key_names,
            base_url=base_url or "https://api.openai.com/v1",
            base_url_source=base_source,
            error="" if key_source else "OpenAI-compatible calls require OPENAI_API_KEY.",
        )

    if normalized == "anthropic":
        key_names = ("ANTHROPIC_API_KEY",)
        key_source, _ = _first_nonempty(env, key_names)
        return APIConfigStatus(
            provider=normalized,
            configured=bool(key_source),
            key_source=key_source,
            accepted_key_variables=key_names,
            base_url="https://api.anthropic.com/v1/messages",
            error="" if key_source else "Anthropic calls require ANTHROPIC_API_KEY.",
        )

    if normalized == "qwen":
        key_names = ("DASHSCOPE_API_KEY", "QWEN_API_KEY")
        key_source, _ = _first_nonempty(env, key_names)
        base_source, base_url = _first_nonempty(env, ("DASHSCOPE_API_BASE", "QWEN_API_BASE"))
        base_url = base_url or DASHSCOPE_DEFAULT_BASE_URL
        endpoint_error = ""
        try:
            qwen_chat_completions_url(base_url)
        except ValueError as exc:
            endpoint_error = str(exc)
        error = "" if key_source else "Qwen calls require a DashScope API key."
        if endpoint_error:
            error = f"{error} {endpoint_error}".strip()
        return APIConfigStatus(
            provider=normalized,
            configured=bool(key_source and not endpoint_error),
            key_source=key_source,
            accepted_key_variables=key_names,
            base_url=base_url,
            base_url_source=base_source,
            error=error,
            note=(
                "The API key and Base URL must belong to the same Alibaba Cloud region. "
                "The default endpoint is Singapore; set DASHSCOPE_API_BASE for Beijing or Virginia."
            ),
        )

    if normalized == "gemini":
        key_names = ("GEMINI_API_KEY", "GOOGLE_API_KEY")
        key_source, _ = _first_nonempty(env, key_names)
        base_source, base_url = _first_nonempty(env, ("GEMINI_API_BASE",))
        api_mode = (env.get("GEMINI_API_MODE") or "").strip().lower()
        mode_error = ""
        if api_mode and api_mode not in {"google", "openai"}:
            mode_error = "GEMINI_API_MODE must be either 'google' or 'openai'."
        note = ""
        if api_mode == "openai":
            note = "Using an OpenAI-compatible Gemini endpoint with Bearer authentication."
        return APIConfigStatus(
            provider=normalized,
            configured=bool(key_source and not mode_error),
            key_source=key_source,
            accepted_key_variables=key_names,
            base_url=base_url or GEMINI_DEFAULT_BASE_URL,
            base_url_source=base_source,
            error=mode_error or ("" if key_source else "Gemini calls require GEMINI_API_KEY (or GOOGLE_API_KEY)."),
            note=note,
        )

    if normalized == "huggingface":
        key_names = ("HF_TOKEN", "HF_API_KEY")
        key_source, _ = _first_nonempty(env, key_names)
        base_source, base_url = _first_nonempty(env, ("HF_API_BASE",))
        base_url = base_url or HF_DEFAULT_BASE_URL
        return APIConfigStatus(
            provider=normalized,
            configured=bool(key_source and base_url),
            key_source=key_source,
            accepted_key_variables=key_names,
            base_url=base_url,
            base_url_source=base_source,
            error="" if key_source else "Hugging Face calls require HF_TOKEN (or HF_API_KEY).",
            note="The default uses Hugging Face's OpenAI-compatible Inference Providers router.",
        )

    return APIConfigStatus(
        provider=normalized or "unknown",
        configured=False,
        key_source="",
        accepted_key_variables=(),
        error=f"Unknown provider: {provider!r}",
    )
