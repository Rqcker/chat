"""OpenAI-compatible chat-completions client (DeepSeek, Zhipu/GLM, etc.).

The paper's LLM_text/LLM_audio are Gemini models; this client is a drop-in
alternative for providers exposing the OpenAI chat-completions API, used when a
Gemini key is unavailable. Error semantics mirror :class:`chat.llm.gemini
.GeminiClient`: transient/network problems and API refusals raise RuntimeError so
the callers' retry loops treat them uniformly.
"""

from __future__ import annotations

import os
from typing import Optional

from .base import LLMClient


class OpenAICompatClient(LLMClient):
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        api_key_env: str = "OPENAI_API_KEY",
        endpoint: str = "https://api.deepseek.com",
        temperature: float = 0.7,
        max_output_tokens: int = 4096,
        timeout: int = 120,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get(api_key_env)
        if not self.api_key:
            raise ValueError(f"no API key: pass api_key or set ${api_key_env}")
        self.endpoint = endpoint.rstrip("/")
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.timeout = timeout

    def generate(self, system: str, user: str, **params) -> str:
        import requests  # imported lazily so importing this module is cheap

        url = f"{self.endpoint}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": params.get("temperature", self.temperature),
            "max_tokens": params.get("max_output_tokens", self.max_output_tokens),
        }
        try:
            resp = requests.post(
                url, headers=headers, json=body, timeout=params.get("timeout", self.timeout)
            )
        except requests.RequestException as exc:  # transient network/timeout -> retryable
            raise RuntimeError(f"LLM request failed: {exc}") from exc
        if resp.status_code >= 400:
            raise RuntimeError(f"LLM API HTTP {resp.status_code}: {resp.text[:1000]}")
        return self._extract_text(resp.json())

    @staticmethod
    def _extract_text(data: dict) -> str:
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"LLM returned no choices: {data}")
        choice = choices[0]
        reason = choice.get("finish_reason")
        text = ((choice.get("message") or {}).get("content") or "").strip()
        if reason == "length":
            raise RuntimeError(
                "LLM stopped at max_tokens; increase max_output_tokens or shorten the segment"
            )
        if reason == "content_filter":
            raise RuntimeError(f"LLM refused for content_filter: {choice}")
        if not text:
            raise RuntimeError(f"LLM returned empty text (finish_reason={reason}): {data}")
        return text
