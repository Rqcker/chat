"""Gemini client (paper: LLM_text and LLM_audio are Gemini models).

Uses the Generative Language REST API so no vendor SDK is required. The API key
is read from the environment by default (see :class:`chat.config.LLMConfig`).
"""

from __future__ import annotations

import os
from typing import Optional

from .base import LLMClient


class GeminiClient(LLMClient):
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        api_key_env: str = "GEMINI_API_KEY",
        endpoint: str = "https://generativelanguage.googleapis.com/v1beta/models",
        temperature: float = 0.7,
        max_output_tokens: int = 4096,
        timeout: int = 120,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get(api_key_env)
        if not self.api_key:
            raise ValueError(
                f"no Gemini API key: pass api_key or set ${api_key_env}"
            )
        self.endpoint = endpoint.rstrip("/")
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.timeout = timeout

    def generate(self, system: str, user: str, **params) -> str:
        import requests  # imported lazily so importing this module is cheap

        url = f"{self.endpoint}/{self.model}:generateContent"
        headers = {"x-goog-api-key": self.api_key}
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {
                "temperature": params.get("temperature", self.temperature),
                "maxOutputTokens": params.get("max_output_tokens", self.max_output_tokens),
            },
        }
        try:
            resp = requests.post(
                url, headers=headers, json=body, timeout=params.get("timeout", self.timeout)
            )
        except requests.RequestException as exc:  # transient network/timeout -> retryable
            raise RuntimeError(f"Gemini request failed: {exc}") from exc
        if resp.status_code >= 400:
            raise RuntimeError(f"Gemini API HTTP {resp.status_code}: {resp.text[:1000]}")
        data = resp.json()
        return self._extract_text(data)

    @staticmethod
    def _extract_text(data: dict) -> str:
        feedback = data.get("promptFeedback") or {}
        if feedback.get("blockReason"):
            raise RuntimeError(f"Gemini blocked the prompt: {feedback}")
        candidates = data.get("candidates") or []
        if not candidates:
            raise RuntimeError(f"Gemini returned no candidates: {data}")
        candidate = candidates[0]
        reason = candidate.get("finishReason")
        parts = candidate.get("content", {}).get("parts", [])
        text = "".join(part.get("text", "") for part in parts).strip()
        if reason == "MAX_TOKENS":
            raise RuntimeError(
                "Gemini stopped at MAX_TOKENS; increase max_output_tokens or "
                "shorten the segment"
            )
        if reason in {"SAFETY", "RECITATION", "PROHIBITED_CONTENT"}:
            raise RuntimeError(f"Gemini stopped for {reason}: {candidate}")
        if not text:
            raise RuntimeError(f"Gemini returned empty text (finishReason={reason}): {data}")
        return text
