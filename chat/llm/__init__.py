"""Language-model clients for CHAT.

The Gemini client lives in :mod:`chat.llm.gemini` and is imported explicitly so
that the lightweight :class:`chat.llm.base.LLMClient` protocol can be used (and
tested) without pulling in network dependencies.
"""

from .base import LLMClient

__all__ = ["LLMClient", "build_llm"]


def build_llm(cfg, role: str = "tdg") -> LLMClient:
    """Construct the configured LLM client for ``role`` ("tdg" or "iar").

    Consumes ``cfg.llm`` (provider, endpoint, timeout, api_key_env) and the model
    from ``cfg.tdg`` / ``cfg.iar``. Providers: "gemini" (paper) and
    "openai_compat" (any OpenAI chat-completions endpoint, e.g. DeepSeek/GLM).
    """
    if role not in ("tdg", "iar"):
        raise ValueError(f"role must be 'tdg' or 'iar', got {role!r}")
    model = cfg.tdg.model if role == "tdg" else cfg.iar.model
    if cfg.llm.provider == "gemini":
        from .gemini import GeminiClient

        return GeminiClient(
            model,
            api_key_env=cfg.llm.api_key_env,
            endpoint=cfg.llm.endpoint,
            timeout=cfg.llm.timeout,
        )
    if cfg.llm.provider == "openai_compat":
        from .openai_compat import OpenAICompatClient

        return OpenAICompatClient(
            model,
            api_key_env=cfg.llm.api_key_env,
            endpoint=cfg.llm.endpoint,
            timeout=cfg.llm.timeout,
        )
    raise ValueError(f"unsupported LLM provider: {cfg.llm.provider!r}")
