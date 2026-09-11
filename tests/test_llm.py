"""Tests for the LLM client factory (chat.llm.build_llm)."""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.dirname(HERE)
sys.path.insert(0, CODE_ROOT)

from chat.config import Config  # noqa: E402
from chat.llm import build_llm  # noqa: E402
from chat.llm.gemini import GeminiClient  # noqa: E402


def test_build_llm_tdg_and_iar_consume_config():
    os.environ["GEMINI_API_KEY"] = "test-key"
    cfg = Config()
    tdg_client = build_llm(cfg, role="tdg")
    assert isinstance(tdg_client, GeminiClient)
    assert tdg_client.model == cfg.tdg.model
    assert tdg_client.timeout == cfg.llm.timeout          # llm.timeout is consumed
    assert tdg_client.endpoint == cfg.llm.endpoint.rstrip("/")
    iar_client = build_llm(cfg, role="iar")
    assert iar_client.model == cfg.iar.model              # iar.model is consumed


def test_build_llm_rejects_unknown_provider():
    os.environ["GEMINI_API_KEY"] = "test-key"
    cfg = Config()
    cfg.llm.provider = "openai"
    try:
        build_llm(cfg)
        assert False, "expected ValueError for unsupported provider"
    except ValueError:
        pass


def test_build_llm_openai_compat_provider():
    os.environ["DEEPSEEK_API_KEY"] = "test-key"
    cfg = Config()
    cfg.llm.provider = "openai_compat"
    cfg.llm.endpoint = "https://api.deepseek.com"
    cfg.llm.api_key_env = "DEEPSEEK_API_KEY"
    cfg.tdg.model = "deepseek-chat"
    from chat.llm.openai_compat import OpenAICompatClient

    client = build_llm(cfg, role="tdg")
    assert isinstance(client, OpenAICompatClient)
    assert client.model == "deepseek-chat"
    assert client.endpoint == "https://api.deepseek.com"


def test_openai_compat_extract_text():
    from chat.llm.openai_compat import OpenAICompatClient

    ok = {"choices": [{"finish_reason": "stop", "message": {"content": " hi "}}]}
    assert OpenAICompatClient._extract_text(ok) == "hi"
    for bad, frag in [
        ({"choices": []}, "no choices"),
        ({"choices": [{"finish_reason": "length", "message": {"content": "x"}}]}, "max_tokens"),
        ({"choices": [{"finish_reason": "content_filter", "message": {"content": ""}}]}, "content_filter"),
        ({"choices": [{"finish_reason": "stop", "message": {"content": ""}}]}, "empty text"),
    ]:
        try:
            OpenAICompatClient._extract_text(bad)
            assert False, f"expected RuntimeError for {frag}"
        except RuntimeError as exc:
            assert frag in str(exc)


def test_build_llm_rejects_bad_role():
    os.environ["GEMINI_API_KEY"] = "test-key"
    cfg = Config()
    try:
        build_llm(cfg, role="video")
        assert False, "expected ValueError for bad role"
    except ValueError:
        pass


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL {t.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return failures


if __name__ == "__main__":
    sys.exit(1 if _run() else 0)
