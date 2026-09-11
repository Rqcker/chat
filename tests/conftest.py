"""Shared `soundfile` stub for the tests that exercise the audio gates.

`tests/test_lse_segments.py` and `tests/test_silence_stability.py` both test code
in `scripts/lse_video.py`, and neither needs real audio: the functions under test
are arithmetic over an RMS envelope. Both therefore install a fake `soundfile`
whose `read` serves arrays from an in-memory dictionary keyed by a made-up path.

Why this file exists. `lse_video.speech_segments` does `import soundfile as sf`
*inside the function*, so it resolves `sys.modules["soundfile"]` at call time, not
at import time. When both test files installed their own stub, whichever imported
second replaced the entry, and the first file's tests then looked their audio up in
the second file's dictionary and died with KeyError. Running either file alone
passed, so the failure only appeared in a full-suite run.

The fix is one stub, shared, backed by a registry that every test file adds to.
Keys are unique per file by convention, and `register` refuses to overwrite a key
that another file already claimed, so a future collision fails loudly here instead
of silently swapping one fixture for another.
"""

import sys
import types

SR = 16000

# key -> (samples, sample_rate). Populated by each test module at collection time.
_REGISTRY = {}


def register(key, samples, sr=SR):
    """Publish one fake wav. Raises if `key` is already taken by another module."""
    if key in _REGISTRY:
        raise KeyError(
            f"audio key {key!r} is already registered; test files must use "
            f"distinct keys so they cannot shadow each other"
        )
    _REGISTRY[key] = (samples, sr)
    return samples


def _read(key):
    try:
        return _REGISTRY[key]
    except KeyError:
        raise KeyError(
            f"no stub audio registered under {key!r}; call "
            f"tests.conftest.register(key, samples) before the code under test "
            f"reads it"
        ) from None


def install():
    """Put the shared stub in `sys.modules` and return it. Idempotent."""
    existing = sys.modules.get("soundfile")
    if getattr(existing, "_is_chat_test_stub", False):
        return existing
    stub = types.ModuleType("soundfile")
    stub.read = _read
    stub._is_chat_test_stub = True
    sys.modules["soundfile"] = stub
    return stub


# Installed at collection time so it is in place before any test module imports
# the code under test.
install()
