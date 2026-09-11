"""Unit tests for the silent-region split in scripts/silence_stability.py.

The statistic itself (median absolute deviation of inter-ocular-normalised mouth opening)
is two numpy calls; what needs testing is the SPLIT, because it must be the exact
complement of the speech gate every LSE number in this project was scored under. A split
that drifts from `lse_video.speech_segments` would silently make the two families of
numbers non-comparable.

`soundfile` is stubbed so this runs anywhere.

Run directly:  python3 tests/test_silence_stability.py
"""

import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.dirname(HERE)
sys.path.insert(0, CODE_ROOT)
sys.path.insert(0, os.path.join(CODE_ROOT, "scripts"))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

import conftest as _conftest  # noqa: E402

SR = _conftest.SR


def _load():
    """Import the modules under test against the shared `soundfile` stub.

    The stub comes from `tests/conftest.py`. It used to be built here, and
    `tests/test_lse_segments.py` built its own; because `lse_video` resolves
    `sys.modules["soundfile"]` inside the function at call time, whichever file
    imported second replaced the other's stub and the first file's tests then looked
    their audio up in the wrong dictionary. Running either file alone passed, so the
    breakage only showed in a full-suite run.

    `lse_video` is also withdrawn from `sys.modules` once `silence_stability` has
    executed its `from lse_video import ...`, so the plain name does not leak either.
    """
    _conftest.install()

    def _mod(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        m = importlib.util.module_from_spec(spec)
        sys.modules[name] = m
        spec.loader.exec_module(m)
        return m

    saved_lse = sys.modules.get("lse_video")
    try:
        # silence_stability imports `from lse_video import ...`, so register that name first
        lse = _mod("lse_video", os.path.join(CODE_ROOT, "scripts", "lse_video.py"))
        ss = _mod("_silence_stability_under_test",
                  os.path.join(CODE_ROOT, "scripts", "silence_stability.py"))
    finally:
        if saved_lse is None:
            sys.modules.pop("lse_video", None)
        else:
            sys.modules["lse_video"] = saved_lse
    return ss, lse


SS, LV = _load()


def _make(pattern, key, seed=0):
    """Register a fake wav under a file-prefixed key in the shared registry."""
    rng = np.random.default_rng(seed)
    parts = []
    for is_speech, dur in pattern:
        n = int(SR * dur)
        parts.append(rng.normal(0, 0.3, n) if is_speech else rng.normal(0, 0.0005, n))
    _conftest.register(f"sil:{key}", np.concatenate(parts))
    return sum(d for _, d in pattern)


def _key(name):
    return f"sil:{name}"


TURNS = [(1, 2.0), (0, 2.0), (1, 2.5), (0, 1.5), (1, 2.0)]


def test_silence_and_speech_do_not_overlap():
    total = _make(TURNS, "t1")
    speech = LV.speech_segments(_key("t1"))
    silent = SS.silent_segments(_key("t1"), total)
    for sa, sb in silent:
        for pa, pb in speech:
            assert sb <= pa or sa >= pb, f"silent {sa,sb} overlaps speech {pa,pb}"


def test_silence_segments_are_ordered_and_inside_the_clip():
    total = _make(TURNS, "t2")
    silent = SS.silent_segments(_key("t2"), total)
    assert silent, "turn-taking fixture must contain scorable silence"
    for a, b in silent:
        assert b > a
        assert a >= -1e-9 and b <= total + 1e-9
    for (_, b0), (a1, _) in zip(silent, silent[1:]):
        assert a1 >= b0


def test_short_silences_are_dropped():
    """THEval's 300 ms floor: a brief inter-word gap is not a scorable silent region."""
    total = _make([(1, 2.0), (0, 0.1), (1, 2.0)], "t3")
    silent = SS.silent_segments(_key("t3"), total, min_silence=0.30)
    for a, b in silent:
        assert b - a >= 0.30 - 1e-9


def test_all_speech_yields_no_silence():
    total = _make([(1, 8.0)], "t4")
    assert SS.silent_segments(_key("t4"), total) == []


def test_leading_and_trailing_silence_are_captured():
    total = _make([(0, 3.0), (1, 2.0), (0, 3.0)], "t5")
    silent = SS.silent_segments(_key("t5"), total)
    assert silent, "leading and trailing silence must be scorable"
    assert silent[0][0] < 1e-6, "leading silence should start at 0"
    assert abs(silent[-1][1] - total) < 1e-6, "trailing silence should reach the clip end"


def test_min_silence_threshold_is_honoured():
    total = _make([(1, 2.0), (0, 0.5), (1, 2.0)], "t6")
    lenient = SS.silent_segments(_key("t6"), total, min_silence=0.05)
    strict = SS.silent_segments(_key("t6"), total, min_silence=2.00)
    assert len(lenient) >= len(strict)
    for a, b in strict:
        assert b - a >= 2.00 - 1e-9


def test_split_uses_the_same_gate_as_lse():
    """Guard against the two families of numbers drifting apart."""
    assert SS.speech_segments is LV.speech_segments


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
