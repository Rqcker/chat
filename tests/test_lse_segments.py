"""Unit tests for the LSE segment selectors in scripts/lse_video.py.

Covers `speech_segments` (the RMS gate this project has always used) and
`length_matched_segments` (its duration-controlled control arm, added 2026-08-10).

The control exists because a plain gated-vs-ungated comparison confounds two effects:
the gate removes silence AND it shortens the scored material, and the 2026-08-08
duration control measured length alone moving LSE-C by 1.560, comparable to the gating
effect being attributed. The invariant that makes the control valid is that it keeps the
gate's segment COUNT and each segment's DURATION while moving where they sit, so these
tests assert exactly that.

`soundfile` is stubbed so this runs anywhere; the functions under test are arithmetic
over an RMS envelope and need no real I/O.

Run directly:  python3 tests/test_lse_segments.py
"""

import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.dirname(HERE)
sys.path.insert(0, CODE_ROOT)
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

import conftest as _conftest  # noqa: E402

SR = _conftest.SR


def _load_module():
    """Import lse_video with soundfile stubbed and argparse side effects contained.

    The stub comes from `tests/conftest.py` rather than being built here. Both this
    file and `tests/test_silence_stability.py` exercise `lse_video`, which resolves
    `sys.modules["soundfile"]` inside the function at call time, so a per-file stub
    meant the second file to import replaced the first file's and its tests then
    looked their audio up in the wrong dictionary.
    """
    _conftest.install()
    spec = importlib.util.spec_from_file_location(
        "_lse_video_under_test", os.path.join(CODE_ROOT, "scripts", "lse_video.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_lse_video_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


LV = _load_module()


def _make(pattern, key, seed=0):
    """pattern = [(is_speech, seconds), ...] -> registers a fake wav under `key`.

    Keys are prefixed so they cannot collide with another test file's fixtures in
    the shared registry.
    """
    rng = np.random.default_rng(seed)
    parts = []
    for is_speech, dur in pattern:
        n = int(SR * dur)
        parts.append(rng.normal(0, 0.3, n) if is_speech else rng.normal(0, 0.0005, n))
    _conftest.register(f"lse:{key}", np.concatenate(parts))
    return sum(d for _, d in pattern)


def _key(name):
    return f"lse:{name}"


TURN_TAKING = [(1, 2.0), (0, 2.0), (1, 2.5), (0, 1.5), (1, 2.0)]


def test_gate_finds_the_speech_turns():
    _make(TURN_TAKING, "turns")
    segs = LV.speech_segments(_key("turns"))
    assert len(segs) == 3, segs
    # segments are ordered, disjoint and non-empty
    for a, b in segs:
        assert b > a
    for (_, b0), (a1, _) in zip(segs, segs[1:]):
        assert a1 >= b0


def test_gate_drops_segments_under_min_seg():
    # 0.4 s of speech is below MIN_SEG (1.2 s) even after 2*PAD of padding
    _make([(0, 3.0), (1, 0.4), (0, 3.0)], "blip")
    assert LV.speech_segments(_key("blip")) == []


def test_length_matched_preserves_count_and_total_duration():
    total = _make(TURN_TAKING, "turns2")
    gated = LV.speech_segments(_key("turns2"))
    matched = LV.length_matched_segments(_key("turns2"))
    assert len(matched) == len(gated)
    dur_gated = sum(b - a for a, b in gated)
    dur_matched = sum(b - a for a, b in matched)
    # equal up to the clip-end clamp, which cannot score audio that does not exist
    assert dur_matched <= dur_gated + 1e-9
    assert abs(dur_matched - min(dur_gated, total)) < 1e-6


def test_length_matched_preserves_each_individual_duration():
    _make([(1, 1.5), (0, 2.0), (1, 3.0), (0, 2.0), (1, 2.0)], "varied")
    gated = LV.speech_segments(_key("varied"))
    matched = LV.length_matched_segments(_key("varied"))
    assert len(gated) >= 2, "fixture should produce several segments"
    for (ga, gb), (ma, mb) in zip(gated, matched):
        assert abs((gb - ga) - (mb - ma)) < 1e-6


def test_length_matched_stays_inside_the_clip():
    total = _make([(0, 6.0), (1, 2.0)], "tail")  # speech only at the very end
    matched = LV.length_matched_segments(_key("tail"))
    assert matched, "gate finds speech here, so the control must too"
    for a, b in matched:
        assert a >= -1e-9
        assert b <= total + 1e-9
        assert b > a


def test_length_matched_actually_moves_the_segments():
    """The point of the control: same duration, different placement."""
    _make([(0, 6.0), (1, 2.0)], "tail2")
    gated = LV.speech_segments(_key("tail2"))
    matched = LV.length_matched_segments(_key("tail2"))
    # the single gated segment sits at the end; the control centres it instead
    assert abs(gated[0][0] - matched[0][0]) > 1.0, (gated, matched)


def test_length_matched_empty_when_gate_is_empty():
    """Callers fall back to whole-video identically in both modes."""
    _make([(0, 3.0), (1, 0.4), (0, 3.0)], "blip2")
    assert LV.speech_segments(_key("blip2")) == []
    assert LV.length_matched_segments(_key("blip2")) == []


def test_length_matched_spreads_over_the_timeline():
    """With several segments the control should sample early, middle and late."""
    total = _make([(1, 1.5), (0, 1.0)] * 5, "many")
    matched = LV.length_matched_segments(_key("many"))
    assert len(matched) >= 3
    centres = [(a + b) / 2 for a, b in matched]
    assert centres == sorted(centres)
    assert centres[0] < total / 3
    assert centres[-1] > 2 * total / 3


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
