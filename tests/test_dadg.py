"""Unit tests for Dyadic Audio Dialogue Generation / IAR deterministic core.

Run directly:  python3 tests/test_dadg.py
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.dirname(HERE)
sys.path.insert(0, CODE_ROOT)

from chat.config import Config  # noqa: E402
from chat.dadg import active_speaker, build_audio_units, listener, plan_audio  # noqa: E402
from chat.dadg.iar import assign_timeline, sample_interactive_words  # noqa: E402
from chat.schema import DialogueScript, IdentityDescriptor, Turn  # noqa: E402


def make_script(num_turns, emotions=None):
    turns = [
        Turn(
            index=k,
            speaker=("i" if k % 2 == 1 else "j"),
            text=f"t{k}",
            emotion=(emotions[k - 1] if emotions else None),
        )
        for k in range(1, num_turns + 1)
    ]
    return DialogueScript(
        index=1,
        turns=turns,
        identity_i=IdentityDescriptor("i", "m"),
        identity_j=IdentityDescriptor("j", "w"),
    )


def test_active_listener_parity():
    assert active_speaker(1) == "i" and listener(1) == "j"
    assert active_speaker(2) == "j" and listener(2) == "i"


def test_sample_iw_extremes_and_seed():
    assert sample_interactive_words(5, 0.0, seed=1) == [None] * 5
    words = sample_interactive_words(5, 1.0, seed=1)
    assert all(w is not None for w in words)
    # reproducible for a given seed, different across seeds (with mid probability)
    assert sample_interactive_words(20, 0.5, seed=7) == sample_interactive_words(20, 0.5, seed=7)
    assert sample_interactive_words(20, 0.5, seed=7) != sample_interactive_words(20, 0.5, seed=8)


def test_sample_iw_validates():
    for bad in (-0.1, 1.1):
        try:
            sample_interactive_words(3, bad)
            assert False, "expected ValueError"
        except ValueError:
            pass


def test_build_units_structure():
    script = make_script(4)
    iws = ["hmm", None, "okay", None]  # turns 1..4
    units_i, units_j = build_audio_units(script, iws)

    assert [u.turn for u in units_i] == [1, 2, 3, 4]
    assert [u.turn for u in units_j] == [1, 2, 3, 4]
    # S^i speaks on odd turns, listens (silence here) on even turns
    assert [u.kind for u in units_i] == ["sentence", "silence", "sentence", "silence"]
    # S^j interjects on odd turns, speaks on even turns
    assert [u.kind for u in units_j] == ["interactive_word", "sentence", "interactive_word", "sentence"]
    assert units_j[0].text == "hmm" and units_j[2].text == "okay"
    assert units_i[0].text == "t1" and units_j[1].text == "t2"


def test_build_units_length_mismatch_raises():
    script = make_script(4)
    try:
        build_audio_units(script, ["hmm"])  # wrong length
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_timeline_alignment():
    script = make_script(4)
    iws = ["hmm", None, "okay", None]
    units_i, units_j = build_audio_units(script, iws)
    total = assign_timeline(units_i, units_j, [1.0, 2.0, 3.0, 4.0], iw_duration=0.5)

    assert total == 10.0
    # both speakers share each turn slot
    assert (units_i[1].timespan.start, units_i[1].timespan.end) == (1.0, 3.0)
    assert (units_j[1].timespan.start, units_j[1].timespan.end) == (1.0, 3.0)
    # sentence/silence span the whole slot
    assert (units_i[0].timespan.start, units_i[0].timespan.end) == (0.0, 1.0)
    # interactive word is short and placed mid-slot (reactive, not at onset)
    assert abs(units_j[0].timespan.start - 0.25) < 1e-9 and abs(units_j[0].timespan.end - 0.75) < 1e-9
    assert abs(units_j[2].timespan.start - 4.25) < 1e-9 and abs(units_j[2].timespan.end - 4.75) < 1e-9


def test_timeline_turn_gaps():
    script = make_script(3)
    units_i, units_j = build_audio_units(script, [None, None, None])
    total = assign_timeline(units_i, units_j, [1.0, 1.0, 1.0], turn_gaps=[0.2, 0.4, 9.9])
    # gaps after turns 1 and 2 only; NO gap after the final turn (9.9 ignored)
    assert abs(total - 3.6) < 1e-9
    assert (units_i[1].timespan.start, units_i[1].timespan.end) == (1.2, 2.2)
    assert (units_i[2].timespan.start, units_i[2].timespan.end) == (2.6, 3.6)
    try:
        assign_timeline(units_i, units_j, [1.0, 1.0, 1.0], turn_gaps=[0.1])
        assert False, "expected ValueError for wrong gap count"
    except ValueError:
        pass


def test_timeline_iw_clamped_to_slot():
    script = make_script(2)
    units_i, units_j = build_audio_units(script, [None, "yeah"])
    # turn 2 slot is only 0.3s, shorter than iw_duration 0.5 -> clamp
    assign_timeline(units_i, units_j, [1.0, 0.3], iw_duration=0.5)
    assert units_i[1].kind == "interactive_word"
    assert abs(units_i[1].timespan.duration - 0.3) < 1e-9  # iw clamped to slot length


def test_plan_audio_deterministic():
    cfg = Config()
    cfg.iar.p_inter = 0.5
    cfg.iar.seed = 3
    script = make_script(8)
    plan_a = plan_audio(script, cfg)
    plan_b = plan_audio(script, cfg)
    assert plan_a.script_index == 1
    assert len(plan_a.units_i) == 8 and len(plan_a.units_j) == 8
    # deterministic for the same config/seed
    assert [u.kind for u in plan_a.units_j] == [u.kind for u in plan_b.units_j]


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
