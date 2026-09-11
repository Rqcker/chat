"""Unit tests for IAR LLM_audio refinement (paper Sec. 4.2).

Run directly:  python3 tests/test_iar_refine.py
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.dirname(HERE)
sys.path.insert(0, CODE_ROOT)

from chat.config import Config  # noqa: E402
from chat.dadg.iar.refine import (  # noqa: E402
    _clamp_float,
    _parse_emotion,
    refine_audio_plan,
    refine_dialogue,
)
from chat.dadg.plan import plan_audio, run_iar  # noqa: E402
from chat.schema import DialogueScript, IdentityDescriptor, Turn  # noqa: E402


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate(self, system, user, **params):
        self.calls.append((system, user, params))
        return self.responses.pop(0)


def make_script(num_turns):
    turns = [
        Turn(index=k, speaker=("i" if k % 2 == 1 else "j"), text=f"t{k}", emotion="neutral")
        for k in range(1, num_turns + 1)
    ]
    return DialogueScript(
        index=1, turns=turns,
        identity_i=IdentityDescriptor("i", "m"), identity_j=IdentityDescriptor("j", "w"),
    )


def refine_json(num_turns, se="studio", skip=(), category="happy"):
    turns = []
    for k in range(1, num_turns + 1):
        if k in skip:
            continue
        turns.append({
            "index": k,
            "text": f"r{k}",
            "emotion": {"category": category, "rate": 1.2, "pitch": 0.1, "energy": 1.0, "pauses": 0.2},
        })
    return json.dumps({"sound_environment": se, "turns": turns})


def test_clamp_float():
    assert _clamp_float(9.0, 0.5, 2.0, 1.0) == 2.0
    assert _clamp_float(-9.0, 0.5, 2.0, 1.0) == 0.5
    assert _clamp_float(1.3, 0.5, 2.0, 1.0) == 1.3
    assert _clamp_float("bad", 0.5, 2.0, 1.0) == 1.0
    assert _clamp_float(None, 0.5, 2.0, 1.0) == 1.0


def test_parse_emotion_clamps_and_defaults():
    e = _parse_emotion({"category": "HAPPY", "rate": 9, "pitch": -9, "energy": -1, "pauses": 5})
    assert e.category == "happy"
    assert e.prosody.rate == 2.0 and e.prosody.pitch == -1.0
    assert e.prosody.energy == 0.0 and e.prosody.pauses == 1.0
    e2 = _parse_emotion("not a dict")
    assert e2.category == "neutral" and e2.prosody.rate == 1.0


def test_refine_dialogue_parses():
    cfg = Config()
    script = make_script(4)
    ref = refine_dialogue(script, FakeLLM([refine_json(4, se="cafe")]), cfg)
    assert len(ref.turns) == 4
    assert ref.sound_environment.name == "cafe"
    assert ref.fallback_indices == []
    assert ref.turns[0].text == "r1" and ref.turns[0].emotion.category == "happy"
    assert abs(ref.turns[0].emotion.prosody.rate - 1.2) < 1e-9


def test_refine_dialogue_fallback_for_missing_turn():
    cfg = Config()
    script = make_script(4)
    ref = refine_dialogue(script, FakeLLM([refine_json(4, skip=(3,))]), cfg)
    assert ref.fallback_indices == [3]
    by = ref.by_turn()
    assert by[3].text == "t3"                 # original text kept
    assert by[3].emotion.category == "neutral"  # neutral fallback emotion


def test_refine_dialogue_retries_then_raises():
    cfg = Config()
    cfg.iar.max_retries = 1  # -> 2 attempts
    script = make_script(2)
    llm = FakeLLM(["no json here", "still nothing"])
    try:
        refine_dialogue(script, llm, cfg)
        assert False, "expected ValueError"
    except ValueError:
        pass
    assert len(llm.calls) == 2  # both attempts consumed


def test_refine_dialogue_empty_turns_raises():
    cfg = Config()
    cfg.iar.max_retries = 0
    script = make_script(2)
    llm = FakeLLM([json.dumps({"sound_environment": "studio", "turns": []})])
    try:
        refine_dialogue(script, llm, cfg)
        assert False, "expected ValueError for no usable turns"
    except ValueError:
        pass


def test_refine_audio_plan_attaches():
    cfg = Config()
    cfg.iar.p_inter = 1.0  # every listener slot becomes an interactive word
    script = make_script(4)
    plan = plan_audio(script, cfg)
    ref = refine_dialogue(script, FakeLLM([refine_json(4)]), cfg)
    refine_audio_plan(plan, ref)

    assert plan.sound_environment.name == "studio"
    # S^i sentences are on odd turns (index 1, 3)
    s_i = [u for u in plan.units_i if u.kind == "sentence"]
    assert [u.text for u in s_i] == ["r1", "r3"]
    assert all(u.emotion.category == "happy" for u in s_i)
    # interactive-word units are empathic back-channels: they echo the turn speaker's
    # emotion ("happy") with prosody attenuated toward neutral (default reactivity 0.5)
    iw = [u for u in plan.units_i + plan.units_j if u.kind == "interactive_word"]
    assert iw and all(u.emotion.category == "happy" for u in iw)
    assert all(abs(u.emotion.prosody.rate - 1.1) < 1e-9 for u in iw)    # 1.0 + 0.5*(1.2-1.0)
    assert all(abs(u.emotion.prosody.pitch - 0.05) < 1e-9 for u in iw)  # 0.0 + 0.5*(0.1-0.0)


def test_run_iar_end_to_end():
    cfg = Config()
    cfg.iar.p_inter = 0.5
    script = make_script(6)
    plan = run_iar(script, FakeLLM([refine_json(6)]), cfg)
    assert plan.sound_environment.name == "studio"
    assert len(plan.units_i) == 6 and len(plan.units_j) == 6
    sentences = [u for u in plan.units_i + plan.units_j if u.kind == "sentence"]
    assert all(u.emotion is not None for u in sentences)


def test_backchannel_reactivity_levels():
    from chat.dadg.iar.refine import _backchannel_emotion
    from chat.schema import Emotion, Prosody

    sp = Emotion("excited", Prosody(rate=1.4, pitch=0.4, energy=1.6, pauses=0.2))
    n = _backchannel_emotion(sp, 0.0)  # neutral acknowledgement
    assert n.category == "neutral" and n.prosody.rate == 1.0 and n.prosody.energy == 1.0
    f = _backchannel_emotion(sp, 1.0)  # full echo
    assert f.category == "excited" and f.prosody.rate == 1.4 and f.prosody.pitch == 0.4
    h = _backchannel_emotion(sp, 0.5)  # halfway from neutral toward the speaker
    assert abs(h.prosody.rate - 1.2) < 1e-9 and abs(h.prosody.energy - 1.3) < 1e-9
    assert _backchannel_emotion(None, 1.0).category == "neutral"  # nothing to react to
    assert _backchannel_emotion(sp, 5.0).prosody.rate == 1.4      # reactivity clamps to 1


def test_clamp_float_nonfinite_and_bool():
    assert _clamp_float(float("nan"), 0.5, 2.0, 1.0) == 1.0
    assert _clamp_float(float("inf"), 0.5, 2.0, 1.0) == 1.0
    assert _clamp_float(float("-inf"), 0.5, 2.0, 1.0) == 1.0
    assert _clamp_float(True, 0.5, 2.0, 1.0) == 1.0
    assert _clamp_float(False, 0.5, 2.0, 1.0) == 1.0


def test_parse_emotion_null_category():
    assert _parse_emotion({"category": None, "rate": 1.0}).category == "neutral"
    assert _parse_emotion({}).category == "neutral"


def test_refine_null_text_and_se_default():
    cfg = Config()
    script = make_script(2)
    resp = json.dumps({"sound_environment": None, "turns": [
        {"index": 1, "text": None, "emotion": {"category": "happy"}},
        {"index": 2, "text": "r2", "emotion": {"category": "calm"}},
    ]})
    ref = refine_dialogue(script, FakeLLM([resp]), cfg)
    assert ref.sound_environment.name == "studio"   # JSON null -> default
    assert 1 in ref.fallback_indices                # null text -> fallback, not "None"
    assert ref.by_turn()[1].text == "t1"
    assert ref.by_turn()[2].text == "r2"


def test_refine_quoted_and_float_index():
    cfg = Config()
    script = make_script(2)
    resp = json.dumps({"sound_environment": "studio", "turns": [
        {"index": "1", "text": "r1", "emotion": {"category": "happy"}},
        {"index": 2.0, "text": "r2", "emotion": {"category": "calm"}},
    ]})
    ref = refine_dialogue(script, FakeLLM([resp]), cfg)
    assert ref.fallback_indices == []
    assert ref.by_turn()[1].text == "r1" and ref.by_turn()[2].text == "r2"


def test_refine_duplicate_index_first_wins():
    cfg = Config()
    script = make_script(2)
    resp = json.dumps({"sound_environment": "studio", "turns": [
        {"index": 1, "text": "first", "emotion": {"category": "happy"}},
        {"index": 1, "text": "second", "emotion": {"category": "sad"}},
        {"index": 2, "text": "r2", "emotion": {"category": "calm"}},
    ]})
    ref = refine_dialogue(script, FakeLLM([resp]), cfg)
    assert ref.by_turn()[1].text == "first"


def test_refine_fallback_preserves_original_emotion():
    cfg = Config()
    turns = [
        Turn(index=1, speaker="i", text="t1", emotion="happy"),
        Turn(index=2, speaker="j", text="t2", emotion="sad"),
    ]
    script = DialogueScript(
        index=1, turns=turns,
        identity_i=IdentityDescriptor("i", "m"), identity_j=IdentityDescriptor("j", "w"),
    )
    resp = json.dumps({"sound_environment": "studio", "turns": [
        {"index": 1, "text": "r1", "emotion": {"category": "excited"}},
    ]})  # turn 2 missing -> fallback must keep original "sad"
    ref = refine_dialogue(script, FakeLLM([resp]), cfg)
    assert ref.fallback_indices == [2]
    assert ref.by_turn()[2].emotion.category == "sad"


def test_run_iar_call_count():
    cfg = Config()
    script = make_script(4)
    llm = FakeLLM([refine_json(4)])
    run_iar(script, llm, cfg)
    assert len(llm.calls) == 1


def test_refine_passes_iar_temperature():
    cfg = Config()
    script = make_script(2)
    llm = FakeLLM([refine_json(2)])
    refine_dialogue(script, llm, cfg)
    assert llm.calls[0][2].get("temperature") == cfg.iar.temperature


def test_refine_retries_on_runtime_error():
    cfg = Config()
    cfg.iar.max_retries = 1  # -> 2 attempts
    script = make_script(2)
    good = refine_json(2)

    class FlakyLLM:
        def __init__(self):
            self.n = 0

        def generate(self, system, user, **params):
            self.n += 1
            if self.n == 1:
                raise RuntimeError("transient 503 from API")
            return good

    ref = refine_dialogue(script, FlakyLLM(), cfg)
    assert len(ref.turns) == 2  # recovered after the transient error


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
