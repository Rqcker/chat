"""Unit tests for Textual Dialogue Generation.

Pure-logic tests with a fake LLM (no network). Run directly:

    python3 tests/test_tdg.py

Tests assert behaviour against the paper's intent, not against the current
implementation: speakers must follow the LLM's declared labels under a strictly
alternating (odd S^i / even S^j) structure, segment sizes must stay within the
5-10 range, and conditioning must carry the previous segment's text and sentiment.
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.dirname(HERE)
sys.path.insert(0, CODE_ROOT)

from chat.config import Config  # noqa: E402
from chat.schema import DialogueScript  # noqa: E402
from chat.tdg.dialogue import (  # noqa: E402
    _first_json,
    _segment_sizes,
    generate_dialogue_scripts,
    parse_identities,
    parse_turns,
)


class FakeLLM:
    """Returns canned responses in order; records the prompts it was given."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate(self, system, user, **params):
        self.calls.append((system, user, params))
        return self.responses.pop(0)


def _seg(start, n, emotion="neutral"):
    """Build a JSON array of n alternating A/B turns with content t{start..}."""
    items = []
    for i in range(n):
        speaker = "A" if i % 2 == 0 else "B"
        items.append({"speaker": speaker, "text": f"t{start + i}", "emotion": emotion})
    return json.dumps(items)


# --- JSON extraction -------------------------------------------------------

def test_first_json_handles_trailing_prose():
    assert _first_json('[{"text":"x"}] and then some notes [1]', "[") == [{"text": "x"}]


def test_first_json_returns_first_of_multiple():
    assert _first_json("[1, 2] [3, 4]", "[") == [1, 2]


def test_first_json_skips_invalid_opener():
    # First '{' is invalid (unquoted) -> must skip to the valid object.
    assert _first_json('note {bad} {"i": "m", "j": "w"}', "{") == {"i": "m", "j": "w"}


def test_first_json_fenced():
    raw = '```json\n[{"text": "x"}]\n```'
    assert _first_json(raw, "[") == [{"text": "x"}]


# --- parse_turns -----------------------------------------------------------

def test_parse_turns_clean_alternating():
    turns = parse_turns(_seg(1, 2, emotion="happy"), 1)
    assert [t.speaker for t in turns] == ["i", "j"]
    assert [t.index for t in turns] == [1, 2]
    assert turns[0].emotion == "happy" and turns[0].text == "t1"


def test_parse_turns_honours_declared_at_offset_start():
    # start_index 3 -> expected i, j; declared A, B agree.
    turns = parse_turns(_seg(3, 2), 3)
    assert [t.index for t in turns] == [3, 4]
    assert [t.speaker for t in turns] == ["i", "j"]


def test_parse_turns_raises_on_non_alternating():
    raw = json.dumps([{"speaker": "A", "text": "a1"}, {"speaker": "A", "text": "a2"}])
    try:
        parse_turns(raw, 1)
        assert False, "expected ValueError for non-alternating speakers"
    except ValueError:
        pass


def test_parse_turns_blank_drop_surfaces_misalignment():
    # A dropped empty turn must NOT silently shift B onto S^i; it must raise.
    raw = json.dumps([{"speaker": "A", "text": ""}, {"speaker": "B", "text": "real"}])
    try:
        parse_turns(raw, 1)
        assert False, "expected ValueError when a dropped turn breaks alternation"
    except ValueError:
        pass


def test_parse_turns_limit():
    turns = parse_turns(_seg(1, 4), 1, limit=2)
    assert [t.text for t in turns] == ["t1", "t2"]


# --- identities ------------------------------------------------------------

def test_parse_identities_variants():
    a, b = parse_identities('{"i": "a thoughtful gentleman", "j": "a vibrant lady"}')
    assert a.speaker == "i" and "gentleman" in a.description
    assert b.speaker == "j" and "lady" in b.description
    a2, b2 = parse_identities('prose {"A": "a calm man", "B": "a lively woman"} trailing')
    assert a2.description == "a calm man" and b2.description == "a lively woman"


# --- segment sizing --------------------------------------------------------

def test_segment_sizes_within_range():
    assert _segment_sizes(24, 6, 10) == [8, 8, 8]
    assert _segment_sizes(30, 6, 10) == [10, 10, 10]
    for total in range(12, 41):
        sizes = _segment_sizes(total, 6, 10)
        assert sum(sizes) == total
        assert all(s <= 10 for s in sizes), (total, sizes)
        assert all(s >= 6 for s in sizes), (total, sizes)


def test_segment_sizes_below_min_total():
    # total below the minimum cannot be tiled; a single short segment is returned.
    assert _segment_sizes(4, 6, 10) == [4]


# --- end-to-end stitching --------------------------------------------------

def test_generate_segment_wise_stitching_and_conditioning():
    cfg = Config()
    cfg.tdg.total_turns = 12
    cfg.tdg.segment_min_turns = 6
    cfg.tdg.segment_max_turns = 6  # forces sizes [6, 6]
    cfg.tdg.num_scripts = 1

    responses = [
        '{"i": "a calm man", "j": "a lively woman"}',  # identity
        _seg(1, 6, emotion="happy"),                    # turns 1-6
        _seg(7, 6, emotion="calm"),                     # turns 7-12
    ]
    llm = FakeLLM(responses)
    scripts = generate_dialogue_scripts("two friends catching up", llm, cfg)

    assert len(scripts) == 1
    script = scripts[0]
    assert isinstance(script, DialogueScript)
    assert script.num_turns == 12
    assert [t.speaker for t in script.turns] == ["i", "j"] * 6
    assert [t.index for t in script.turns] == list(range(1, 13))
    assert script.identity_i.description == "a calm man"
    assert len(llm.calls) == 3  # identity + 2 segments

    second_segment_prompt = llm.calls[2][1]
    assert "t6" in second_segment_prompt          # previous segment text fed back
    assert "(happy)" in second_segment_prompt     # previous segment sentiment fed back


def test_turns_of_speaker():
    cfg = Config()
    cfg.tdg.total_turns = 4
    cfg.tdg.segment_min_turns = 4
    cfg.tdg.segment_max_turns = 4
    llm = FakeLLM(['{"i":"man","j":"woman"}', _seg(1, 4)])
    script = generate_dialogue_scripts("scene", llm, cfg)[0]
    assert [t.text for t in script.turns_of("i")] == ["t1", "t3"]
    assert [t.text for t in script.turns_of("j")] == ["t2", "t4"]


def test_save_and_load_roundtrip():
    import tempfile

    cfg = Config()
    cfg.tdg.total_turns = 6
    cfg.tdg.segment_min_turns = 6
    cfg.tdg.segment_max_turns = 10
    llm = FakeLLM(['{"i":"man","j":"woman"}', _seg(1, 6, emotion="happy")])
    script = generate_dialogue_scripts("scene", llm, cfg)[0]

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "dg.json")
        script.save(path)
        loaded = DialogueScript.load(path)
    assert loaded.num_turns == 6
    assert loaded.turns[0].emotion == "happy"
    assert loaded.identity_i.description == "man"


def test_generate_retries_on_runtime_error():
    cfg = Config()
    cfg.tdg.total_turns = 2
    cfg.tdg.segment_min_turns = 2
    cfg.tdg.segment_max_turns = 2
    seg = json.dumps([{"speaker": "A", "text": "a1"}, {"speaker": "B", "text": "b1"}])

    class FlakyLLM:
        def __init__(self):
            self.n = 0

        def generate(self, system, user, **params):
            self.n += 1
            if self.n == 1:
                return '{"i": "a calm man", "j": "a lively woman"}'  # identity
            if self.n == 2:
                raise RuntimeError("transient 503 from API")          # segment attempt 1 fails
            return seg                                                # retry succeeds

    scripts = generate_dialogue_scripts("scene", FlakyLLM(), cfg)
    assert scripts[0].num_turns == 2


def test_parse_turns_requires_speaker_label():
    raw = json.dumps([{"text": "hi, no speaker field"}])
    try:
        parse_turns(raw, 1)
        assert False, "expected ValueError for a turn missing its speaker label"
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
