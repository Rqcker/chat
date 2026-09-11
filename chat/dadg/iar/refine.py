"""LLM_audio refinement for IAR (paper Sec. 4.2, outputs (ii) emotion, (iv) SE).

Given a dialogue, an LLM (Gemini) rewrites each turn into natural spoken text and
attaches a structured emotion (discrete category + prosody: rate/pitch/energy/
pauses); it also names one sound environment for the dialogue. Interactive words
(output (i)) are sampled separately (``interactive_words``); timestamps (output
(iii)) are assigned from unit durations (``timeline``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ...config import Config
from ...jsonutil import extract_first_json
from ...llm.base import LLMClient
from ...schema import (
    DialogueScript,
    Emotion,
    InteractiveAudioPlan,
    Prosody,
    SoundEnvironment,
)

# Prosody attribute ranges as (min, max, neutral-default).
RATE_RANGE = (0.5, 2.0, 1.0)
PITCH_RANGE = (-1.0, 1.0, 0.0)
ENERGY_RANGE = (0.0, 2.0, 1.0)
PAUSES_RANGE = (0.0, 1.0, 0.0)

# Default category for a listener back-channel when there is nothing to react to.
BACKCHANNEL_CATEGORY = "neutral"


def _backchannel_emotion(speaker_emotion: Optional[Emotion], reactivity: float) -> Emotion:
    """Empathic back-channel: an attenuated echo of the active speaker's emotion.

    A listener's "uh-huh"/"yeah" entrains to the speaker's affect rather than being
    flat. ``reactivity`` in [0, 1] interpolates the prosody from neutral (0) toward
    the speaker's (1); the speaker's category is adopted whenever reactivity > 0.
    Falls back to a neutral acknowledgement when there is no speaker emotion.
    """
    r = max(0.0, min(1.0, reactivity))
    if r == 0.0 or speaker_emotion is None:
        return Emotion(BACKCHANNEL_CATEGORY)
    sp = speaker_emotion.prosody
    prosody = Prosody(
        rate=RATE_RANGE[2] + r * (sp.rate - RATE_RANGE[2]),
        pitch=PITCH_RANGE[2] + r * (sp.pitch - PITCH_RANGE[2]),
        energy=ENERGY_RANGE[2] + r * (sp.energy - ENERGY_RANGE[2]),
        pauses=PAUSES_RANGE[2] + r * (sp.pauses - PAUSES_RANGE[2]),
    )
    return Emotion(speaker_emotion.category, prosody)

AUDIO_REFINE_SYSTEM = (
    "You are a speech director preparing a two-person dialogue for expressive "
    "text-to-speech. For every turn, rewrite the text as natural spoken English "
    "(conversational, second-person where it fits), keeping the meaning and the "
    "speaker. For every turn also give the speaking speaker's emotion as a single "
    "lowercase category word plus prosody numbers: rate (0.5 slow .. 2.0 fast, 1.0 "
    "normal), pitch (-1.0 low .. 1.0 high, 0.0 normal), energy (0.0 flat .. 2.0 "
    "strong, 1.0 normal), pauses (0.0 none .. 1.0 frequent). Give one overall "
    "sound_environment for the whole dialogue (e.g. studio, cafe, open space). "
    "Reply with only a JSON object of the form: "
    '{"sound_environment": "studio", "turns": [{"index": 1, "text": "...", '
    '"emotion": {"category": "happy", "rate": 1.0, "pitch": 0.0, "energy": 1.0, '
    '"pauses": 0.0}}]}. Include one turns entry per input turn, keyed by its index.'
)


@dataclass
class TurnRefinement:
    index: int
    text: str
    emotion: Emotion


@dataclass
class DialogueRefinement:
    turns: List[TurnRefinement]
    sound_environment: SoundEnvironment
    fallback_indices: List[int] = field(default_factory=list)

    def by_turn(self) -> Dict[int, TurnRefinement]:
        return {t.index: t for t in self.turns}


def _clamp_float(value, lo: float, hi: float, default: float) -> float:
    if isinstance(value, bool):  # bool is a subclass of int; reject explicitly
        return default
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(v):  # NaN/inf -> neutral default, not a clamped extreme
        return default
    return max(lo, min(hi, v))


def _parse_emotion(data) -> Emotion:
    if not isinstance(data, dict):
        return Emotion(BACKCHANNEL_CATEGORY)
    cat_raw = data.get("category")
    category = (cat_raw.strip().lower() if isinstance(cat_raw, str) else "") or "neutral"
    prosody = Prosody(
        rate=_clamp_float(data.get("rate"), *RATE_RANGE),
        pitch=_clamp_float(data.get("pitch"), *PITCH_RANGE),
        energy=_clamp_float(data.get("energy"), *ENERGY_RANGE),
        pauses=_clamp_float(data.get("pauses"), *PAUSES_RANGE),
    )
    return Emotion(category, prosody)


def _build_user_prompt(script: DialogueScript) -> str:
    lines = []
    for t in script.turns:
        label = "A" if t.speaker == "i" else "B"
        hint = f" [{t.emotion}]" if t.emotion else ""
        lines.append(f"{t.index}. {label}{hint}: {t.text}")
    return "Dialogue:\n" + "\n".join(lines)


def _parse_refinement(raw: str, script: DialogueScript) -> DialogueRefinement:
    obj = extract_first_json(raw, "{")
    if not isinstance(obj, dict):
        raise ValueError("expected a JSON object")
    se_raw = obj.get("sound_environment")
    se_name = (se_raw.strip() if isinstance(se_raw, str) else "") or "studio"
    turns_data = obj.get("turns")
    if not isinstance(turns_data, list):
        raise ValueError("refinement missing 'turns' array")

    by_index: Dict[int, tuple] = {}
    for item in turns_data:
        if not isinstance(item, dict):
            continue
        idx = item.get("index")
        if isinstance(idx, bool):  # bool is an int subclass; not a valid index
            continue
        try:
            idx = int(float(idx))  # accept 1, "1", 1.0 (common LLM quirks)
        except (TypeError, ValueError):
            continue
        if idx in by_index:
            continue  # keep the first occurrence of a duplicate index
        text_raw = item.get("text")
        text = text_raw.strip() if isinstance(text_raw, str) else ""  # null -> skip
        if not text:
            continue
        by_index[idx] = (text, _parse_emotion(item.get("emotion")))
    if not by_index:
        raise ValueError("refinement produced no usable turns")

    refined: List[TurnRefinement] = []
    fallback: List[int] = []
    for turn in script.turns:
        if turn.index in by_index:
            text, emotion = by_index[turn.index]
        else:
            text, emotion = turn.text, Emotion(turn.emotion or "neutral")
            fallback.append(turn.index)
        refined.append(TurnRefinement(turn.index, text, emotion))
    return DialogueRefinement(refined, SoundEnvironment(se_name), fallback)


def refine_dialogue(script: DialogueScript, llm: LLMClient, cfg: Config) -> DialogueRefinement:
    """Call LLM_audio to refine text + emotion + sound environment (with retries).

    ``llm`` is the LLM_audio client, e.g. ``chat.llm.build_llm(cfg, role="iar")``
    (which uses ``cfg.iar.model``).
    """
    attempts = max(1, cfg.iar.max_retries + 1)
    last_error: Optional[Exception] = None
    for _ in range(attempts):
        try:
            raw = llm.generate(
                AUDIO_REFINE_SYSTEM, _build_user_prompt(script),
                temperature=cfg.iar.temperature,
                max_output_tokens=cfg.iar.max_output_tokens,
            )
            return _parse_refinement(raw, script)
        except (ValueError, RuntimeError) as exc:  # parse errors AND transient LLM/API errors
            last_error = exc
    raise ValueError(f"LLM_audio refinement failed after retries: {last_error}")


def refine_audio_plan(
    plan: InteractiveAudioPlan,
    refinement: DialogueRefinement,
    backchannel_reactivity: float = 0.5,
) -> InteractiveAudioPlan:
    """Attach refined text + emotion to a plan's units and set its sound environment.

    Sentence units take the refined text and the turn's emotion. An interactive-word
    unit is a listener back-channel, so it takes an empathic emotion that reacts to
    that turn's active speaker (see :func:`_backchannel_emotion`), scaled by
    ``backchannel_reactivity``. Silence units are left untouched. Mutates and returns
    ``plan``.
    """
    by_turn = refinement.by_turn()
    for units in (plan.units_i, plan.units_j):
        for unit in units:
            if unit.kind == "sentence":
                tr = by_turn.get(unit.turn)
                if tr is not None:
                    unit.text = tr.text
                    unit.emotion = tr.emotion
            elif unit.kind == "interactive_word":
                tr = by_turn.get(unit.turn)  # the active speaker at this turn
                speaker_emotion = tr.emotion if tr is not None else None
                unit.emotion = _backchannel_emotion(speaker_emotion, backchannel_reactivity)
    plan.sound_environment = refinement.sound_environment
    return plan
