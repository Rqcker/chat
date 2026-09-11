"""Build the interactive audio plan for a dialogue script (paper Sec. 4.2).

This covers the deterministic part of IAR: organising the dialogue into per-speaker
units and sampling interactive words. Emotion refinement and the sound environment
(LLM_audio) and waveform synthesis (XTTS) are filled by later stages; timestamps
are assigned once unit durations are known (see :func:`iar.timeline.assign_timeline`).
"""

from __future__ import annotations

from ..config import Config
from ..llm.base import LLMClient
from ..schema import DialogueScript, InteractiveAudioPlan
from .iar.interactive_words import sample_interactive_words
from .iar.refine import refine_audio_plan, refine_dialogue
from .organise import build_audio_units


def plan_audio(script: DialogueScript, cfg: Config) -> InteractiveAudioPlan:
    interactive_words = sample_interactive_words(
        script.num_turns, cfg.iar.p_inter, seed=cfg.iar.seed + script.index
    )
    units_i, units_j = build_audio_units(script, interactive_words)
    return InteractiveAudioPlan(
        script_index=script.index, units_i=units_i, units_j=units_j
    )


def run_iar(script: DialogueScript, llm: LLMClient, cfg: Config) -> InteractiveAudioPlan:
    """Full IAR: deterministic plan (organise + interactive words) followed by
    LLM_audio refinement (refined text + emotion + sound environment)."""
    plan = plan_audio(script, cfg)
    refinement = refine_dialogue(script, llm, cfg)
    return refine_audio_plan(plan, refinement, cfg.iar.backchannel_reactivity)
