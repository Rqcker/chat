"""Dyadic Audio Dialogue Generation (paper Sec. 4.1-4.2).

Organises a dialogue into per-speaker, interactive, emotion-aware, time-stamped
audio units (the IAR block). Waveform synthesis (XTTS) is a later GPU phase.
"""

from .organise import active_speaker, build_audio_units, listener
from .plan import plan_audio, run_iar

__all__ = ["plan_audio", "run_iar", "build_audio_units", "active_speaker", "listener"]
