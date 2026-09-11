"""Shared data structures for the CHAT pipeline.

Notation follows the paper. Two speakers are S^i and S^j, represented here by the
speaker codes ``"i"`` and ``"j"``. A dialogue script corresponds to one generated
dialogue dg_n with its two textual identity descriptors (ID-txt^i, ID-txt^j).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import List, Optional

# Speaker codes for S^i and S^j.
SPEAKERS = ("i", "j")


@dataclass
class Turn:
    """One conversational turn expressed by one speaker.

    A turn may contain multiple sentences (paper Sec. 4.1). ``index`` is the
    1-based position k of the turn within the dialogue; the speaker alternates by
    parity (odd -> S^i, even -> S^j), matching the odd/even organisation used by
    DADG.
    """

    index: int
    speaker: str
    text: str
    emotion: Optional[str] = None  # coarse hint from TDG; refined later by IAR

    def __post_init__(self) -> None:
        if self.speaker not in SPEAKERS:
            raise ValueError(f"speaker must be one of {SPEAKERS}, got {self.speaker!r}")


@dataclass
class IdentityDescriptor:
    """Textual identity description of a speaker (paper: ID-txt^i / ID-txt^j)."""

    speaker: str
    description: str

    def __post_init__(self) -> None:
        if self.speaker not in SPEAKERS:
            raise ValueError(f"speaker must be one of {SPEAKERS}, got {self.speaker!r}")


@dataclass
class DialogueScript:
    """One generated dialogue dg_n with paired identity descriptors."""

    index: int
    turns: List[Turn] = field(default_factory=list)
    identity_i: Optional[IdentityDescriptor] = None
    identity_j: Optional[IdentityDescriptor] = None
    prompt: str = ""  # originating scenario prompt P

    @property
    def num_turns(self) -> int:
        return len(self.turns)

    def turns_of(self, speaker: str) -> List[Turn]:
        """Return the turns expressed by one speaker (paper: dg_n^i / dg_n^j)."""
        if speaker not in SPEAKERS:
            raise ValueError(f"speaker must be one of {SPEAKERS}, got {speaker!r}")
        return [t for t in self.turns if t.speaker == speaker]

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str) -> None:
        import os

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> "DialogueScript":
        turns = [Turn(**t) for t in data.get("turns", [])]
        id_i = data.get("identity_i")
        id_j = data.get("identity_j")
        return cls(
            index=data["index"],
            turns=turns,
            identity_i=IdentityDescriptor(**id_i) if id_i else None,
            identity_j=IdentityDescriptor(**id_j) if id_j else None,
            prompt=data.get("prompt", ""),
        )

    @classmethod
    def load(cls, path: str) -> "DialogueScript":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


# --- Audio-side structures (DADG / IAR, paper Sec. 4.2) --------------------

# Interactive-word vocabulary (paper: W_inter).
W_INTER = ("uh-huh", "hmm", "okay", "yeah")

# Kinds an audio unit can take.
AUDIO_UNIT_KINDS = ("sentence", "interactive_word", "silence")


@dataclass
class TimeSpan:
    start: float
    end: float

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError(f"TimeSpan end {self.end} < start {self.start}")

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class Prosody:
    """Continuous prosody attributes (paper: rate, pitch, energy, pauses)."""

    rate: float = 1.0
    pitch: float = 0.0
    energy: float = 1.0
    pauses: float = 0.0


@dataclass
class Emotion:
    """Structured emotion descriptor: a discrete category plus prosody."""

    category: str
    prosody: Prosody = field(default_factory=Prosody)


@dataclass
class SoundEnvironment:
    """Sound environment of a dialogue (paper: SE, e.g. studio or open space)."""

    name: str


@dataclass
class AudioUnit:
    """One audio unit for one speaker at a turn: a sentence, an interactive word,
    or a silent gap. ``timespan`` and ``emotion`` are filled by IAR/synthesis."""

    turn: int
    speaker: str
    kind: str
    text: str = ""
    emotion: Optional[Emotion] = None
    timespan: Optional[TimeSpan] = None

    def __post_init__(self) -> None:
        if self.speaker not in SPEAKERS:
            raise ValueError(f"speaker must be one of {SPEAKERS}, got {self.speaker!r}")
        if self.kind not in AUDIO_UNIT_KINDS:
            raise ValueError(f"kind must be one of {AUDIO_UNIT_KINDS}, got {self.kind!r}")


@dataclass
class InteractiveAudioPlan:
    """IAR output: per-speaker interactive, emotion-aware, time-stamped audio units
    for one dialogue script (paper Sec. 4.2), before waveform synthesis."""

    script_index: int
    units_i: List[AudioUnit] = field(default_factory=list)
    units_j: List[AudioUnit] = field(default_factory=list)
    sound_environment: Optional[SoundEnvironment] = None

    def units_for(self, speaker: str) -> List[AudioUnit]:
        if speaker not in SPEAKERS:
            raise ValueError(f"speaker must be one of {SPEAKERS}, got {speaker!r}")
        return self.units_i if speaker == "i" else self.units_j
