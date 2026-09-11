"""Dyadic organisation of a dialogue into per-speaker audio units.

A turn k is spoken by the active speaker (odd k -> S^i, even k -> S^j); the other
speaker is the listener, who either emits an interactive word or stays silent for
that turn. This produces the paper's organisation (Sec. 4.2):

    dg^i = {dg(1), iw(2), dg(3), ..., iw(K)}   (S^i speaks on odd turns)
    dg^j = {iw(1), dg(2), iw(3), ..., dg(K)}   (S^j speaks on even turns)

Each returned per-speaker list has one unit per turn, ordered by turn, so the
i-th entries of the two lists belong to the same turn and share its time slot.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from ..schema import AudioUnit, DialogueScript, Emotion


def active_speaker(turn: int) -> str:
    return "i" if turn % 2 == 1 else "j"


def listener(turn: int) -> str:
    return "j" if turn % 2 == 1 else "i"


def build_audio_units(
    script: DialogueScript, interactive_words: Sequence[Optional[str]]
) -> Tuple[List[AudioUnit], List[AudioUnit]]:
    """Build the per-speaker audio-unit lists for a script.

    ``interactive_words[k-1]`` is the listener's interactive word for turn k, or
    ``None`` for silence.
    """
    if len(interactive_words) != script.num_turns:
        raise ValueError(
            f"interactive_words has {len(interactive_words)} entries, "
            f"expected {script.num_turns}"
        )
    units = {"i": [], "j": []}
    for turn in script.turns:
        k = turn.index
        spk = active_speaker(k)
        lis = listener(k)
        emotion = Emotion(turn.emotion) if turn.emotion else None
        units[spk].append(
            AudioUnit(turn=k, speaker=spk, kind="sentence", text=turn.text, emotion=emotion)
        )
        iw = interactive_words[k - 1]
        if iw:
            units[lis].append(
                AudioUnit(turn=k, speaker=lis, kind="interactive_word", text=iw)
            )
        else:
            units[lis].append(AudioUnit(turn=k, speaker=lis, kind="silence"))
    return units["i"], units["j"]
