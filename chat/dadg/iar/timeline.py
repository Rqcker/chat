"""Timestamp assignment for the dyadic audio plan (paper Sec. 4.2, output (iii)).

Turns are laid out sequentially. Turn k occupies one time slot whose length is the
active speaker's sentence duration; both speakers' units for that turn share the
slot, so the two tracks stay aligned. A sentence (or silence) spans the whole
slot; an interactive word occupies a short span at the slot start.
"""

from __future__ import annotations

from typing import List, Sequence

from ...schema import AudioUnit, TimeSpan


def assign_timeline(
    units_i: List[AudioUnit],
    units_j: List[AudioUnit],
    turn_durations: Sequence[float],
    iw_duration: float = 0.5,
    turn_gaps: Sequence[float] = None,
) -> float:
    """Assign a :class:`TimeSpan` to every unit in place; return total duration.

    ``units_i[idx]`` and ``units_j[idx]`` must both belong to turn ``idx + 1``.
    ``turn_gaps[idx]`` (optional) inserts a silent gap AFTER turn ``idx + 1`` —
    natural turn-taking has ~200 ms gaps, scaled by the speaker's ``pauses``
    prosody; no gap is added after the final turn.
    """
    k = len(turn_durations)
    if not (len(units_i) == len(units_j) == k):
        raise ValueError(
            f"length mismatch: units_i={len(units_i)}, units_j={len(units_j)}, "
            f"durations={k}"
        )
    if iw_duration < 0:
        raise ValueError(f"iw_duration must be >= 0, got {iw_duration}")
    if turn_gaps is not None and len(turn_gaps) != k:
        raise ValueError(f"turn_gaps has {len(turn_gaps)} entries, expected {k}")
    cum = 0.0
    for idx in range(k):
        duration = float(turn_durations[idx])
        if duration < 0:
            raise ValueError(f"turn duration {idx} is negative: {duration}")
        start, end = cum, cum + duration
        for unit in (units_i[idx], units_j[idx]):
            if unit.turn != idx + 1:
                raise ValueError(
                    f"unit out of order at index {idx}: got turn {unit.turn}"
                )
            if unit.kind == "interactive_word":
                iw_dur = min(iw_duration, duration)
                # Start the back-channel partway through the slot so it reacts to the
                # speaker rather than beginning at the same instant (onset).
                iw_start = start + max(0.0, (duration - iw_dur) * 0.5)
                unit.timespan = TimeSpan(iw_start, iw_start + iw_dur)
            else:
                unit.timespan = TimeSpan(start, end)
        cum = end
        if turn_gaps is not None and idx < k - 1:
            gap = float(turn_gaps[idx])
            if gap < 0:
                raise ValueError(f"turn gap {idx} is negative: {gap}")
            cum += gap
    return cum
