"""Interactive-word sampling (paper Sec. 4.2, Eq. for iw(k)).

    iw(k) = random(W_inter)  if random() < P_inter
            silence          otherwise

Sampling is seeded for reproducibility.
"""

from __future__ import annotations

import random
from typing import List, Optional, Sequence

from ...schema import W_INTER


def sample_interactive_words(
    num_turns: int,
    p_inter: float,
    seed: int = 0,
    vocab: Sequence[str] = W_INTER,
) -> List[Optional[str]]:
    """Return one interactive word (or None for silence) per turn."""
    if num_turns < 0:
        raise ValueError(f"num_turns must be >= 0, got {num_turns}")
    if not 0.0 <= p_inter <= 1.0:
        raise ValueError(f"p_inter must be in [0, 1], got {p_inter}")
    if not vocab:
        raise ValueError("vocab must be non-empty")
    rng = random.Random(seed)
    result: List[Optional[str]] = []
    for _ in range(num_turns):
        result.append(rng.choice(list(vocab)) if rng.random() < p_inter else None)
    return result
