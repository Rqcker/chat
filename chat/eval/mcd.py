"""MCD: Mel-Cepstral Distortion between reference and generated speech (paper Table 3).

MCD = (10/ln 10) * sqrt(2 * sum_{d>=1} (c_ref[d] - c_gen[d])^2), averaged over frames.
The 0th coefficient (energy) is excluded by convention. Inputs are time-aligned
mel-cepstra / MFCCs; alignment (DTW) and extraction are the caller's responsibility.
Pure DSP, so this runs on CPU.
"""

from __future__ import annotations

import numpy as np

_MCD_CONST = 10.0 / np.log(10.0) * np.sqrt(2.0)


def mel_cepstral_distortion(ref, gen, exclude_c0: bool = True) -> float:
    """ref, gen: (T, D) time-aligned mel-cepstra -> mean per-frame MCD in dB."""
    ref = np.asarray(ref, float)
    gen = np.asarray(gen, float)
    if ref.shape != gen.shape or ref.ndim != 2:
        raise ValueError(f"ref/gen must be aligned (T, D), got {ref.shape} and {gen.shape}")
    if exclude_c0:
        ref, gen = ref[:, 1:], gen[:, 1:]
    diff = ref - gen
    per_frame = _MCD_CONST * np.sqrt((diff ** 2).sum(axis=1))
    return float(per_frame.mean())
