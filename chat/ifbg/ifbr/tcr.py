"""Temporal Continuity Refinement (TCR), paper Sec. 4.3.

A half-Gaussian boundary cross-fade that smooths transitions between adjacent
facial-behaviour segments. It touches only a narrow window of ``W`` frames on each
side of a boundary (paper: W=10, segments of ~100 frames), using a
content-independent weight (paper 2026-07-01)

    alpha(tau) = 0.5 * exp(-(tau-1)^2 / (2 sigma^2)),   tau = 1..W,   sigma = W/3

which equals 0.5 at the boundary (tau=1) and decays outwards (3-sigma rule). For
adjacent segments v(k-1) and v(k), with F_{k-1} the number of frames in v(k-1),
the frame tau steps from the boundary is blended on each side (paper Eq. for TCR):

    v(k-1, F_{k-1}-tau+1) = (1-alpha(tau)) * v~(k-1, F_{k-1}-tau+1) + alpha(tau) * v~(k, tau)
    v(k,   tau)           =     alpha(tau) * v~(k-1, F_{k-1}-tau+1) + (1-alpha(tau)) * v~(k, tau)

So the last frame of v(k-1) (tau=1, at the boundary) blends 50/50 with the first
frame of v(k), and the blend weakens with distance from the boundary.

Because ``alpha`` depends only on ``(W, sigma)``, applying the same blend to both
speakers' streams is the symmetric transform the paper uses to preserve mutual
responsiveness. Segments should be longer than ``2*W`` so a segment's leading and
trailing blend windows do not overlap.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np


def gaussian_alpha(window: int, sigma: Optional[float] = None) -> np.ndarray:
    """Half-Gaussian cross-fade weights alpha(tau) = 0.5*exp(-(tau-1)^2/(2 sigma^2))
    for tau = 1..window; alpha(1)=0.5 at the boundary, decaying outwards."""
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    if sigma is None:
        sigma = window / 3.0
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}")
    tau = np.arange(1, window + 1, dtype=np.float64)
    return 0.5 * np.exp(-((tau - 1) ** 2) / (2.0 * sigma ** 2))


def _blend_dtype(*arrays: np.ndarray) -> np.dtype:
    if all(np.issubdtype(a.dtype, np.floating) for a in arrays):
        return np.result_type(*arrays)
    return np.dtype(np.float64)


def blend_boundary(
    seg_prev: np.ndarray,
    seg_cur: np.ndarray,
    window: int,
    sigma: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Blend the boundary between two adjacent segments. Inputs are not modified."""
    prev = np.asarray(seg_prev)
    cur = np.asarray(seg_cur)
    if prev.ndim == 0 or cur.ndim == 0:
        raise ValueError("segments must have a leading time axis")
    t_prev = prev.shape[0]
    if window > t_prev or window > cur.shape[0]:
        raise ValueError(
            f"window {window} exceeds segment length "
            f"(prev={t_prev}, cur={cur.shape[0]})"
        )

    dtype = _blend_dtype(prev, cur)
    alpha = gaussian_alpha(window, sigma).astype(dtype)
    alpha = alpha.reshape((window,) + (1,) * (prev.ndim - 1))

    new_prev = prev.astype(dtype, copy=True)
    new_cur = cur.astype(dtype, copy=True)
    # Pair by distance tau=1..W from the boundary: the last frame of the previous
    # segment (F_{k-1}-tau+1) with the tau-th frame of the current one. Reverse the
    # previous tail so index 0 is the boundary frame, aligning with alpha(1).
    tail = prev[t_prev - window:t_prev][::-1].astype(dtype)   # index 0 = boundary frame
    head = cur[0:window].astype(dtype)                        # index 0 = boundary frame

    new_prev[t_prev - window:t_prev] = ((1.0 - alpha) * tail + alpha * head)[::-1]
    new_cur[0:window] = alpha * tail + (1.0 - alpha) * head
    return new_prev, new_cur


def apply_tcr(
    segments: Sequence[np.ndarray],
    window: int,
    sigma: Optional[float] = None,
) -> List[np.ndarray]:
    """Blend every adjacent pair of segments in one speaker's stream.

    Returns new arrays; inputs are not modified. Use the same ``window`` for both
    speakers so the transform stays symmetric (paper Sec. 4.3).
    """
    out = [np.asarray(s).copy() for s in segments]
    # Interior segments get blended on BOTH boundaries (trailing window from the
    # left neighbour, leading window from the right); if 2*window exceeds the
    # segment length those two regions overlap and get blended twice, silently
    # over-smoothing instead of the paper's clean symmetric cross-fade. First/last
    # segments only touch one boundary each, so blend_boundary's own per-call
    # window<=length check already covers them.
    for k in range(1, len(out) - 1):
        if 2 * window > out[k].shape[0]:
            raise ValueError(
                f"window {window} needs 2*window <= segment length for interior "
                f"segment {k} (length {out[k].shape[0]}) so its leading and "
                "trailing blend windows don't overlap (paper: W=10, ~100-frame "
                "segments)"
            )
    for k in range(1, len(out)):
        out[k - 1], out[k] = blend_boundary(out[k - 1], out[k], window, sigma)
    return out
