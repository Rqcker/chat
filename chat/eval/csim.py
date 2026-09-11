"""CSIM: identity similarity between generated and reference faces (paper Table 1).

Mean cosine similarity of paired identity embeddings. Callers supply the embeddings
from a face-recognition backbone (ArcFace, loaded on a GPU).
"""

from __future__ import annotations

import numpy as np


def identity_similarity(pred_embeds, ref_embeds, eps: float = 1e-8) -> float:
    """pred_embeds, ref_embeds: (N, D) paired identity embeddings -> mean cosine."""
    p = np.asarray(pred_embeds, float)
    r = np.asarray(ref_embeds, float)
    if p.shape != r.shape or p.ndim != 2:
        raise ValueError(f"embeddings must be identical (N, D), got {p.shape} and {r.shape}")
    p = p / (np.linalg.norm(p, axis=1, keepdims=True) + eps)
    r = r / (np.linalg.norm(r, axis=1, keepdims=True) + eps)
    return float((p * r).sum(axis=1).mean())
