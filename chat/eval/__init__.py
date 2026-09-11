"""Evaluation harness for CHAT (paper Tables 1, 3, 5).

Pure-math metric cores + the frozen paper targets. The heavy feature extractors
(Inception for FID, I3D for FVD, SyncNet for LSE-C/D, ArcFace for CSIM, an emotion
classifier for Emo-Acc, ViSQOL, and a pixel->25d-emotion extractor for FRCorr/
FRDiv/FRDist) are supplied by the caller on a GPU: this package computes the
metrics FROM the extracted features/embeddings/coefficients and compares them to
the paper's reported values. Frechet metrics are protocol-dependent, so a
comparison is only meaningful when the reference set, framing and frame sampling
match.
"""

from .csim import identity_similarity
from .frechet import frechet_distance, frechet_from_features
from .mcd import mel_cepstral_distortion
from .targets import (
    IHG_TARGETS,
    TARGETS,
    MetricResult,
    beats_paper,
    meets_paper,
    report,
)

__all__ = [
    "TARGETS",
    "IHG_TARGETS",
    "MetricResult",
    "beats_paper",
    "meets_paper",
    "report",
    "frechet_distance",
    "frechet_from_features",
    "identity_similarity",
    "mel_cepstral_distortion",
]
