"""Frechet distance for FID (image features) and FVD (video features).

FID/FVD = ||mu_r - mu_g||^2 + Tr(Sigma_r + Sigma_g - 2 (Sigma_r Sigma_g)^{1/2}).
The matrix-sqrt trace is computed from the eigenvalues of the covariance product
(numpy-only; no scipy). Callers supply the extracted features: Inception-pool3 for
FID, an I3D/video backbone for FVD (both loaded on a GPU).
"""

from __future__ import annotations

import numpy as np


def frechet_distance(mu1, cov1, mu2, cov2) -> float:
    mu1, mu2 = np.asarray(mu1, float), np.asarray(mu2, float)
    cov1, cov2 = np.asarray(cov1, float), np.asarray(cov2, float)
    diff = mu1 - mu2
    # Tr((cov1 cov2)^{1/2}) = sum sqrt(eigvals(cov1 cov2)); the product of two PSD
    # matrices has real non-negative eigenvalues (clip guards numerical noise).
    eig = np.linalg.eigvals(cov1 @ cov2)
    covmean_trace = np.sqrt(np.clip(eig.real, 0.0, None)).sum()
    return float(diff @ diff + np.trace(cov1) + np.trace(cov2) - 2.0 * covmean_trace)


def frechet_from_features(feats_real, feats_gen) -> float:
    """feats_*: (N, D) arrays of extracted features."""
    a = np.asarray(feats_real, float)
    b = np.asarray(feats_gen, float)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError(f"features must be (N, D) with matching D, got {a.shape} and {b.shape}")
    if a.shape[0] < 2 or b.shape[0] < 2:
        raise ValueError("need >= 2 samples per set to estimate a covariance")
    mu_a, cov_a = a.mean(0), np.cov(a, rowvar=False)
    mu_b, cov_b = b.mean(0), np.cov(b, rowvar=False)
    return frechet_distance(mu_a, cov_a, mu_b, cov_b)
