"""Paper-reported metric values and comparison helpers (paper Tables 1, 3, 5).

Values are the Full CHAT column. Directions are "up" where higher is better and
"down" where lower is better. FRCorr, FRDiv and FRDist are reported x1e-2 in the
paper and are kept here as printed.

Note that FID and FVD are protocol-dependent. Numbers are only comparable when the
reference set, the framing and the frame sampling all match, so these values are
not a like-for-like target for a differently configured evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

# metric -> (Full CHAT target, direction)
TARGETS: Dict[str, Tuple[float, str]] = {
    "FID": (17.33, "down"),
    "FVD": (365.03, "down"),
    "LSE-C": (6.89, "up"),
    "LSE-D": (8.07, "down"),
    "FRCorr": (50.02, "up"),
    "FRDiv": (15.34, "up"),
    "CSIM": (0.85, "up"),
    "LPIPS": (0.38, "down"),
    "MCD": (4.23, "down"),
    "Emo-Acc": (78.4, "up"),
    "ViSQOL": (3.98, "up"),
}

# Downstream IHG (Table 5), the "w/ CHAT-AVD-50k" targets per model.
IHG_TARGETS: Dict[str, Dict[str, Tuple[float, str]]] = {
    "PerFRDiff": {"FRCorr": (40.11, "up"), "FRDist": (89.45, "down")},
    "ReactDiff": {"FRCorr": (26.12, "up"), "FRDist": (83.87, "down")},
}


def _target(metric: str, targets) -> Tuple[float, str]:
    if metric not in targets:
        raise KeyError(f"unknown metric {metric!r}; known: {sorted(targets)}")
    return targets[metric]


def beats_paper(metric: str, value: float, targets=TARGETS) -> bool:
    """Strictly better than the paper (up: value > target; down: value < target)."""
    target, d = _target(metric, targets)
    return value > target if d == "up" else value < target


def meets_paper(metric: str, value: float, tol: float = 0.0, targets=TARGETS) -> bool:
    """At least as good as the paper within tolerance (never worse)."""
    target, d = _target(metric, targets)
    return value >= target - tol if d == "up" else value <= target + tol


@dataclass
class MetricResult:
    metric: str
    value: float
    target: float
    direction: str
    beats: bool
    meets: bool


def report(values: Dict[str, float], tol: float = 0.0, targets=TARGETS) -> List[MetricResult]:
    """Compare measured {metric: value} against the paper targets (known metrics only)."""
    out: List[MetricResult] = []
    for metric, value in values.items():
        if metric not in targets:
            continue
        target, d = targets[metric]
        out.append(
            MetricResult(
                metric=metric,
                value=float(value),
                target=target,
                direction=d,
                beats=beats_paper(metric, value, targets),
                meets=meets_paper(metric, value, tol, targets),
            )
        )
    return out
