"""Tests for the evaluation harness cores (paper Tables 1, 3, 5).

Verifies the frozen-target direction/beats/meets logic, the Frechet distance
(FID/FVD core), CSIM, and MCD against closed-form values. No models/data/GPU.

Run directly:  python3 tests/test_eval.py
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.dirname(HERE)
sys.path.insert(0, CODE_ROOT)

import numpy as np  # noqa: E402

from chat.eval import (  # noqa: E402
    IHG_TARGETS,
    beats_paper,
    frechet_distance,
    frechet_from_features,
    identity_similarity,
    mcd,
    meets_paper,
    mel_cepstral_distortion,
    report,
)


def test_targets_direction():
    # FID is lower-better: below target beats, above does not.
    assert beats_paper("FID", 17.0) and not beats_paper("FID", 18.0)
    # LSE-C is higher-better.
    assert beats_paper("LSE-C", 7.0) and not beats_paper("LSE-C", 6.0)
    # equal is not strictly better, but meets (never worse).
    assert not beats_paper("CSIM", 0.85)
    assert meets_paper("CSIM", 0.85) and meets_paper("FID", 17.33)


def test_meets_tolerance():
    assert meets_paper("FID", 17.4, tol=0.1)      # 17.4 <= 17.33 + 0.1
    assert not meets_paper("FID", 17.5, tol=0.1)
    assert meets_paper("LSE-C", 6.85, tol=0.1)    # 6.85 >= 6.89 - 0.1


def test_unknown_metric_raises():
    try:
        beats_paper("NOPE", 1.0)
        assert False, "expected KeyError"
    except KeyError:
        pass


def test_report_known_only():
    res = {r.metric: r for r in report({"FID": 17.0, "CSIM": 0.90, "bogus": 1.0})}
    assert set(res) == {"FID", "CSIM"}
    assert res["FID"].beats and res["FID"].meets
    assert res["CSIM"].direction == "up"


def test_ihg_targets():
    t = IHG_TARGETS["PerFRDiff"]
    assert beats_paper("FRCorr", 41.0, t) and not beats_paper("FRCorr", 40.0, t)
    assert beats_paper("FRDist", 89.0, t)  # lower-better


def test_frechet_identical_is_zero():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((500, 8))
    assert abs(frechet_from_features(x, x)) < 1e-4


def test_frechet_mean_shift_closed_form():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((800, 8))
    y = x + 3.0  # same covariance, mean shifted by 3 in each of 8 dims
    # FID = ||dmu||^2 + 0 = 9 * 8 = 72
    assert abs(frechet_from_features(x, y) - 72.0) < 0.5


def test_frechet_shape_validation():
    for bad in [(np.zeros((5, 4)), np.zeros((5, 3))), (np.zeros((1, 4)), np.zeros((2, 4)))]:
        try:
            frechet_from_features(*bad)
            assert False, "expected ValueError"
        except ValueError:
            pass


def test_frechet_distance_direct():
    mu = np.zeros(3)
    cov = np.eye(3)
    assert abs(frechet_distance(mu, cov, mu, cov)) < 1e-9
    assert abs(frechet_distance(mu, cov, mu + 2.0, cov) - 12.0) < 1e-9  # 4*3 dims


def test_csim_identical_and_orthogonal():
    a = np.array([[1.0, 0.0], [0.0, 2.0]])
    assert abs(identity_similarity(a, a) - 1.0) < 1e-6
    b = np.array([[0.0, 1.0], [1.0, 0.0]])  # each row orthogonal to a's row
    assert abs(identity_similarity(a, b)) < 1e-6


def test_csim_shape_validation():
    try:
        identity_similarity(np.zeros((2, 3)), np.zeros((2, 4)))
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_mcd_identical_is_zero():
    c = np.random.default_rng(2).standard_normal((10, 13))
    assert mel_cepstral_distortion(c, c) < 1e-9


def test_mcd_closed_form():
    ref = np.array([[0.0, 1.0, 0.0]])
    gen = np.array([[0.0, 0.0, 0.0]])
    # excludes c0; diff on dims {1,2} = [1,0] -> ||.||=1 -> const*1
    assert abs(mel_cepstral_distortion(ref, gen) - mcd._MCD_CONST) < 1e-9


def test_mcd_shape_validation():
    try:
        mel_cepstral_distortion(np.zeros((4, 3)), np.zeros((5, 3)))
        assert False, "expected ValueError"
    except ValueError:
        pass


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL {t.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return failures


if __name__ == "__main__":
    sys.exit(1 if _run() else 0)
