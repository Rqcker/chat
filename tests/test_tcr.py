"""Unit tests for Temporal Continuity Refinement (paper Sec. 4.3).

Run directly:  python3 tests/test_tcr.py
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.dirname(HERE)
sys.path.insert(0, CODE_ROOT)

import numpy as np  # noqa: E402

from chat.ifbg.ifbr import apply_tcr, blend_boundary, gaussian_alpha  # noqa: E402


def test_gaussian_alpha_bounds_and_monotonic():
    a = gaussian_alpha(10)
    assert a.shape == (10,)
    assert np.all(a > 0) and np.all(a <= 0.5)
    assert np.all(np.diff(a) < 0)  # strictly decreasing away from the boundary
    # half-Gaussian: alpha(1) = 0.5 at the boundary
    assert np.isclose(a[0], 0.5)


def test_gaussian_alpha_validates():
    for bad in (0, -1):
        try:
            gaussian_alpha(bad)
            assert False, "expected ValueError"
        except ValueError:
            pass


def test_blend_identity_when_overlap_equal():
    # If the prev tail equals the cur head, the symmetric blend is a no-op there.
    prev = np.zeros((5, 2))
    cur = np.zeros((5, 2))
    prev[:] = np.arange(10).reshape(5, 2)
    cur[:] = np.arange(100, 110).reshape(5, 2)
    # boundary cross-fade pairs the cur head with the REVERSED prev tail
    cur[0:2] = prev[3:5][::-1]
    new_prev, new_cur = blend_boundary(prev, cur, window=2)
    assert np.allclose(new_prev[3:5], prev[3:5])  # overlap unchanged
    assert np.allclose(new_cur[0:2], cur[0:2])
    assert np.allclose(new_prev[:3], prev[:3])    # non-overlap untouched
    assert np.allclose(new_cur[2:], cur[2:])


def test_blend_sum_preserved():
    rng = np.random.default_rng(0)
    prev = rng.standard_normal((8, 3))
    cur = rng.standard_normal((8, 3))
    new_prev, new_cur = blend_boundary(prev, cur, window=4)
    # per-pair conservation: prev frame (F-tau+1) pairs with cur frame tau, so
    # compare the reversed prev tail against the cur head.
    assert np.allclose(new_prev[4:][::-1] + new_cur[:4], prev[4:][::-1] + cur[:4])


def test_blend_nonoverlap_untouched_and_shape():
    rng = np.random.default_rng(1)
    prev = rng.standard_normal((6, 2, 2))
    cur = rng.standard_normal((6, 2, 2))
    new_prev, new_cur = blend_boundary(prev, cur, window=2)
    assert new_prev.shape == prev.shape and new_cur.shape == cur.shape
    assert np.allclose(new_prev[:4], prev[:4])  # frames before the tail window
    assert np.allclose(new_cur[2:], cur[2:])    # frames after the head window


def test_blend_window_too_large_raises():
    prev = np.zeros((5, 2))
    cur = np.zeros((5, 2))
    try:
        blend_boundary(prev, cur, window=10)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_blend_does_not_mutate_inputs():
    prev = np.ones((4, 1))
    cur = np.zeros((4, 1))
    prev_copy = prev.copy()
    cur_copy = cur.copy()
    blend_boundary(prev, cur, window=2)
    assert np.array_equal(prev, prev_copy) and np.array_equal(cur, cur_copy)


def test_apply_tcr_three_segments():
    rng = np.random.default_rng(2)
    segs = [rng.standard_normal((10,)) for _ in range(3)]
    orig = [s.copy() for s in segs]
    out = apply_tcr(segs, window=2)
    assert len(out) == 3
    # first segment: only its trailing window (last 2) is blended
    assert np.allclose(out[0][:8], orig[0][:8])
    assert not np.allclose(out[0][8:], orig[0][8:])
    # last segment: only its leading window (first 2) is blended
    assert np.allclose(out[2][2:], orig[2][2:])
    assert not np.allclose(out[2][:2], orig[2][:2])
    # middle segment: both ends blended, interior untouched
    assert np.allclose(out[1][2:8], orig[1][2:8])
    # inputs not mutated
    assert np.allclose(segs[1], orig[1])


def test_apply_tcr_single_segment_is_copy():
    seg = np.arange(5.0)
    out = apply_tcr([seg], window=2)
    assert len(out) == 1
    assert np.array_equal(out[0], seg)
    assert out[0] is not seg


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
