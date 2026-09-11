"""Test the FID Inception feature extractor end-to-end (paper Table 1, FID).

Requires torch + pytorch-fid; skips
cleanly where they are absent. Uses synthetic images: identical sets give FID ~ 0,
a perturbed set gives a strictly larger FID.

Run:  python tests/test_eval_extractors.py
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.dirname(HERE)
sys.path.insert(0, CODE_ROOT)

try:
    import torch  # noqa: F401
    from pytorch_fid.inception import InceptionV3  # noqa: F401
    _HAVE = True
except Exception:  # noqa: BLE001
    _HAVE = False

import numpy as np  # noqa: E402


def test_extractor_and_fid():
    if not _HAVE:
        print("SKIP test_extractor_and_fid: torch/pytorch-fid not installed")
        return
    import torch

    from chat.eval.extractors import InceptionExtractor, fid

    torch.manual_seed(0)
    real = torch.rand(16, 3, 64, 64)
    ext = InceptionExtractor(device="cpu")

    feats = ext.features(real)
    assert feats.shape == (16, 2048), feats.shape

    fid_same = fid(real, real, extractor=ext)
    perturbed = (real + 0.5 * torch.rand_like(real)).clamp(0, 1)
    fid_diff = fid(real, perturbed, extractor=ext)

    assert fid_same < 1.0, f"FID(x,x) should be ~0, got {fid_same}"
    assert fid_diff > fid_same, f"perturbed FID {fid_diff} not > identical FID {fid_same}"
    print(f"PASS test_extractor_and_fid (FID same={fid_same:.4f}, diff={fid_diff:.4f})")


def test_features_shape_validation():
    if not _HAVE:
        print("SKIP test_features_shape_validation: torch/pytorch-fid not installed")
        return
    from chat.eval.extractors import InceptionExtractor

    ext = InceptionExtractor(device="cpu")
    try:
        ext.features(np.zeros((4, 1, 64, 64)))  # 1 channel, not RGB
        assert False, "expected ValueError"
    except ValueError:
        print("PASS test_features_shape_validation")


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failures = 0
    for t in tests:
        try:
            t()
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL {t.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} ok")
    return failures


if __name__ == "__main__":
    sys.exit(1 if _run() else 0)
