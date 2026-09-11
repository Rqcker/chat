"""Shape-level smoke tests for IAR TFM-C/TFM-E encoders (paper Sec. 4.2).

Verifies construction, content/emotion encoding, the element-wise-sum latent z,
padding-masked pooling, backward, and a 1-token edge case. No weights/data/GPU.

Run directly:  python3 tests/test_iar_encoders.py
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.dirname(HERE)
sys.path.insert(0, CODE_ROOT)

import torch  # noqa: E402

from chat.dadg.iar.encoders import (  # noqa: E402
    IARLatentEncoder,
    TFMConfig,
    TFMContent,
    TFMEmotion,
)

torch.manual_seed(0)

CFG = TFMConfig(
    vocab_size=100, num_emotions=8, embed_dim=32, num_heads=4, num_layers=2,
    ff_dim=64, prosody_dim=4,
)
B, L = 2, 10


def _inputs(length=L):
    return dict(
        content_ids=torch.randint(0, CFG.vocab_size, (B, length)),
        category_ids=torch.randint(0, CFG.num_emotions, (B,)),
        prosody=torch.randn(B, CFG.prosody_dim),
    )


def test_content_and_emotion_shapes():
    d = _inputs()
    assert TFMContent(CFG)(d["content_ids"]).shape == (B, CFG.embed_dim)
    assert TFMEmotion(CFG)(d["category_ids"], d["prosody"]).shape == (B, CFG.embed_dim)


def test_latent_sum_shape():
    model = IARLatentEncoder(CFG)
    d = _inputs()
    z = model(d["content_ids"], d["category_ids"], d["prosody"])
    assert z.shape == (B, CFG.embed_dim)
    # z is the element-wise sum of the two encoders' outputs
    expected = model.tfm_c(d["content_ids"]) + model.tfm_e(d["category_ids"], d["prosody"])
    assert torch.allclose(z, expected)


def test_padding_mask_excludes_padded_tokens():
    tfm_c = TFMContent(CFG).eval()
    d = _inputs()
    mask = torch.zeros(B, L, dtype=torch.bool)
    mask[:, L - 3:] = True  # last 3 tokens are padding
    z1 = tfm_c(d["content_ids"], mask)
    ids2 = d["content_ids"].clone()
    ids2[:, L - 3:] = (ids2[:, L - 3:] + 5) % CFG.vocab_size  # change ONLY padded tokens
    z2 = tfm_c(ids2, mask)
    assert torch.allclose(z1, z2, atol=1e-5)  # padded content must not affect the pooled latent


def test_all_padded_row_is_finite():
    tfm_c = TFMContent(CFG).eval()
    d = _inputs()
    mask = torch.zeros(B, L, dtype=torch.bool)
    mask[0, :] = True  # an entire row is padding
    assert torch.isfinite(tfm_c(d["content_ids"], mask)).all()


def test_batch_of_one():
    model = IARLatentEncoder(CFG)
    z = model(
        torch.randint(0, CFG.vocab_size, (1, L)),
        torch.randint(0, CFG.num_emotions, (1,)),
        torch.randn(1, CFG.prosody_dim),
    )
    assert z.shape == (1, CFG.embed_dim)


def test_out_of_range_category_raises():
    model = IARLatentEncoder(CFG)
    d = _inputs()
    bad = d["category_ids"].clone()
    bad[0] = CFG.num_emotions  # outside [0, num_emotions)
    try:
        model(d["content_ids"], bad, d["prosody"])
        assert False, "expected an index error for out-of-range category"
    except (IndexError, RuntimeError):
        pass


def test_batch_mismatch_raises():
    model = IARLatentEncoder(CFG)
    d = _inputs()
    try:
        model(d["content_ids"], d["category_ids"][:1], d["prosody"])
        assert False, "expected an assertion for mismatched batch sizes"
    except AssertionError:
        pass


def test_backward_all_params():
    model = IARLatentEncoder(CFG)
    d = _inputs()
    z = model(d["content_ids"], d["category_ids"], d["prosody"])
    z.sum().backward()
    n_train = sum(1 for p in model.parameters() if p.requires_grad)
    n_grad = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None)
    assert n_grad == n_train, f"only {n_grad}/{n_train} params got grads"


def test_single_token_content():
    model = IARLatentEncoder(CFG)
    d = _inputs(length=1)
    z = model(d["content_ids"], d["category_ids"], d["prosody"])
    assert z.shape == (B, CFG.embed_dim)


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
