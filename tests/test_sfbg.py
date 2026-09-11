"""Shape-level smoke tests for SFBG (paper Sec. 4.3, SilentDiff).

Verifies construction, neighbour+identity conditioning, the denoiser forward,
one training step (loss + backward), sampling, and a diffusion-schedule check.
No weights/data/GPU; trained-output quality is out of scope.

Run directly:  python3 tests/test_sfbg.py
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.dirname(HERE)
sys.path.insert(0, CODE_ROOT)

import torch  # noqa: E402

from chat.ifbg.ifbr.sfbg import SFBG, SFBGConfig  # noqa: E402

torch.manual_seed(0)

CFG = SFBGConfig(
    latent_channels=4, embed_dim=32, num_heads=4, num_layers=2, num_timesteps=8,
    mlp_ratio=2,
)
B, C, H, W = 2, 4, 8, 8
FP, FN, FS = 2, 2, 3  # prev / next talking segments, silent segment length


def _inputs():
    return dict(
        silent_gt=torch.randn(B, FS, C, H, W),
        prev=torch.randn(B, FP, C, H, W),
        next_seg=torch.randn(B, FN, C, H, W),
        identity=torch.randn(B, C, H, W),
    )


def test_frame_order_is_visible():
    """Neighbour-segment tokens carry a frame-order code: reversing the prev
    segment's frames must change the output (attention alone is permutation-
    invariant over keys, which would make motion direction invisible)."""
    model = SFBG(CFG).eval()
    d = _inputs()
    t = torch.zeros(B, dtype=torch.long)
    cond_fwd = model.denoiser.build_condition(d["prev"], d["next_seg"], d["identity"])
    cond_rev = model.denoiser.build_condition(d["prev"].flip(1), d["next_seg"], d["identity"])
    out_fwd = model.denoiser(d["silent_gt"], t, cond_fwd)
    out_rev = model.denoiser(d["silent_gt"], t, cond_rev)
    assert not torch.allclose(out_fwd, out_rev, atol=1e-5), "frame order invisible to the model"


def test_batch_independence():
    """Perturbing sample 1's conditioning must not change sample 0's output."""
    model = SFBG(CFG).eval()
    d = _inputs()
    t = torch.tensor([0, CFG.num_timesteps - 1])
    cond1 = model.denoiser.build_condition(d["prev"], d["next_seg"], d["identity"])
    out1 = model.denoiser(d["silent_gt"], t, cond1)
    prev2, next2, iden2 = d["prev"].clone(), d["next_seg"].clone(), d["identity"].clone()
    prev2[1] += 3.0
    next2[1] += 3.0
    iden2[1] += 3.0
    cond2 = model.denoiser.build_condition(prev2, next2, iden2)
    out2 = model.denoiser(d["silent_gt"], t, cond2)
    assert torch.allclose(out1[0], out2[0], atol=1e-5), "sample 1 cond leaked into sample 0"
    assert not torch.allclose(out1[1], out2[1], atol=1e-5), "sample 1 cond had no effect"


def test_build_condition_shape():
    model = SFBG(CFG)
    d = _inputs()
    cond = model.denoiser.build_condition(d["prev"], d["next_seg"], d["identity"])
    p = CFG.ctx_pool  # neighbour segments are spatially pooled; identity is full-res
    expected_tokens = FP * (H // p) * (W // p) + FN * (H // p) * (W // p) + H * W
    assert cond.shape == (B, expected_tokens, CFG.embed_dim)


def test_denoiser_forward_shape():
    model = SFBG(CFG)
    d = _inputs()
    cond = model.denoiser.build_condition(d["prev"], d["next_seg"], d["identity"])
    t = torch.randint(0, CFG.num_timesteps, (B,))
    out = model.denoiser(d["silent_gt"], t, cond)
    assert out.shape == (B, FS, C, H, W)


def test_training_loss_backward_all_params():
    model = SFBG(CFG)
    d = _inputs()
    loss = model.training_loss(d["silent_gt"], d["prev"], d["next_seg"], d["identity"])
    assert loss.dim() == 0 and torch.isfinite(loss)
    loss.backward()
    n_train = sum(1 for p in model.parameters() if p.requires_grad)
    n_grad = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None)
    assert n_grad == n_train, f"only {n_grad}/{n_train} params got grads"


def test_sample_shape():
    model = SFBG(CFG)
    d = _inputs()
    out = model.sample(d["prev"], d["next_seg"], d["identity"], num_frames=FS)
    assert out.shape == (B, FS, C, H, W)
    assert torch.isfinite(out).all()


def test_q_sample_matches_schedule():
    model = SFBG(CFG)
    x0 = torch.randn(B, FS, C, H, W)
    noise = torch.randn_like(x0)
    t = torch.tensor([0, CFG.num_timesteps - 1])
    xt = model.q_sample(x0, t, noise)
    ac = model.alphas_cumprod[t].view(-1, 1, 1, 1, 1)
    assert torch.allclose(xt, ac.sqrt() * x0 + (1.0 - ac).sqrt() * noise)
    assert model.alphas_cumprod[0] > model.alphas_cumprod[-1]


def test_role_embeddings_differentiate_prev_next():
    model = SFBG(CFG)
    d = _inputs()
    cond_ab = model.denoiser.build_condition(d["prev"], d["next_seg"], d["identity"])
    cond_ba = model.denoiser.build_condition(d["next_seg"], d["prev"], d["identity"])
    # swapping prev <-> next must change the conditioning (distinct role embeddings)
    assert not torch.allclose(cond_ab, cond_ba)


def test_single_frame_silent_segment():
    model = SFBG(CFG)
    d = _inputs()
    out = model.sample(d["prev"], d["next_seg"], d["identity"], num_frames=1)
    assert out.shape == (B, 1, C, H, W)


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
