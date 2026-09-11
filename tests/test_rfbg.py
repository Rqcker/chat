"""Shape-level smoke tests for RFBG (paper Sec. 4.3).

Verifies construction, multi-scale conditioning, the coarse-to-fine scale schedule,
the denoiser forward (concatenated masked multi-scale memory + dedicated audio/
emotion cross-attention, refining an initial segment), one training step, sampling
(full + opt-in SDEdit), and diffusion-schedule sanity. No weights/data/GPU.

Run directly:  python3 tests/test_rfbg.py
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.dirname(HERE)
sys.path.insert(0, CODE_ROOT)

import torch  # noqa: E402

from chat.ifbg.ifbr.rfbg import RFBG, RFBGConfig  # noqa: E402

torch.manual_seed(0)

CFG = RFBGConfig(
    latent_channels=4, embed_dim=32, cond_dim=16, num_heads=4, num_layers=2,
    num_scales=3, num_timesteps=8, mlp_ratio=2, audio_dim=78, emotion_dim=32,
)
B, F_, FP, C, H, W = 2, 3, 5, 4, 8, 8  # FP != F_: the 3-turn partner window is longer


def _inputs():
    return dict(
        x0=torch.randn(B, F_, C, H, W),        # refined target tilde_v
        init=torch.randn(B, F_, C, H, W),      # initial segment v being refined
        partner=torch.randn(B, FP, C, H, W),   # v_bar^j(k-1:k+1), F_p independent of F
        identity=torch.randn(B, C, H, W),      # ID^i
        audio=torch.randn(B, 5, 78),           # a^j(k-1:k+1)
        emotion=torch.randn(B, 3, 32),         # e^j(k-1:k+1)
    )


def test_conditioner_shapes():
    model = RFBG(CFG)
    d = _inputs()
    conds = model.conditioner(d["partner"], d["identity"])
    assert len(conds) == CFG.num_scales
    for l, c in enumerate(conds):  # each scale keeps its NATIVE resolution
        assert c.shape == (B, FP, C, H // 2 ** l, W // 2 ** l)


def test_scale_indices_formula():
    model = RFBG(CFG)
    L, T = CFG.num_scales, CFG.num_timesteps
    assert model.scale_indices_for_t(0) == list(range(L))    # t=0: all scales
    assert model.scale_indices_for_t(T - 1) == [L - 1]       # t=T-1: coarsest only
    assert model.scale_indices_for_t(3) == [1, 2]            # ceil: l(3)=max(1,ceil(9/8)=2)=2
    sizes = [len(model.scale_indices_for_t(t)) for t in range(T)]
    assert sizes == sorted(sizes, reverse=True)              # shrinks as t grows


def test_scale_table_matches_indices():
    model = RFBG(CFG)
    for t in range(CFG.num_timesteps):
        selected = [i for i, on in enumerate(model.scale_table[t].tolist()) if on]
        assert selected == model.scale_indices_for_t(t)


def test_denoiser_forward_shape():
    model = RFBG(CFG)
    d = _inputs()
    conds = model.conditioner(d["partner"], d["identity"])
    t = torch.randint(0, CFG.num_timesteps, (B,))
    sm = model.scale_table[t]
    out = model.denoiser(d["x0"], t, conds, sm, d["init"], audio=d["audio"], emotion=d["emotion"])
    assert out.shape == (B, F_, C, H, W)
    out_no_aux = model.denoiser(d["x0"], t, conds, sm, d["init"])  # aux optional
    assert out_no_aux.shape == (B, F_, C, H, W)


def test_all_params_get_grad_when_all_scales_active():
    model = RFBG(CFG)
    d = _inputs()
    # t=0 selects every scale (mask all True), so every per-scale CA block, the
    # scale embedding, and the dedicated aux cross-attention are exercised.
    conds = model.conditioner(d["partner"], d["identity"])
    t = torch.zeros(B, dtype=torch.long)
    noise = torch.randn_like(d["x0"])
    xt = model.q_sample(d["x0"], t, noise)
    pred = model.denoiser(xt, t, conds, model.scale_table[t], d["init"], audio=d["audio"], emotion=d["emotion"])
    torch.nn.functional.mse_loss(pred, noise).backward()
    n_train = sum(1 for p in model.parameters() if p.requires_grad)
    n_grad = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None)
    assert n_grad == n_train, f"only {n_grad}/{n_train} params got grads"


def test_training_loss_api_backward():
    model = RFBG(CFG)
    d = _inputs()
    loss = model.training_loss(d["x0"], d["init"], d["partner"], d["audio"], d["emotion"], d["identity"])
    assert loss.dim() == 0 and torch.isfinite(loss)
    loss.backward()
    assert sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None) > 0


def test_sample_shape_full():
    model = RFBG(CFG)
    d = _inputs()
    out = model.sample(d["init"], d["partner"], d["audio"], d["emotion"], d["identity"])  # default strength=1.0
    assert out.shape == (B, F_, C, H, W) and torch.isfinite(out).all()


def test_sdedit_strength_inherits_init():
    model = RFBG(CFG)
    d = _inputs()
    torch.manual_seed(1)
    out_low = model.sample(d["init"], d["partner"], d["audio"], d["emotion"], d["identity"], strength=0.1)
    out_full = model.sample(d["init"], d["partner"], d["audio"], d["emotion"], d["identity"], strength=1.0)
    # low strength renoises little, so it stays closer to the initial segment
    assert (out_low - d["init"]).abs().mean() < (out_full - d["init"]).abs().mean()


def test_masked_scales_do_not_affect_output():
    model = RFBG(CFG).eval()
    d = _inputs()
    conds = model.conditioner(d["partner"], d["identity"])
    t = torch.full((B,), CFG.num_timesteps - 1, dtype=torch.long)  # coarsest scale only
    sm = model.scale_table[t]
    out1 = model.denoiser(d["x0"], t, conds, sm, d["init"], audio=d["audio"], emotion=d["emotion"])
    conds2 = [c.clone() for c in conds]
    conds2[0] = conds2[0] + 5.0  # perturb the FINEST scale, which is masked out at t=T-1
    out2 = model.denoiser(d["x0"], t, conds2, sm, d["init"], audio=d["audio"], emotion=d["emotion"])
    assert torch.allclose(out1, out2, atol=1e-5)  # key-padding mask must exclude unselected scales


def test_batch_independence_per_sample_t():
    """Pin batch-axis mask/memory pairing with PER-SAMPLE timesteps (a repeat vs
    repeat_interleave regression passes uniform-t tests but fails this)."""
    model = RFBG(CFG).eval()
    d = _inputs()
    conds = model.conditioner(d["partner"], d["identity"])
    t = torch.tensor([0, CFG.num_timesteps - 1])  # sample 0: all scales; sample 1: coarsest only
    sm = model.scale_table[t]
    out1 = model.denoiser(d["x0"], t, conds, sm, d["init"], audio=d["audio"], emotion=d["emotion"])
    # (i) perturb the FINEST scale: masked for sample 1 only -> sample 1 unchanged, sample 0 changed
    conds2 = [c.clone() for c in conds]
    conds2[0] = conds2[0] + 5.0
    out2 = model.denoiser(d["x0"], t, conds2, sm, d["init"], audio=d["audio"], emotion=d["emotion"])
    assert torch.allclose(out1[1], out2[1], atol=1e-5), "masked scale leaked into sample 1"
    assert not torch.allclose(out1[0], out2[0], atol=1e-5), "selected scale had no effect on sample 0"
    # (ii) perturb EVERYTHING of sample 1 -> sample 0 must be unchanged
    def bump(x):
        y = x.clone()
        y[1] = y[1] + 3.0
        return y
    conds3 = [bump(c) for c in conds]
    out3 = model.denoiser(
        bump(d["x0"]), t, conds3, sm, bump(d["init"]),
        audio=bump(d["audio"]), emotion=bump(d["emotion"]),
    )
    assert torch.allclose(out1[0], out3[0], atol=1e-5), "sample 1 inputs leaked into sample 0"


def test_q_sample_matches_schedule():
    model = RFBG(CFG)
    x0 = torch.randn(B, F_, C, H, W)
    noise = torch.randn_like(x0)
    t = torch.tensor([0, CFG.num_timesteps - 1])
    xt = model.q_sample(x0, t, noise)
    ac = model.alphas_cumprod[t].view(-1, 1, 1, 1, 1)
    assert torch.allclose(xt, ac.sqrt() * x0 + (1.0 - ac).sqrt() * noise)
    assert model.alphas_cumprod[0] > model.alphas_cumprod[-1]


def test_num_scales_one():
    cfg1 = RFBGConfig(
        latent_channels=4, embed_dim=16, cond_dim=8, num_heads=4, num_layers=1,
        num_scales=1, num_timesteps=4, mlp_ratio=2, audio_dim=78, emotion_dim=32,
    )
    model = RFBG(cfg1)
    d = _inputs()
    assert len(model.conditioner(d["partner"], d["identity"])) == 1
    model.training_loss(d["x0"], d["init"], d["partner"], d["audio"], d["emotion"], d["identity"]).backward()
    out = model.sample(d["init"], d["partner"], d["audio"], d["emotion"], d["identity"])
    assert out.shape == (B, F_, C, H, W)


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
