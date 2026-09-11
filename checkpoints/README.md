# Released IFBR weights

The two weights the Interactive Facial Behaviour Refinement stage needs. Stage 5
(`scripts/x1_refine.py`) takes `--rfbg-ckpt` as a required argument, so without
`rfbg_qknorm_ft_030500.pt` the refinement block cannot run at all and the pipeline
reduces to the talking-face backbone on its own.

| file | size | module | training |
|---|---|---|---|
| `rfbg_qknorm_ft_030500.pt` | 13 MB | RFBG (`chat/ifbg/ifbr/rfbg.py`) | HDTF cold-start pre-train, then dyadic fine-tune on NoXI pairs with QK-normalisation enabled |
| `sfbg_pre_020000.pt` | 5.3 MB | SFBG (`chat/ifbg/ifbr/sfbg.py`) | 20k-step HDTF cold-start pre-train on contiguous [prev, target, next] triples |

Integrity, and what each file contains when loaded:

```
13801971 bytes  sha256 0d328cc8252abf47150530ae5d96636c3eacc97fb0469eb871803585948e4420
                rfbg_qknorm_ft_030500.pt   1,707,836 params  qk_norm=True  L=3  T=50
 5542293 bytes  sha256 902da86367e652e4eb8ea0325ddb2ac30f66d0cc24dd2e58f505b03ff39261ac
                sfbg_pre_020000.pt         1,375,492 params                     T=50
```

Both files carry weights only. The AdamW state written during training was stripped
before release, which is what makes them 13 MB and 5.3 MB rather than 25 MB and 17 MB.
`train_rfbg.py --resume` and `train_sfbg.py --resume` detect the absence and continue
from the weights with a fresh optimiser, restarting the step counter.

`T=50` is the paper's denoising-step count (Sec. 5.1). Check a download with
`shasum -a 256 checkpoints/*.pt`.

## What these are, precisely

Both are **reproduction artefacts**, not the paper's trained models. They were trained
at a reduced parameter count, primarily on a single 16 GB GPU, and on substituted
corpora: the paper pre-trains RFBG on REACT 2024, whose annotation licence is not
available here, so NoXI stands in for the dyadic stage. Numbers produced with these
weights are reported as reproduction results throughout, separately from the paper's.
[`DOCS.md`](../DOCS.md) records every such difference and tabulates the
paper's technical details against their implementation status.

Each file stores its own `RFBGConfig` / `SFBGConfig` alongside the weights, so the
architecture is read from the checkpoint rather than assumed by the loader. The RFBG
file also carries an EMA copy of the weights, which `--weights auto` prefers; pass
`--weights raw` for the non-averaged weights. The SFBG file has no EMA copy, so
`--weights auto` falls back to its raw weights.

## Loading them

```python
import sys; sys.path.insert(0, "scripts")
from x1_refine import load_rfbg, load_sfbg

rfbg = load_rfbg("checkpoints/rfbg_qknorm_ft_030500.pt", "cuda")
sfbg = load_sfbg("checkpoints/sfbg_pre_020000.pt", "cuda")
```

`scripts/x1_gen_core.sh` picks both up from this directory by default. Override with
the `RFBG_CKPT` and `SFBG_CKPT` environment variables.

## Two things worth knowing before you use them

**SFBG needs `--sfbg-strength` below 1.0, and 0.3 is now the default.** At 1.0 it
generates each silent segment from pure noise through all 50 steps, which this 20k-step
checkpoint is not strong enough to do: a 10-side pilot measured +128 FID and the segments
decode to colour noise rather than faces. Below 1.0 it renoises the RFBG output to a
mid-chain timestep and denoises from there, which is how RFBG itself is run. Every
reported SFBG number used 0.3, which is what `scripts/x1_refine.py` now defaults to.

**The audio and emotion conditioning pathway in RFBG is untrained.** `RFBGDenoiser`
carries `audio_proj`, `emo_proj` and a per-block `aux_attn` for the partner's audio and
emotion (paper Sec. 4.3), but no training stage supplied them, so those parameters are
at their initialisation in this checkpoint and `scripts/x1_refine.py` passes `None` for
both. The refinement conditions on the partner's visual window only. This is recorded
in [`DOCS.md`](../DOCS.md).
