"""RFBG pre-training on HDTF latent windows (cold-start stage).

The paper trains SFBG/RFBG end-to-end on CHAT-AVD-50k; that dataset is produced BY
the pipeline, so we cold-start with self-supervised pre-training on HDTF:
  x0       = a 16-frame latent window of a real talking face (the refinement target)
  init     = x0 + Gaussian corruption (sigma ~ U[0.05, 0.3]) — teaches "refine an
             imperfect initial segment" (the SDEdit-style role of \bar v)
  partner  = a window from a DIFFERENT video (pseudo-partner: activates the
             multi-scale conditioning pathway; true responsiveness needs dyadic
             data, wired in the CHAT-AVD stage)
  identity = another frame latent of the SAME video
  audio/emotion = optional (None in stage 1; the aux stream then gets no gradient,
             which is fine for single-GPU pre-training — no DDP here)

Data: directory of .pt latent files, each (F_total, 4, 64, 64) fp16/fp32 (SD1.5-VAE,
x0.18215 scaled), one per clip — OR mp4 clips encoded on the fly via chat.ifbg.bridge
(slow path). Run in the `video` conda env:

  python scripts/train_rfbg.py --data <latents_dir> --out runs/rfbg_pre \
      [--steps 20000] [--batch 1] [--accum 8] [--frames 16] [--lr 1e-4]
"""

import argparse
import csv
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.dirname(HERE)
sys.path.insert(0, CODE_ROOT)

import torch  # noqa: E402

from chat.ifbg.ifbr.rfbg import RFBG, RFBGConfig  # noqa: E402


class LatentWindows(torch.utils.data.Dataset):
    """Random (x0, init, partner, identity) windows from per-clip latent files."""

    def __init__(self, data_dir: str, frames: int = 16, partner_frames: int = 48,
                 corrupt_lo: float = 0.05, corrupt_hi: float = 0.3, seed: int = 0):
        self.files = sorted(
            os.path.join(r, f) for r, _, fs in os.walk(data_dir) for f in fs if f.endswith(".pt")
        )
        if len(self.files) < 2:
            raise SystemExit(f"need >=2 latent .pt files under {data_dir}, found {len(self.files)}")
        self.frames = frames
        self.partner_frames = partner_frames
        self.corrupt = (corrupt_lo, corrupt_hi)
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.files)

    def _load(self, path: str) -> torch.Tensor:
        lat = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(lat, torch.Tensor):
            raise ValueError(f"{path}: not a latent tensor (got {type(lat).__name__}) — keep checkpoints out of the data dir")
        if lat.ndim != 4 or lat.shape[1] != 4:
            raise ValueError(f"{path}: expected (F,4,h,w), got {tuple(lat.shape)}")
        return lat.float()

    def _window(self, lat: torch.Tensor, n: int) -> torch.Tensor:
        if lat.shape[0] <= n:
            reps = -(-n // lat.shape[0])
            lat = lat.repeat(reps, 1, 1, 1)
        start = self.rng.randint(0, lat.shape[0] - n)
        return lat[start : start + n]

    def __getitem__(self, idx: int):
        lat = self._load(self.files[idx])
        x0 = self._window(lat, self.frames)
        sigma = self.rng.uniform(*self.corrupt)
        init = x0 + sigma * torch.randn_like(x0)
        identity = lat[self.rng.randint(0, lat.shape[0] - 1)]
        other = self.files[self.rng.randrange(len(self.files) - 1)]
        if other == self.files[idx]:
            other = self.files[-1]
        partner = self._window(self._load(other), self.partner_frames)
        return x0, init, partner, identity


class PairedWindows(torch.utils.data.Dataset):
    """TRUE-RESPONSE pairs from dyadic latents (NoXI: session/Expert_video|Novice_video/k.pt).

    x0 = a window of one side's clip k; partner = the SAME clip index of the OTHER
    side (temporally aligned dyadic footage) — the conditioning now carries real
    responsive semantics, unlike the pseudo-partner pre-training stage.

    corrupt_lo default raised from an earlier 0.05: at that level `init` is
    nearly x0 itself, so the epsilon-prediction loss is solvable from init alone
    and the partner cross-attention pathway is under-constrained -- diagnosed as
    the likely mechanism behind a reproducible training collapse observed at this
    reproduction's scale. partner_dropout (zero the partner window
    some fraction of steps, CFG-style) keeps a working init-only fallback so the
    model can't become dependent on a conditioning pathway it then abandons."""

    def __init__(self, data_dir: str, frames: int = 8, partner_frames: int = 24,
                 corrupt_lo: float = 0.2, corrupt_hi: float = 0.3,
                 partner_dropout: float = 0.15, seed: int = 0):
        self.pairs = []
        for session in sorted(os.listdir(data_dir)):
            sdir = os.path.join(data_dir, session)
            a, b = os.path.join(sdir, "Expert_video"), os.path.join(sdir, "Novice_video")
            if not (os.path.isdir(a) and os.path.isdir(b)):
                continue
            for f in sorted(os.listdir(a)):
                if f.endswith(".pt") and os.path.exists(os.path.join(b, f)):
                    self.pairs.append((os.path.join(a, f), os.path.join(b, f)))
        if not self.pairs:
            raise SystemExit(f"no Expert/Novice latent pairs under {data_dir}")
        self.frames = frames
        self.partner_frames = partner_frames
        self.corrupt = (corrupt_lo, corrupt_hi)
        self.partner_dropout = partner_dropout
        self.rng = random.Random(seed)

    def __len__(self):
        return 2 * len(self.pairs)  # each pair used in both directions

    def _load(self, path):
        lat = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(lat, torch.Tensor):
            raise ValueError(f"{path}: not a latent tensor")
        return lat.float()

    def _window(self, lat, n, start=None):
        if lat.shape[0] <= n:
            lat = lat.repeat(-(-n // lat.shape[0]), 1, 1, 1)
        if start is None:
            start = self.rng.randint(0, lat.shape[0] - n)
        start = min(start, lat.shape[0] - n)
        return lat[start : start + n], start

    def __getitem__(self, idx):
        a_path, b_path = self.pairs[idx % len(self.pairs)]
        if idx >= len(self.pairs):  # swap roles: each side learns to respond
            a_path, b_path = b_path, a_path
        me, other = self._load(a_path), self._load(b_path)
        x0, start = self._window(me, self.frames)
        # partner window aligned by PROPORTIONAL position in the timeline, not a
        # clamped absolute index -- the old `max(0, start - frames)` silently
        # clamped to other's length when the two sides' clip lengths differ,
        # anti-aligning x0 (late in me) against partner (forced to other's start).
        frac = start / max(1, me.shape[0] - self.frames)
        p_target = frac * max(1, other.shape[0] - self.partner_frames)
        p_start = max(0, min(int(round(p_target)), other.shape[0] - self.partner_frames))
        partner, _ = self._window(other, self.partner_frames, start=p_start)
        sigma = self.rng.uniform(*self.corrupt)
        init = x0 + sigma * torch.randn_like(x0)
        identity = me[self.rng.randint(0, me.shape[0] - 1)]
        if self.rng.random() < self.partner_dropout:
            partner = torch.zeros_like(partner)
        return x0, init, partner, identity


def _build_probe(args, cfg):
    """Prepare fixed HDTF/NoXI probe tensors once, for periodic decoded-sample
    logging during training -- catches a visual collapse (loss can stay bounded
    while samples go to noise) between manual checkpoint inspections, directly on
    the training dashboard."""
    import numpy as np
    from PIL import Image

    from chat.ifbg.bridge import load_vae, encode_video, decode_video

    dev = args.device
    vae = load_vae(args.probe_vae, device=dev)
    frames = args.frames
    out = {"vae": vae, "decode_video": decode_video}

    def read_prefix(path, n):
        import av

        with av.open(path) as c:
            fr = []
            for f in c.decode(c.streams.video[0]):
                fr.append(f.to_ndarray(format="rgb24"))
                if len(fr) >= n:
                    break
        arr = np.stack(fr).astype(np.float32) / 255.0
        return torch.from_numpy(arr.transpose(0, 3, 1, 2))

    if args.probe_x1_dir:
        d = args.probe_x1_dir
        f_i, f_j = read_prefix(f"{d}/video_i.mp4", 64), read_prefix(f"{d}/video_j.mp4", 64)
        lat_s = encode_video(vae, f_i).float().cpu()
        lat_p = encode_video(vae, f_j).float().cpu()
        ident_img = np.array(Image.open(f"{d}/identity_i.png").convert("RGB"))
        ident = torch.from_numpy(ident_img.astype(np.float32) / 255.0).permute(2, 0, 1)
        ident_lat = encode_video(vae, ident.unsqueeze(0)).float().cpu()[0]
        out["hdtf"] = (lat_s[:frames], lat_p[: 3 * frames], ident_lat)

    if args.probe_noxi_pair:
        a_path, b_path = args.probe_noxi_pair
        lat_s = torch.load(a_path, map_location="cpu", weights_only=True).float()
        lat_p = torch.load(b_path, map_location="cpu", weights_only=True).float()
        out["noxi"] = (lat_s[:frames], lat_p[: 3 * frames], lat_s[0])

    return out


def _run_probe(probe, model, device):
    """Sample the probe model against every prepared domain, return {name: (H,W,3) uint8}."""
    import numpy as np

    images = {}
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for name in ("hdtf", "noxi"):
            if name not in probe:
                continue
            x0, partner, identity = probe[name]
            out = model.sample(x0.unsqueeze(0).to(device), partner.unsqueeze(0).to(device),
                                None, None, identity.unsqueeze(0).to(device), strength=0.5)
            pixels = probe["decode_video"](probe["vae"], out[0].float().cpu()).float().cpu()
            frame0 = (pixels[0].permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
            images[name] = frame0
    if was_training:
        model.train()
    return images


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dir of latent .pt files (flat, or dyadic tree with --paired)")
    ap.add_argument("--paired", action="store_true", help="data is a dyadic session tree (true-response fine-tuning)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--partner-frames", type=int, default=48)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bf16", action="store_true", default=True)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--resume", help="checkpoint to resume from")
    ap.add_argument("--resume-weights-only", action="store_true",
                    help="load only model weights from --resume, start AdamW fresh -- use when "
                         "fine-tuning on a substantially different data distribution/config than "
                         "the checkpoint was trained under, so stale Adam moments (tuned to the old "
                         "distribution's gradient statistics) can't cause an instability spike")
    ap.add_argument("--spike-factor", type=float, default=6.0,
                    help="skip a macro-step if any micro-batch loss exceeds this multiple of the "
                         "running loss EMA (0=disabled); guards against a single corrupt/outlier "
                         "clip permanently knocking the model into a bad region")
    ap.add_argument("--corrupt-lo", type=float, default=None,
                    help="override the dataset's min init-corruption sigma (default depends on "
                         "--paired: 0.2 paired / 0.05 unpaired)")
    ap.add_argument("--partner-dropout", type=float, default=0.15,
                    help="--paired only: probability of zeroing the partner window per sample "
                         "(CFG-style dropout so the model keeps a working init-only fallback "
                         "instead of becoming dependent on a conditioning pathway it can abandon)")
    ap.add_argument("--ema-decay", type=float, default=0.999,
                    help="EMA of model weights, saved alongside raw weights as 'model_ema' in "
                         "checkpoints (0=disabled). A passive lag filter -- does not prevent a "
                         "collapse, just gives a usable pre-collapse snapshot as insurance")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tiny", action="store_true", help="tiny model config for CPU smoke tests")
    ap.add_argument("--embed-dim", type=int, default=None, help="capacity override (default 128 = paper-scale 1.7M)")
    ap.add_argument("--cond-dim", type=int, default=None)
    ap.add_argument("--num-layers", type=int, default=None)
    ap.add_argument("--mlp-ratio", type=int, default=None)
    ap.add_argument("--num-heads", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=2, help="dataloader workers (bump on NFS to hide latent-load latency)")
    ap.add_argument("--qk-norm", action="store_true",
                    help="build the model with QK-normalised conditioning cross-attention "
                         "(bounds attention logits; resume a graft_qknorm checkpoint with this on)")
    ap.add_argument("--contrast-weight", type=float, default=0.0,
                    help="--paired only: weight of the partner-discrimination hinge (0=off). Forces "
                         "the model to use the partner: correct partner must beat a wrong partner by "
                         "--contrast-margin. Doubles forward cost. Needs batch>=2.")
    ap.add_argument("--contrast-margin", type=float, default=0.02)
    ap.add_argument("--wandb", action="store_true", help="log scalars (+ image probes, if configured) to Weights & Biases")
    ap.add_argument("--wandb-project", default="chat-rfbg")
    ap.add_argument("--wandb-run-name", default=None)
    ap.add_argument("--probe-vae", default=None,
                    help="sd-vae-ft-mse path -- enables periodic decoded-sample image logging to "
                         "wandb every --ckpt-every steps (needs --probe-x1-dir and/or "
                         "--probe-noxi-pair). Cheap (one 8-frame sample), and directly operationalises "
                         "'always render a frame, don't trust the loss curve' during a long run.")
    ap.add_argument("--probe-x1-dir", default=None,
                    help="x1 pipeline output dir (has video_i.mp4/video_j.mp4/identity_i.png) for an "
                         "HDTF-domain probe")
    ap.add_argument("--probe-noxi-pair", nargs=2, default=None, metavar=("EXPERT_PT", "NOVICE_PT"),
                    help="a real NoXI Expert/Novice .pt pair for a NoXI-domain probe")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    if args.paired:
        kwargs = {"frames": args.frames, "partner_frames": args.partner_frames,
                  "partner_dropout": args.partner_dropout, "seed": args.seed}
        if args.corrupt_lo is not None:
            kwargs["corrupt_lo"] = args.corrupt_lo
        ds = PairedWindows(args.data, **kwargs)
        print(f"dataset: {len(ds)} directed dyadic pairs "
              f"(corrupt_lo={ds.corrupt[0]}, partner_dropout={ds.partner_dropout})")
    else:
        kwargs = {"frames": args.frames, "partner_frames": args.partner_frames, "seed": args.seed}
        if args.corrupt_lo is not None:
            kwargs["corrupt_lo"] = args.corrupt_lo
        ds = LatentWindows(args.data, **kwargs)
        print(f"dataset: {len(ds)} clips")
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True,
                                     num_workers=args.num_workers, drop_last=True, persistent_workers=True)

    cfg = RFBGConfig(qk_norm=args.qk_norm)
    if args.tiny:
        cfg = RFBGConfig(embed_dim=32, cond_dim=16, num_layers=2, num_timesteps=8, mlp_ratio=2,
                         qk_norm=args.qk_norm)
    # capacity overrides (for the A100 scale-up; defaults keep the paper-scale 1.7M model)
    for k in ("embed_dim", "cond_dim", "num_layers", "mlp_ratio", "num_heads"):
        v = getattr(args, k, None)
        if v is not None:
            setattr(cfg, k, v)
    model = RFBG(cfg).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"RFBG params: {n_params/1e6:.1f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    ema_model = None
    if args.ema_decay > 0:
        ema_model = RFBG(cfg).to(args.device)
        ema_model.load_state_dict(model.state_dict())
        for p in ema_model.parameters():
            p.requires_grad_(False)
        ema_model.eval()

    wandb_run = None
    if args.wandb:
        import wandb

        wandb_run = wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    probe = None
    if args.probe_vae and (args.probe_x1_dir or args.probe_noxi_pair):
        probe = _build_probe(args, cfg)

    start_step = 0
    if args.resume:
        state = torch.load(args.resume, map_location=args.device, weights_only=True)
        model.load_state_dict(state["model"])
        # Released checkpoints ship weights only, so "opt" may be absent; that is the
        # same situation as --resume-weights-only and is handled as such.
        weights_only = args.resume_weights_only or "opt" not in state
        if not weights_only:
            opt.load_state_dict(state["opt"])
        elif "opt" not in state:
            print(f"{args.resume} carries no optimiser state: resuming weights only, "
                  "fresh AdamW, step counter restarts at 0")
        # weights-only resume = a fresh training phase (fresh AdamW state); the step
        # counter must also start fresh, else --steps N runs only (N - checkpoint's
        # step) real steps and --ckpt-every is offset, silently delaying/shortening
        # this phase (bit us: a 12000-step finetune from a 6000-step pretrain ckpt
        # only ran 6000 real steps and didn't checkpoint until step 7000).
        start_step = 0 if weights_only else state["step"]
        for g in opt.param_groups:  # CLI lr must win over the checkpointed lr
            g["lr"] = args.lr
        if ema_model is not None:
            ema_model.load_state_dict(state.get("model_ema", state["model"]))
        tag = " [weights-only, fresh AdamW state]" if weights_only else ""
        print(f"resumed from {args.resume} at step {start_step} (lr set to {args.lr}){tag}")

    log_path = os.path.join(args.out, "train_log.csv")
    log_new = not os.path.exists(log_path)
    log = open(log_path, "a", newline="")
    writer = csv.writer(log)
    if log_new:
        writer.writerow(["step", "loss", "lr", "sec_per_step"])

    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                              enabled=(args.bf16 and args.device == "cuda"))
    model.train()
    step = start_step
    it = iter(dl)
    t0 = time.time()
    running = 0.0
    loss_ema = None
    skipped_spikes = 0
    while step < args.steps:
        opt.zero_grad(set_to_none=True)
        acc = 0.0
        spike = None
        for _ in range(args.accum):
            try:
                x0, init, partner, identity = next(it)
            except StopIteration:
                it = iter(dl)
                x0, init, partner, identity = next(it)
            x0, init = x0.to(args.device), init.to(args.device)
            partner, identity = partner.to(args.device), identity.to(args.device)
            # wrong partner = the batch rolled by one, so each sample sees a mismatched
            # partner (a valid negative for the contrastive term). Needs batch>=2.
            wrong = partner.roll(1, 0) if (args.contrast_weight > 0 and partner.shape[0] > 1) else None
            with autocast:
                loss = model.training_loss(x0, init, partner, None, None, identity,
                                           wrong_partner=wrong, contrast_weight=args.contrast_weight,
                                           contrast_margin=args.contrast_margin) / args.accum
            raw = float(loss.detach()) * args.accum
            is_spike = (args.spike_factor > 0 and loss_ema is not None and step > 200
                        and raw > args.spike_factor * loss_ema)
            # EMA updates UNCONDITIONALLY (even on batches we're about to skip) so it tracks
            # the true recent loss distribution; updating only on accepted batches would
            # self-reinforce toward whatever subset passes, dragging the EMA down and making
            # progressively more normal batches look like "spikes".
            loss_ema = raw if loss_ema is None else 0.98 * loss_ema + 0.02 * raw
            if is_spike:
                spike = raw
                # free the un-backwarded graph BEFORE the next forward runs; keeping it
                # alive across the skip transiently doubles activation memory and OOMs
                # a near-full GPU (observed: first-ever skip at step 332 -> instant OOM
                # on a 24GB a30 sitting at ~23.2GB steady state)
                del loss
                break
            loss.backward()
            acc += float(loss.detach())
        if spike is not None:
            opt.zero_grad(set_to_none=True)  # discard partial grads from this macro-step
            skipped_spikes += 1
            print(f"step {step}: SKIP anomalous batch (loss {spike:.3f} vs ema {loss_ema:.3f}), "
                  f"skipped {skipped_spikes} total", flush=True)
            continue
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if ema_model is not None:
            with torch.no_grad():
                for ema_p, p in zip(ema_model.parameters(), model.parameters()):
                    ema_p.mul_(args.ema_decay).add_(p, alpha=1 - args.ema_decay)
        step += 1
        running += acc
        if step % 20 == 0:
            dt = (time.time() - t0) / 20
            t0 = time.time()
            print(f"step {step}/{args.steps}  loss {running/20:.4f}  {dt:.2f}s/step", flush=True)
            writer.writerow([step, f"{running/20:.5f}", args.lr, f"{dt:.2f}"])
            log.flush()
            if wandb_run is not None:
                wandb_run.log({"loss": running / 20, "lr": args.lr, "sec_per_step": dt,
                                "skipped_spikes_total": skipped_spikes, "loss_ema": loss_ema}, step=step)
            running = 0.0
        if step % args.ckpt_every == 0 or step == args.steps:
            path = os.path.join(args.out, f"rfbg_{step:06d}.pt")
            state = {"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                     "cfg": vars(model.cfg)}
            if ema_model is not None:
                state["model_ema"] = ema_model.state_dict()
            torch.save(state, path)
            print(f"ckpt -> {path}", flush=True)
            if wandb_run is not None and probe is not None:
                import wandb as _wandb

                log_dict = {}
                for name, img in _run_probe(probe, model, args.device).items():
                    log_dict[f"probe/{name}_raw"] = _wandb.Image(img, caption=f"step {step} raw")
                if ema_model is not None:
                    for name, img in _run_probe(probe, ema_model, args.device).items():
                        log_dict[f"probe/{name}_ema"] = _wandb.Image(img, caption=f"step {step} ema")
                if log_dict:
                    wandb_run.log(log_dict, step=step)
                    print(f"probe images logged to wandb @ step {step}", flush=True)
    log.close()
    if wandb_run is not None:
        wandb_run.finish()
    print("done")


if __name__ == "__main__":
    main()
