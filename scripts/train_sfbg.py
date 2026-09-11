"""SFBG pre-training on HDTF latent windows (cold-start stage).

SilentDiff generates the silent segment between two talking segments (paper Sec.
4.3). Cold-start proxy on HDTF (no listener footage): sample a contiguous
[prev | target | next] triple from one clip —
  x0    = the middle window (generation target)
  prev  = the window before it, next = the window after it (conditioning)
  identity = another frame latent of the SAME video
This teaches the structural task (contextual in-betweening with identity
preservation); true LISTENING behaviour needs dyadic data (REACT / CHAT-AVD),
wired in a later stage. Run in the `video` conda env:

  python scripts/train_sfbg.py --data <latents_dir> --out runs/sfbg_pre \
      [--steps 20000] [--batch 2] [--accum 4] [--frames 8] [--ctx-frames 8]
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

from chat.ifbg.ifbr.sfbg import SFBG, SFBGConfig  # noqa: E402


class TripleWindows(torch.utils.data.Dataset):
    """Contiguous (prev, x0, next, identity) triples from per-clip latent files."""

    def __init__(self, data_dir: str, frames: int = 8, ctx_frames: int = 8, seed: int = 0):
        self.files = sorted(
            os.path.join(r, f) for r, _, fs in os.walk(data_dir) for f in fs if f.endswith(".pt")
        )
        if not self.files:
            raise SystemExit(f"no latent .pt files under {data_dir}")
        self.frames = frames
        self.ctx = ctx_frames
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        lat = torch.load(self.files[idx], map_location="cpu", weights_only=True)
        if not isinstance(lat, torch.Tensor):
            raise ValueError(f"{self.files[idx]}: not a latent tensor (got {type(lat).__name__}) — keep checkpoints out of the data dir")
        lat = lat.float()
        need = self.frames + 2 * self.ctx
        if lat.shape[0] < need:
            reps = -(-need // lat.shape[0])
            lat = lat.repeat(reps, 1, 1, 1)
        start = self.rng.randint(0, lat.shape[0] - need)
        prev = lat[start : start + self.ctx]
        x0 = lat[start + self.ctx : start + self.ctx + self.frames]
        nxt = lat[start + self.ctx + self.frames : start + need]
        identity = lat[self.rng.randint(0, lat.shape[0] - 1)]
        return x0, prev, nxt, identity


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--ctx-frames", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--resume")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tiny", action="store_true", help="tiny model config for CPU smoke tests")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    ds = TripleWindows(args.data, frames=args.frames, ctx_frames=args.ctx_frames, seed=args.seed)
    print(f"dataset: {len(ds)} clips")
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True,
                                     num_workers=2, drop_last=True, persistent_workers=True)

    cfg = SFBGConfig()
    if args.tiny:
        cfg = SFBGConfig(embed_dim=32, num_layers=2, num_timesteps=8, mlp_ratio=2)
    model = SFBG(cfg).to(args.device)
    print(f"SFBG params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    start_step = 0
    if args.resume:
        state = torch.load(args.resume, map_location=args.device, weights_only=True)
        model.load_state_dict(state["model"])
        # Released checkpoints ship weights only, so "opt" may be absent. Resume the
        # weights and start a fresh AdamW rather than failing with a KeyError; the step
        # counter restarts too, so --steps N means N real steps in that case.
        if "opt" in state:
            opt.load_state_dict(state["opt"])
            start_step = state["step"]
        else:
            start_step = 0
            print(f"{args.resume} carries no optimiser state: resuming weights only, "
                  "fresh AdamW, step counter restarts at 0")
        for g in opt.param_groups:  # CLI lr must win over the checkpointed lr
            g["lr"] = args.lr
        print(f"resumed from {args.resume} at step {start_step} (lr set to {args.lr})")

    log_path = os.path.join(args.out, "train_log.csv")
    log_new = not os.path.exists(log_path)
    log = open(log_path, "a", newline="")
    writer = csv.writer(log)
    if log_new:
        writer.writerow(["step", "loss", "lr", "sec_per_step"])

    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                              enabled=(args.device == "cuda"))
    model.train()
    step = start_step
    it = iter(dl)
    t0 = time.time()
    running = 0.0
    while step < args.steps:
        opt.zero_grad(set_to_none=True)
        for _ in range(args.accum):
            try:
                x0, prev, nxt, identity = next(it)
            except StopIteration:
                it = iter(dl)
                x0, prev, nxt, identity = next(it)
            x0, prev = x0.to(args.device), prev.to(args.device)
            nxt, identity = nxt.to(args.device), identity.to(args.device)
            with autocast:
                loss = model.training_loss(x0, prev, nxt, identity) / args.accum
            loss.backward()
            running += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        if step % 20 == 0:
            dt = (time.time() - t0) / 20
            t0 = time.time()
            print(f"step {step}/{args.steps}  loss {running/20:.4f}  {dt:.2f}s/step", flush=True)
            writer.writerow([step, f"{running/20:.5f}", args.lr, f"{dt:.2f}"])
            log.flush()
            running = 0.0
        if step % args.ckpt_every == 0 or step == args.steps:
            path = os.path.join(args.out, f"sfbg_{step:06d}.pt")
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                        "cfg": vars(model.cfg)}, path)
            print(f"ckpt -> {path}", flush=True)
    log.close()
    print("done")


if __name__ == "__main__":
    main()
