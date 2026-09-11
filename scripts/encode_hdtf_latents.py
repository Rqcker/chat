"""Pre-encode HDTF clips to SD1.5-VAE latents for RFBG/SFBG pre-training.

Each 81-frame clip -> (81, 4, 64, 64) fp16 tensor (sd-vae-ft-mse, x0.18215 scaled)
saved as <out>/<clip_stem>.pt (~2.6 MB each). Resumable: existing outputs are
skipped. Run in the `video` conda env (GPU):

  python scripts/encode_hdtf_latents.py --clips <hdtf/clips_extracted> \
      --vae <hallo2/pretrained_models/sd-vae-ft-mse> --out <latents_dir> \
      [--limit 2000] [--seed 0]
"""

import argparse
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.dirname(HERE)
sys.path.insert(0, CODE_ROOT)

from chat.ifbg.bridge import encode_video, load_vae  # noqa: E402


def read_frames(path: str) -> "np.ndarray":
    import av

    with av.open(path) as c:
        frames = [f.to_ndarray(format="rgb24") for f in c.decode(c.streams.video[0])]
    arr = np.stack(frames).astype(np.float32) / 255.0
    return arr.transpose(0, 3, 1, 2)  # (F,3,H,W)


def main() -> None:
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True)
    ap.add_argument("--vae", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=2000, help="number of clips to encode")
    ap.add_argument("--seed", type=int, default=0, help="random subset selection seed")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-frames", type=int, default=0,
                    help="cap frames per clip before encoding (0=all); long clips waste VAE time")
    ap.add_argument("--preserve-tree", action="store_true",
                    help="mirror the input directory structure (needed for paired dyadic data)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    clips = sorted(
        os.path.join(r, f) for r, _, fs in os.walk(args.clips) for f in fs if f.endswith(".mp4")
    )
    rng = random.Random(args.seed)
    rng.shuffle(clips)
    clips = clips[: args.limit]
    print(f"encoding {len(clips)} clips -> {args.out}")

    vae = load_vae(args.vae, device="cuda")
    done = skipped = failed = 0
    for idx, path in enumerate(clips):
        if args.preserve_tree:
            rel = os.path.relpath(path, args.clips)
            dst = os.path.join(args.out, os.path.splitext(rel)[0] + ".pt")
            os.makedirs(os.path.dirname(dst), exist_ok=True)
        else:
            stem = os.path.splitext(os.path.basename(path))[0]
            dst = os.path.join(args.out, stem + ".pt")
        if os.path.exists(dst):
            skipped += 1
            continue
        try:
            frames = torch.from_numpy(read_frames(path))
            if args.max_frames and frames.shape[0] > args.max_frames:
                frames = frames[: args.max_frames]
            if frames.shape[-1] != 512 or frames.shape[-2] != 512:
                frames = torch.nn.functional.interpolate(frames, size=(512, 512),
                                                         mode="bilinear", align_corners=False)
            lat = encode_video(vae, frames, batch_size=args.batch).to(torch.float16).cpu()
            torch.save(lat, dst + ".tmp")
            os.replace(dst + ".tmp", dst)  # atomic: resume never sees partial files
            done += 1
        except Exception as exc:  # noqa: BLE001 — one corrupt clip must not kill the run
            failed += 1
            print(f"  FAIL {stem}: {exc}", flush=True)
        if (idx + 1) % 100 == 0:
            print(f"  {idx + 1}/{len(clips)} (done {done}, skipped {skipped}, failed {failed})",
                  flush=True)
    print(f"finished: done {done}, skipped {skipped}, failed {failed}")


if __name__ == "__main__":
    main()
