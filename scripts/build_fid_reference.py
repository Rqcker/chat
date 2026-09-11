"""Build the FROZEN FID reference statistics from HDTF clips (paper Table 1 FID).

Samples clips deterministically (fixed seed), extracts frames uniformly, runs the
standard pytorch-fid InceptionV3 pool3 extractor, and saves the reference mean +
covariance (mu, cov) plus a manifest documenting exactly which files/frames built
it. The manifest makes the reference set reproducible and citable, which matters
because FID and FVD are meaningless without the reference distribution that
produced them.

Run in the `video` conda env (has torch cu128 + av + pytorch-fid):
  python scripts/build_fid_reference.py --clips <dir with .mp4> --out fid_ref_hdtf.npz \
      [--n-clips 2500] [--frames-per-clip 4] [--seed 0] [--device cuda]

Notes: 2500 clips x 4 frames = 10k reference frames (2048-d features need ~10k+
samples for a stable covariance). The generated side should also use >=10k frames
for a trustworthy number; fewer frames = pipeline smoke only.
"""

import argparse
import json
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.dirname(HERE)
sys.path.insert(0, CODE_ROOT)

from chat.eval.extractors import InceptionExtractor  # noqa: E402


def uniform_frames(path: str, k: int) -> np.ndarray:
    """Decode k uniformly-spaced RGB frames from a video -> (k, 3, H, W) float [0,1]."""
    import av

    with av.open(path) as container:
        stream = container.streams.video[0]
        frames = [f.to_ndarray(format="rgb24") for f in container.decode(stream)]
    if not frames:
        raise ValueError(f"no frames decoded from {path}")
    idx = np.linspace(0, len(frames) - 1, min(k, len(frames))).round().astype(int)
    arr = np.stack([frames[i] for i in idx]).astype(np.float32) / 255.0  # (k,H,W,3)
    return arr.transpose(0, 3, 1, 2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True, help="directory containing reference .mp4 clips")
    ap.add_argument("--out", required=True, help="output .npz (mu, cov); manifest saved alongside")
    ap.add_argument("--n-clips", type=int, default=2500)
    ap.add_argument("--frames-per-clip", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--files-from", default=None,
                    help="manifest.json of an existing reference: reuse its exact clip list "
                         "(builds a protocol variant over the same clips)")
    ap.add_argument("--reframe", action="store_true",
                    help="apply the identity-framing transform (reframe_utils: face ~55%% of a "
                         "512 frame, blur-extended background) to every reference frame, matching "
                         "the framing of generated eval videos (FFHQ-style both-sides alignment). "
                         "One detection per clip; frames without a detectable face are skipped.")
    ap.add_argument("--insightface-root",
                    default=os.environ.get("INSIGHTFACE_ROOT",
                                           os.path.expanduser("~/.insightface")),
                    help="insightface model root; insightface ignores INSIGHTFACE_HOME, "
                         "so it has to be passed explicitly")
    ap.add_argument("--face-crop", action="store_true",
                    help="tight square crop around the largest face (pad=0.35) then resize\n                         to 299 before Inception — paper X_suppl \"face video frames\" protocol.\n                         Mutually exclusive with --reframe in intent; both-sides must match.")
    args = ap.parse_args()

    if args.files_from:
        with open(args.files_from, encoding="utf-8") as f:
            chosen = [os.path.join(args.clips, p) for p in json.load(f)["files"]]
        print(f"reusing {len(chosen)} clips from {args.files_from}")
    else:
        mp4s = sorted(
            os.path.join(r, f)
            for r, _, fs in os.walk(args.clips)
            for f in fs
            if f.endswith(".mp4")
        )
        if not mp4s:
            raise SystemExit(f"no .mp4 found under {args.clips}")
        rng = random.Random(args.seed)
        rng.shuffle(mp4s)
        chosen = mp4s[: args.n_clips]
        print(f"{len(mp4s)} clips found; using {len(chosen)} x {args.frames_per_clip} frames")

    if args.reframe:
        from reframe_utils import apply_reframe, measure_reframe
    if args.face_crop:
        from reframe_utils import apply_face_crop, measure_face_crop

    ext = InceptionExtractor(device=args.device, batch_size=args.batch_size)
    feats, used, failed = [], [], []
    for i, path in enumerate(chosen):
        try:
            frames = uniform_frames(path, args.frames_per_clip)
            if args.face_crop:
                first = (frames[0].transpose(1, 2, 0) * 255).astype(np.uint8)
                params = measure_face_crop(first, args.insightface_root)
                if params is None:
                    raise ValueError("no face detected for face-crop")
                frames = np.stack([
                    apply_face_crop((f.transpose(1, 2, 0) * 255).astype(np.uint8), params)
                    for f in frames
                ]).astype(np.float32).transpose(0, 3, 1, 2) / 255.0
            elif args.reframe:
                first = (frames[0].transpose(1, 2, 0) * 255).astype(np.uint8)
                params = measure_reframe(first, args.insightface_root)
                if params is None:
                    raise ValueError("no face detected for reframe")
                frames = np.stack([
                    apply_reframe((f.transpose(1, 2, 0) * 255).astype(np.uint8), params)
                    for f in frames
                ]).astype(np.float32).transpose(0, 3, 1, 2) / 255.0
        except Exception as exc:  # noqa: BLE001 — a corrupt clip must not kill a long build
            failed.append({"file": os.path.relpath(path, args.clips), "error": str(exc)})
            continue
        feats.append(ext.features(frames))
        used.append(os.path.relpath(path, args.clips))
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(chosen)} clips")
    feats = np.concatenate(feats, axis=0)
    print(f"features: {feats.shape}; failed clips: {len(failed)}")

    mu = feats.mean(axis=0)
    cov = np.cov(feats, rowvar=False)
    np.savez(args.out, mu=mu, cov=cov, n_frames=feats.shape[0])
    manifest = {
        "clips_dir": os.path.abspath(args.clips),
        "n_clips_requested": args.n_clips,
        "n_clips_used": len(used),
        "frames_per_clip": args.frames_per_clip,
        "n_frames": int(feats.shape[0]),
        "seed": args.seed,
        "extractor": "pytorch-fid InceptionV3 pool3 (2048)",
        "reframe": bool(args.reframe),
        "face_crop": bool(args.face_crop),
        "files_from": args.files_from,
        "files": used,
        "failed": failed,
    }
    with open(args.out + ".manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"saved {args.out} (+ .manifest.json)")


if __name__ == "__main__":
    main()
