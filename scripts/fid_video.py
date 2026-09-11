"""Compute FID of generated video(s) against a frozen reference (paper Table 1).

Takes .mp4 files / directories, extracts every frame (or a uniform subset), runs
the standard Inception extractor, and reports FID against the reference built by
scripts/build_fid_reference.py, plus the beats/meets verdict for the paper target.

Run in the `video` conda env:
  python scripts/fid_video.py --ref fid_ref_hdtf.npz --videos out1.mp4 dir/ \
      [--frames-per-video 0 (=all)] [--device cuda]

A trustworthy FID needs >=10k generated frames; small counts are smoke-only.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.dirname(HERE)
sys.path.insert(0, CODE_ROOT)

from chat.eval import TARGETS, frechet_distance, meets_paper  # noqa: E402
from chat.eval.extractors import InceptionExtractor  # noqa: E402


def video_frames(path: str, k: int) -> np.ndarray:
    """Decode RGB frames -> (n, 3, H, W) float [0,1]; k=0 means all frames."""
    import av

    with av.open(path) as container:
        stream = container.streams.video[0]
        frames = [f.to_ndarray(format="rgb24") for f in container.decode(stream)]
    if not frames:
        raise ValueError(f"no frames decoded from {path}")
    if k and len(frames) > k:
        idx = np.linspace(0, len(frames) - 1, k).round().astype(int)
        frames = [frames[i] for i in idx]
    arr = np.stack(frames).astype(np.float32) / 255.0
    return arr.transpose(0, 3, 1, 2)


def collect_mp4s(paths):
    out = []
    for p in paths:
        if os.path.isdir(p):
            out.extend(
                os.path.join(r, f)
                for r, _, fs in os.walk(p)
                for f in fs
                if f.endswith(".mp4")
            )
        elif p.endswith(".mp4"):
            out.append(p)
    return sorted(set(out))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help=".npz from build_fid_reference.py")
    ap.add_argument("--videos", nargs="+", required=True, help=".mp4 files and/or directories")
    ap.add_argument("--frames-per-video", type=int, default=0, help="0 = every frame")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--face-crop", action="store_true",
                    help="tight face crop each frame before Inception (must match ref built with --face-crop)")
    ap.add_argument("--insightface-root",
                    default=os.environ.get("INSIGHTFACE_ROOT",
                                           os.path.expanduser("~/.insightface")),
                    help="insightface model root; insightface ignores INSIGHTFACE_HOME, "
                         "so it has to be passed explicitly")
    args = ap.parse_args()

    ref = np.load(args.ref)
    mu_r, cov_r = ref["mu"], ref["cov"]
    print(f"reference: {args.ref} ({int(ref['n_frames'])} frames)")

    mp4s = collect_mp4s(args.videos)
    if not mp4s:
        raise SystemExit("no .mp4 inputs found")
    if args.face_crop:
        from reframe_utils import apply_face_crop, measure_face_crop

    ext = InceptionExtractor(device=args.device, batch_size=args.batch_size)
    feats, skipped_clips = [], []
    for path in mp4s:
        frames = video_frames(path, args.frames_per_video)
        if args.face_crop:
            first = (frames[0].transpose(1, 2, 0) * 255).astype(np.uint8)
            params = measure_face_crop(first, args.insightface_root)
            if params is None:
                print(f"  {path}: SKIP no face")
                skipped_clips.append(path)
                continue
            frames = np.stack([
                apply_face_crop((f.transpose(1, 2, 0) * 255).astype(np.uint8), params)
                for f in frames
            ]).astype(np.float32).transpose(0, 3, 1, 2) / 255.0
        feats.append(ext.features(frames))
        print(f"  {path}: {frames.shape[0]} frames" + (" (face-crop)" if args.face_crop else ""))
    feats = np.concatenate(feats, axis=0)
    if skipped_clips:
        print(f"\n{len(skipped_clips)} of {len(mp4s)} clips were dropped for face-detection "
              "failure and are not in this FID. A clip the detector cannot find a face in is "
              "usually a bad clip, so dropping it flatters the score.")
        for p in skipped_clips:
            print(f"  dropped: {p}")
    if feats.shape[0] < 2048:
        print(f"WARNING: only {feats.shape[0]} generated frames (<2048-d); "
              "FID is biased — use >=10k frames for a reportable number")

    mu_g = feats.mean(axis=0)
    cov_g = np.cov(feats, rowvar=False)
    fid = frechet_distance(mu_r, cov_r, mu_g, cov_g)
    target = TARGETS["FID"][0]
    print(f"\nFID = {fid:.2f}  (paper Full CHAT target: {target}, lower better)")
    print("meets paper (never worse):", meets_paper("FID", fid))
    print("NOTE: the paper target was measured on 1000 dialogues. At 186 identities the")
    print("      identity-coverage term dominates and real video alone scores above 17.33,")
    print("      so the target is unreachable at this scale.")


if __name__ == "__main__":
    main()
