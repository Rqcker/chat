"""Compute FVD (Frechet Video Distance) of generated videos vs a real reference
(paper Table 1, target 365.03, lower better).

Uses cd-fvd's standard Kinetics-I3D feature extractor, so the FVD values are
comparable with the literature (the paper cites the same FVD, Unterthiner 2018).
Each video is chunked into non-overlapping T-frame clips (I3D operates on clips,
not single frames); clips are resized to a common size and fed as uint8
(N, T, H, W, C), which is cd-fvd's expected format.

Run in the `video` conda env (needs cd-fvd; I3D weights cache in TORCH_HOME):
  python scripts/fvd_video.py --ref-videos hdtf_dir/ --videos gen_dir/ \
      [--clip-len 16] [--size 256] [--max-clips 512] [--device cuda]
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.dirname(HERE)
sys.path.insert(0, CODE_ROOT)

from chat.eval import TARGETS, meets_paper  # noqa: E402
from fid_video import collect_mp4s  # noqa: E402  (same mp4-collection convention)


def video_clips(path, clip_len, size, max_clips, reframe_root=None):
    """Decode an mp4 into non-overlapping clips -> (n, clip_len, size, size, 3) uint8.

    reframe_root: if set, apply the identity-framing transform (reframe_utils) to the
    selected clips' frames before the resize — one detection per video so the transform
    is temporally stable. Clip selection happens BEFORE the (expensive) transform.
    """
    import av
    import cv2

    with av.open(path) as c:
        frames = [f.to_ndarray(format="rgb24") for f in c.decode(c.streams.video[0])]
    if len(frames) < clip_len:
        return np.empty((0, clip_len, size, size, 3), dtype=np.uint8)
    n = len(frames) // clip_len
    sel = np.arange(n)
    if max_clips and n > max_clips:
        sel = np.linspace(0, n - 1, max_clips).round().astype(int)

    params = None
    if reframe_root is not None:
        from reframe_utils import apply_reframe, measure_reframe

        params = measure_reframe(frames[0], reframe_root)
        if params is None:
            return np.empty((0, clip_len, size, size, 3), dtype=np.uint8)

    clips = []
    for ci in sel:
        blk = frames[ci * clip_len: (ci + 1) * clip_len]
        if params is not None:
            blk = [apply_reframe(f, params) for f in blk]
        clips.append(np.stack([cv2.resize(f, (size, size)) for f in blk]))
    return np.stack(clips).astype(np.uint8)


def collect_clips(paths, clip_len, size, max_clips, label, per_video=0, reframe_root=None):
    mp4s = collect_mp4s(paths)
    if not mp4s:
        raise SystemExit(f"no .mp4 found for {label}: {paths}")
    out = []
    for p in mp4s:
        c = video_clips(p, clip_len, size, max_clips=per_video, reframe_root=reframe_root)
        if c.shape[0]:
            out.append(c)
        print(f"  [{label}] {os.path.basename(p)}: {c.shape[0]} clips")
    allc = np.concatenate(out, axis=0)
    if max_clips and allc.shape[0] > max_clips:  # global cap across videos
        idx = np.linspace(0, allc.shape[0] - 1, max_clips).round().astype(int)
        allc = allc[idx]
    return allc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-videos", nargs="+", required=True, help="real reference mp4s/dirs")
    ap.add_argument("--videos", nargs="+", required=True, help="generated mp4s/dirs")
    ap.add_argument("--clip-len", type=int, default=16)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--max-clips", type=int, default=512, help="global cap per side (memory)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ref-reframe", action="store_true",
                    help="apply the identity-framing transform to REFERENCE clips so both "
                         "sides share the framing distribution (FFHQ-style alignment of both "
                         "sides); implies a small per-video clip cap to bound transform cost")
    ap.add_argument("--insightface-root",
                    default=os.environ.get("INSIGHTFACE_ROOT",
                                           os.path.expanduser("~/.insightface")),
                    help="insightface model root; insightface ignores INSIGHTFACE_HOME, "
                         "so it has to be passed explicitly")
    args = ap.parse_args()

    from cdfvd import fvd

    print("collecting reference clips...")
    ref_per_video = 3 if args.ref_reframe else 0
    real = collect_clips(args.ref_videos, args.clip_len, args.size, args.max_clips, "ref",
                         per_video=ref_per_video,
                         reframe_root=args.insightface_root if args.ref_reframe else None)
    print("collecting generated clips...")
    gen = collect_clips(args.videos, args.clip_len, args.size, args.max_clips, "gen")
    if real.shape[0] < 2 or gen.shape[0] < 2:
        raise SystemExit(f"need >=2 clips per side (ref={real.shape[0]}, gen={gen.shape[0]})")

    ev = fvd.cdfvd("i3d", device=args.device)
    score = ev.compute_fvd(real, gen)
    target = TARGETS["FVD"][0]
    print(f"\nreference clips={real.shape[0]}  generated clips={gen.shape[0]}")
    print(f"FVD = {score:.2f}  (paper Full CHAT target: {target}, lower better)")
    print("meets paper (never worse):", meets_paper("FVD", score))
    print("NOTE: FVD was retired from the reported set after a floor control showed it")
    print("      does not separate generated from real video here; see DOCS.md.")
    if gen.shape[0] < 64:
        print("NOTE: <64 clips — FVD is high-variance at this N; scale up for a reportable number")


if __name__ == "__main__":
    main()
