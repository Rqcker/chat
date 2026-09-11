"""CSIM: identity similarity of generated video frames vs the source identity image
(paper Table 1, CSIM = 0.85 target, higher better).

Extracts the ArcFace (antelopev2/glintr100) embedding of the reference identity
image and of every sampled generated frame, then reports the mean pairwise cosine
via chat.eval.identity_similarity plus the beats/meets verdict.

Run in the `video` conda env:
  python scripts/csim_video.py --ref-image portrait.png --videos out.mp4 dir/ \
      --insightface-root ~/.insightface [--frames-per-video 16]
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.dirname(HERE)
sys.path.insert(0, CODE_ROOT)

from chat.eval import TARGETS, identity_similarity, meets_paper  # noqa: E402
from chat.eval.extractors import ArcFaceExtractor  # noqa: E402
from fid_video import collect_mp4s  # noqa: E402  (same frame-decoding conventions)


def video_frames_uint8(path: str, k: int):
    import av

    with av.open(path) as container:
        frames = [f.to_ndarray(format="rgb24") for f in container.decode(container.streams.video[0])]
    if not frames:
        raise ValueError(f"no frames decoded from {path}")
    if k and len(frames) > k:
        idx = np.linspace(0, len(frames) - 1, k).round().astype(int)
        frames = [frames[i] for i in idx]
    return frames


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-image", required=True, help="source identity image")
    ap.add_argument("--videos", nargs="+", required=True, help=".mp4 files and/or directories")
    ap.add_argument("--insightface-root", required=True)
    ap.add_argument("--frames-per-video", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from PIL import Image

    ext = ArcFaceExtractor(root=args.insightface_root, device=args.device)
    ref = ext.embed_frame(np.array(Image.open(args.ref_image).convert("RGB")))
    if ref is None:
        raise SystemExit(f"no face detected in {args.ref_image}")

    mp4s = collect_mp4s(args.videos)
    if not mp4s:
        raise SystemExit("no .mp4 inputs found")
    embs, skipped = [], 0
    for path in mp4s:
        e, s = ext.embed_frames(video_frames_uint8(path, args.frames_per_video))
        embs.append(e)
        skipped += s
        print(f"  {path}: {e.shape[0]} faces ({s} frames skipped)")
    embs = np.concatenate(embs, axis=0)

    csim = identity_similarity(embs, np.tile(ref, (embs.shape[0], 1)))
    target = TARGETS["CSIM"][0]
    print(f"\nCSIM = {csim:.4f}  (paper Full CHAT target: {target}, higher better; "
          f"{embs.shape[0]} faces, {skipped} skipped)")
    print("meets paper (never worse):", meets_paper("CSIM", csim))


if __name__ == "__main__":
    main()
