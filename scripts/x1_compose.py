"""X1 step 5 (video env): compose the paired dyadic clip + score it.

Places the two talking-head videos side by side, mixes both speakers' tracks into
one stereo-ish mono mix, writes dyadic.mp4, and reports CSIM for each side against
its identity portrait.

Usage:
  python scripts/x1_compose.py --out x1_out --insightface-root ~/.insightface
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.dirname(HERE)
sys.path.insert(0, CODE_ROOT)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--insightface-root", required=True)
    ap.add_argument("--frames-per-video", type=int, default=16)
    args = ap.parse_args()

    out = os.path.abspath(args.out)
    import soundfile as sf
    from moviepy.editor import AudioFileClip, VideoFileClip, clips_array

    def pick(side: str) -> str:
        sr = os.path.join(out, f"video_sr_{side}.mp4")  # prefer the CodeFormer SR output
        return sr if os.path.isfile(sr) else os.path.join(out, f"video_{side}.mp4")

    def pick_track(side: str) -> str:
        hq = os.path.join(out, f"track_{side}_24k.wav")  # prefer the native-rate mix track
        return hq if os.path.isfile(hq) else os.path.join(out, f"track_{side}.wav")

    # stereo mix: speaker i panned left, j panned right; RMS-balanced, peak-limited
    wi, sr_i = sf.read(pick_track("i"), dtype="float32")
    wj, sr_j = sf.read(pick_track("j"), dtype="float32")
    assert sr_i == sr_j, f"track rates differ: {sr_i} vs {sr_j}"
    n = max(len(wi), len(wj))
    wi = np.pad(wi, (0, n - len(wi)))
    wj = np.pad(wj, (0, n - len(wj)))
    rms = lambda x: float(np.sqrt(np.mean(x**2)) + 1e-9)  # noqa: E731
    target = max(rms(wi), rms(wj))
    wi, wj = wi * (target / rms(wi)), wj * (target / rms(wj))
    stereo = np.stack([0.75 * wi + 0.25 * wj, 0.25 * wi + 0.75 * wj], axis=1)
    peak = float(np.abs(stereo).max())
    if peak > 0.95:
        stereo *= 0.95 / peak
    mix_path = os.path.join(out, "mix_stereo.wav")
    sf.write(mix_path, stereo, sr_i)

    vi = VideoFileClip(pick("i"))
    vj = VideoFileClip(pick("j"))
    amix = AudioFileClip(mix_path)
    # clamp to the shortest stream minus an epsilon: moviepy raises if audio is
    # read past its decoded duration (video is often a frame or two longer)
    dur = min(vi.duration, vj.duration, amix.duration) - 0.05
    vi, vj = vi.subclip(0, dur), vj.subclip(0, dur)
    dyad = clips_array([[vi, vj]]).set_audio(amix.subclip(0, dur))
    dst = os.path.join(out, "dyadic.mp4")
    dyad.write_videofile(dst, codec="libx264", audio_codec="aac", fps=25,
                         audio_fps=sr_i, logger=None)
    print(f"saved {dst} ({dur:.2f}s, stereo mix @ {sr_i}Hz)")

    # CSIM per side vs its identity portrait
    from PIL import Image

    from chat.eval import identity_similarity
    from chat.eval.extractors import ArcFaceExtractor

    sys.path.insert(0, HERE)
    from csim_video import video_frames_uint8

    ext = ArcFaceExtractor(root=args.insightface_root)
    for side in ("i", "j"):
        ref = ext.embed_frame(np.array(Image.open(os.path.join(out, f"identity_{side}.png")).convert("RGB")))
        frames = video_frames_uint8(pick(side), args.frames_per_video)
        embs, skipped = ext.embed_frames(frames)
        csim = identity_similarity(embs, np.tile(ref, (embs.shape[0], 1)))
        print(f"CSIM speaker {side}: {csim:.4f} ({embs.shape[0]} faces, {skipped} skipped)")


if __name__ == "__main__":
    main()
