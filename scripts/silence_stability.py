"""Silent-region lip stability for dyadic talking-head video.

**What this measures and why it exists.** Our evaluation scores lip-sync on speech regions
only (`lse_video.py --speech-only`). Dyadic dialogue tracks are roughly half silence by
turn-taking design, so that gate discards the half of the signal where a listening face is
most likely to misbehave, and no paper metric reports it directly. The paper's SFBG
block exists to control exactly that behaviour. SFBG is wired as an opt-in,
default-off inference path; this diagnostic is the protocol for comparing the default
path, SFBG arms and real-video controls. It is a diagnostic only: it carries no paper
target and no pass/fail line, and no number in DOCS.md depends on it.

The statistic follows THEval (arXiv 2511.04520v3), which splits a clip with a voice-activity
detector and sends the two regions to two different measurements rather than folding silence
into a synchronisation score: speech frames to a lip-sync metric, silent frames (>= 300 ms) to
a **Silent Lip Stability** score. Stability there is the **median absolute deviation** of
mouth opening about its own median, normalised by inter-ocular distance, which is robust to
the occasional bad landmark frame in a way that a standard deviation is not.

    stability = median_j( | d_lip,j - median(d_lip) | )        lower is better

**A mouth that is closed and still scores near zero. A mouth that flaps while its owner is
listening does not.** The normalisation by inter-ocular distance makes it comparable across
face scales; it is a ratio, not pixels.

Silence is taken as the complement of this repository's own RMS speech gate, so the split is
identical to the one `lse_video.py --speech-only` already uses and the two are directly
complementary. That deliberately does NOT introduce a second VAD: a different splitter would
make the numbers non-comparable with every LSE figure in the ledger.

Landmarks come from insightface `landmark_3d_68` (inner lip 62/66, eye corners 36/45), the
same detector and the same mouth-opening definition `x1_faces.anim_score` already uses.

**This is a diagnostic, not a paper metric.** It appears in no table of the CHAT paper. Report
it separately, as evidence about listening behaviour that FRCorr would otherwise be the only
source of, and FRCorr is licence-blocked.

Run in the `video` conda env:
  python scripts/silence_stability.py --video refined_j.mp4 --audio track_j.wav
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.dirname(HERE)
sys.path.insert(0, CODE_ROOT)
sys.path.insert(0, HERE)

from lse_video import MIN_SEG, speech_segments  # noqa: E402  (shared gate: one splitter only)

MIN_SILENCE = 0.30   # seconds; THEval's threshold for a scorable silent segment


def silent_segments(wav_path, duration, thresh_frac=0.15, min_silence=MIN_SILENCE):
    """Complement of the RMS speech gate, as [(start_s, end_s), ...].

    Uses `lse_video.speech_segments` so the split matches the one every LSE number in this
    project was scored under. Segments shorter than `min_silence` are dropped, following
    THEval's 300 ms floor.
    """
    speech = speech_segments(wav_path, thresh_frac=thresh_frac)
    out, cursor = [], 0.0
    for a, b in speech:
        if a - cursor >= min_silence:
            out.append((cursor, a))
        cursor = max(cursor, b)
    if duration - cursor >= min_silence:
        out.append((cursor, duration))
    return out


def mouth_opening_series(video, segments, insightface_root, stride=1):
    """Inter-ocular-normalised mouth opening per frame inside `segments`.

    Returns (values, n_frames_seen, n_frames_without_face).
    """
    import cv2
    from reframe_utils import detect_largest

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    wanted = []
    for a, b in segments:
        lo, hi = int(round(a * fps)), int(round(b * fps))
        wanted.append((lo, hi))

    vals, seen, missed = [], 0, 0
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        inside = any(lo <= idx < hi for lo, hi in wanted)
        if inside and (idx % stride == 0):
            seen += 1
            face, off = detect_largest(frame[:, :, ::-1], insightface_root)
            lmk = getattr(face, "landmark_3d_68", None) if face is not None else None
            if lmk is None:
                missed += 1
            else:
                pts = np.asarray(lmk)[:, :2] - off
                # inner lip 62 (upper) / 66 (lower); eye corners 36 (outer L) / 45 (outer R)
                mouth = float(np.linalg.norm(pts[66] - pts[62]))
                inter_ocular = float(np.linalg.norm(pts[45] - pts[36]))
                if inter_ocular > 1e-6:
                    vals.append(mouth / inter_ocular)
                else:
                    missed += 1
        idx += 1
    cap.release()
    return np.asarray(vals, dtype=float), seen, missed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--audio", required=True, help="wav defining the speech/silence split")
    ap.add_argument("--insightface-root",
                    default=os.environ.get("INSIGHTFACE_ROOT",
                                           os.path.expanduser("~/.insightface")),
                    help="insightface model root; insightface ignores INSIGHTFACE_HOME, "
                         "so the location is passed explicitly")
    ap.add_argument("--stride", type=int, default=1,
                    help="sample every Nth frame inside silent regions (1 = every frame)")
    ap.add_argument("--min-silence", type=float, default=MIN_SILENCE,
                    help=f"drop silent runs shorter than this (default {MIN_SILENCE}s, "
                         "THEval's threshold)")
    args = ap.parse_args()

    import cv2
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    duration = n_frames / fps if fps > 0 else 0.0

    speech = speech_segments(args.audio)
    silent = silent_segments(args.audio, duration, min_silence=args.min_silence)
    speech_s = sum(b - a for a, b in speech)
    silent_s = sum(b - a for a, b in silent)
    print(f"clip = {duration:.2f}s @ {fps:.1f}fps  "
          f"speech {speech_s:.2f}s ({len(speech)} seg, gate >= {MIN_SEG}s)  "
          f"silence {silent_s:.2f}s ({len(silent)} seg, >= {args.min_silence}s)")

    if not silent:
        print("no scorable silent region; nothing to report")
        print("SILENT-STABILITY = NA")
        return

    vals, seen, missed = mouth_opening_series(args.video, silent, args.insightface_root,
                                              stride=args.stride)
    if vals.size == 0:
        print(f"no landmarks recovered in {seen} silent frames; cannot score")
        print("SILENT-STABILITY = NA")
        return

    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    print(f"silent frames scored = {vals.size} (of {seen} sampled, {missed} without landmarks)")
    print(f"median mouth opening = {med:.4f}  (inter-ocular normalised)")
    print(f"SILENT-STABILITY = {mad:.4f}  (MAD about the median, lower = steadier)")
    print(f"silent-open-p90 = {float(np.percentile(vals, 90)):.4f}")


if __name__ == "__main__":
    main()
