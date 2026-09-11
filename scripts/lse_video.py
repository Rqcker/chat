"""LSE-C / LSE-D lip-sync error of a generated talking video (paper Table 1:
LSE-C 6.89 higher-better, LSE-D 8.07 lower-better).

Uses the standard SyncNet (Chung & Zisserman), the same metric family the paper
cites (prajwal2020lip / li2024latentsync), so the numbers are literature-comparable.
LSE-D = min audio-video feature distance over +-vshift offsets; LSE-C = median-minus-min
confidence. Default crop matches LatentSync's geometry (scale 1.4, biased toward
mouth, eval/syncnet_detect.py:180,191-204), adopted 2026-08-10 after a paired
showed it improves LSE-D by 0.507 paired (t -2.82) on 21/24 pilot sides. Use
--crop none to recover the legacy whole-frame resize.

Run in the `video` conda env (needs syncnet weights + python_speech_features):
  python scripts/lse_video.py --video refined_j.mp4 --audio track_j.wav \
      [--syncnet-dir <syncnet_python checkout>] [--device cuda]
"""
import argparse
import os
import subprocess
import sys
import tempfile
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.dirname(HERE)
sys.path.insert(0, CODE_ROOT)

from chat.eval import TARGETS, meets_paper  # noqa: E402

MIN_SEG = 1.2   # seconds; SyncNet needs a healthy window of frames
PAD = 0.15      # seconds of context kept around each speech region


def speech_segments(wav_path, thresh_frac=0.15):
    """RMS-gated speech regions of a wav -> [(start_s, end_s), ...].

    Threshold = thresh_frac * 95th-percentile RMS (robust to absolute level);
    adjacent regions closer than 2*PAD merge; regions shorter than MIN_SEG drop.
    """
    import soundfile as sf

    w, sr = sf.read(wav_path)
    if w.ndim > 1:
        w = w.mean(axis=1)
    hop = sr // 20  # 50 ms
    rms = np.sqrt(np.convolve(w ** 2, np.ones(hop) / hop, mode="same"))
    thr = thresh_frac * np.percentile(rms, 95)
    on = rms > thr
    segs, start = [], None
    for i, v in enumerate(on):
        if v and start is None:
            start = i
        elif not v and start is not None:
            segs.append((start / sr, i / sr))
            start = None
    if start is not None:
        segs.append((start / sr, len(on) / sr))
    # pad + merge + min-length
    segs = [(max(0.0, a - PAD), b + PAD) for a, b in segs]
    merged = []
    for a, b in segs:
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return [(a, b) for a, b in merged if b - a >= MIN_SEG]


def length_matched_segments(wav_path, thresh_frac=0.15):
    """Segments matching the speech gate's COUNT and DURATIONS, spread over the whole clip.

    The control for `speech_segments`. A plain gated-vs-ungated comparison confounds two
    things: the gate removes silence, and it also shortens the scored material. The
    2026-08-08 duration control measured length alone moving LSE-C by 1.560, comparable to
    the gating effect being attributed. This reproduces the gate's segment count and each
    segment's duration, but places them at evenly spaced positions over the full timeline,
    so they cover speech and silence in whatever proportion the clip actually has.

    Returns [] when the gate would return [], so callers fall back identically.
    """
    import soundfile as sf

    speech = speech_segments(wav_path, thresh_frac=thresh_frac)
    if not speech:
        return []
    w, sr = sf.read(wav_path)
    if w.ndim > 1:
        w = w.mean(axis=1)
    total = len(w) / sr
    durations = [b - a for a, b in speech]
    n = len(durations)
    # Evenly spaced segment CENTRES over the clip, then clamp each to the clip bounds.
    # Anchoring on centres keeps the spread symmetric; clamping preserves each duration
    # exactly, which is the quantity being controlled.
    out = []
    for k, dur in enumerate(durations):
        dur = min(dur, total)
        centre = total * (k + 0.5) / n
        a = centre - dur / 2
        a = max(0.0, min(a, total - dur))
        out.append((a, a + dur))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="generated talking-face video")
    ap.add_argument("--audio", default=None, help="wav to mux (if the video has no audio track)")
    ap.add_argument("--syncnet-dir",
                    default=os.environ.get("CHAT_SYNCNET_DIR", "syncnet_python"),
                    help="checkout of the SyncNet reference implementation "
                         "(https://github.com/joonson/syncnet_python) holding "
                         "syncnet_v2.model; overridable with $CHAT_SYNCNET_DIR")
    ap.add_argument("--vshift", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--crop", choices=["none", "s3fd"], default="s3fd",
                    help="s3fd = detect the face (S3FD, median box over sampled frames, "
                         "static-camera assumption) and crop a LatentSync-style face square "
                         "before the 224 resize — the standard LSE protocol; 'none' keeps "
                         "the legacy whole-frame resize. Default is s3fd, matching the crop "
                         "geometry of LatentSync (eval/syncnet_detect.py:180,191-204), which "
                         "the paper cites for LSE. Adopted 2026-08-10: LSE-D "
                         "improved 0.507 paired, t -2.82 on 21/24 pilot sides.")
    ap.add_argument("--crop-scale", type=float, default=1.4,
                    help="side of the face square as a multiple of the detection box's longer "
                         "edge (--crop s3fd only). Default 1.4 matches LatentSync, which cuts "
                         "2.8*s where s is half the longer edge (eval/syncnet_detect.py:180). "
                         "The former default 2.2 was a no-op at 512x512 (clamped to full frame).")
    ap.add_argument("--crop-vbias", type=float, default=0.1429,
                    help="shift the crop square down by this fraction of its side (--crop s3fd "
                         "only), so the mouth sits nearer the centre. Default 0.1429 reproduces "
                         "LatentSync's framing (my-bs to my+1.8*bs on a 2.8*bs side, "
                         "eval/syncnet_detect.py:191-204). The crop-alignment study showed the vertical bias "
                         "contributes -1.12 of the -0.51 paired LSE-D improvement; scale alone "
                         "(+0.62) is worse.")
    ap.add_argument("--paper-lsed", action="store_true",
                    help="additionally report an alternative LSE-D reading, the mean L2 "
                         "distance between consecutive SyncNet lip embeddings, in raw, "
                         "unit-normalised and cosine forms (diagnostic; the headline LSE-D "
                         "above stays the standard audio-video distance)")
    ap.add_argument("--speech-only", action="store_true",
                    help="score only speaking segments (RMS-gated from --audio). Dyadic "
                         "dialogue tracks are ~50%% silence by turn-taking design; SyncNet "
                         "LSE is defined on speech, and silent stretches (idle face, no "
                         "audio) dilute the confidence statistic")
    ap.add_argument("--length-matched", action="store_true",
                    help="control arm for --speech-only: score the SAME TOTAL DURATION the "
                         "speech gate would keep, but drawn from the whole timeline instead of "
                         "from speech regions, in segments of the same length and count. "
                         "Needs --audio. Gating and duration are confounded in a plain "
                         "gated-vs-ungated comparison, because the ungated arm is also longer "
                         "and the 2026-08-08 duration control measured length alone moving "
                         "LSE-C by 1.560. Holding duration fixed isolates what the gate itself "
                         "is worth. Mutually exclusive with --speech-only")
    args = ap.parse_args()
    if args.length_matched and args.speech_only:
        raise SystemExit("--length-matched is the control FOR --speech-only; pass one, not both")

    sys.path.insert(0, args.syncnet_dir)
    import torch
    from SyncNetInstance import SyncNetInstance  # noqa: E402

    dev = args.device if torch.cuda.is_available() else "cpu"

    vf = "scale=224:224,fps=25"
    if args.crop == "s3fd":
        import cv2
        from detectors import S3FD  # noqa: E402  (vendored syncnet_python)

        # S3FD loads its weights from a cwd-relative constant; run its constructor
        # from the syncnet dir (weights symlinked at detectors/s3fd/weights/).
        cwd = os.getcwd()
        os.chdir(args.syncnet_dir)
        try:
            det = S3FD(device=dev)
        finally:
            os.chdir(cwd)
        cap = cv2.VideoCapture(args.video)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        boxes = []
        for fi in np.linspace(0, max(n - 1, 0), 5).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok, frame = cap.read()
            if not ok:
                continue
            dets = det.detect_faces(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                                    conf_th=0.9, scales=[0.25, 0.5])
            if len(dets):
                boxes.append(max(dets, key=lambda d: (d[2] - d[0]) * (d[3] - d[1]))[:4])
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if boxes:
            x1, y1, x2, y2 = np.median(np.stack(boxes), axis=0)
            side = args.crop_scale * max(x2 - x1, y2 - y1)
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            cy += args.crop_vbias * side
            side = min(side, W, H)
            x0 = int(np.clip(cx - side / 2, 0, max(W - side, 0)))
            y0 = int(np.clip(cy - side / 2, 0, max(H - side, 0)))
            vf = f"crop={int(side)}:{int(side)}:{x0}:{y0},scale=224:224,fps=25"
            print(f"s3fd crop: box=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}) "
                  f"-> square {int(side)}px at ({x0},{y0}) "
                  f"[scale={args.crop_scale} vbias={args.crop_vbias}]")
        else:
            print("s3fd crop: NO face detected, falling back to whole-frame resize")
            print("  NOTE: this video is now scored under a different crop protocol than the")
            print("        others. Aggregating it with cropped videos mixes two protocols.")

    # dir=None puts the scratch clip under the system temp location, which honours
    # $TMPDIR. Set that when the default filesystem is small or quota-limited.
    with tempfile.TemporaryDirectory() as tmp:
        # prepare a 224x224, 25fps, audio-bearing clip for SyncNet
        prepped = os.path.join(tmp, "prepped.mp4")
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", args.video]
        if args.audio:
            cmd += ["-i", args.audio]
        cmd += ["-vf", vf, "-c:v", "libx264", "-pix_fmt", "yuv420p"]
        if args.audio:
            cmd += ["-map", "0:v:0", "-map", "1:a:0", "-c:a", "aac", "-shortest"]
        cmd += [prepped]
        subprocess.run(cmd, check=True)

        s = SyncNetInstance(device=dev)
        s.loadParameters(os.path.join(args.syncnet_dir, "data", "syncnet_v2.model"))

        # Diagnostic, kept for provenance. The paper's MAIN TEXT defines LSE-D only by
        # citation (li2024latentsync), i.e. the standard audio-video feature distance at the
        # best offset, which is what this script computes as the headline LSE-D. An
        # alternative reading of the name describes LSE-D as "the average L2 distance
        # between consecutive lip feature vectors" — a temporal-consistency measure, a
        # different quantity. This flag captures SyncNet's visual embeddings and reports
        # every reasonable reading of that wording so the question can be settled by scale
        # rather
        # than by preference. It was: only the raw variant lands anywhere near the paper's
        # reported LSE-D values, and it still sits about 3.7 SEM below their lowest, so the
        # standard definition is the one the paper's numbers are consistent with.
        lip_feats = []
        if args.paper_lsed:
            # The model attribute is named __S__, with trailing underscores, so Python does
            # NOT name-mangle it; look it up by feature rather than by any assumed spelling.
            model = next((m for m in (getattr(s, a, None)
                                      for a in ("__S__", "_SyncNetInstance__S__", "S", "net"))
                          if m is not None and hasattr(m, "forward_lip")), None)
            if model is None:
                raise SystemExit("--paper-lsed: cannot locate the SyncNet module on the instance")
            _orig_forward_lip = model.forward_lip

            def _capturing_forward_lip(x, _f=_orig_forward_lip):
                out = _f(x)
                lip_feats.append(out.detach().float().cpu())
                return out

            model.forward_lip = _capturing_forward_lip

        segments = [None]  # None = whole video
        if args.speech_only:
            if not args.audio:
                raise SystemExit("--speech-only needs --audio (RMS gating reads the wav)")
            segments = speech_segments(args.audio)
            print(f"speech segments (>= {MIN_SEG:.2f}s): "
                  f"{[f'{a:.2f}-{b:.2f}' for a, b in segments]}")
            if not segments:
                print("no speech segments found; falling back to whole video")
                segments = [None]
        elif args.length_matched:
            if not args.audio:
                raise SystemExit("--length-matched needs --audio (it mirrors the RMS gate)")
            segments = length_matched_segments(args.audio)
            print(f"length-matched segments (n and durations mirror the speech gate): "
                  f"{[f'{a:.2f}-{b:.2f}' for a, b in segments]}")
            print(f"  scored duration = {sum(b - a for a, b in segments):.2f}s "
                  f"over {len(segments)} segments")
            if not segments:
                print("gate would find no speech; falling back to whole video")
                segments = [None]

        rows, offsets = [], []
        for k, seg in enumerate(segments):
            src = prepped
            if seg is not None:
                src = os.path.join(tmp, f"seg{k}.mp4")
                subprocess.run(["ffmpeg", "-y", "-loglevel", "error",
                                "-ss", f"{seg[0]:.3f}", "-t", f"{seg[1] - seg[0]:.3f}",
                                "-i", prepped, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                                "-c:a", "aac", src], check=True)
            opt = types.SimpleNamespace(tmp_dir=os.path.join(tmp, f"work{k}"),
                                        reference="lse", vshift=args.vshift,
                                        batch_size=args.batch_size)
            os.makedirs(opt.tmp_dir, exist_ok=True)
            try:
                offset, conf, dists = s.evaluate(opt, videofile=src)
            except Exception as exc:  # noqa: BLE001 — one bad segment must not kill the video
                print(f"  segment {k} failed ({exc}); skipped")
                continue
            rows.append(np.asarray(dists))
            offsets.append(int(np.asarray(offset)))
        if not rows:
            raise SystemExit("no scorable segments")

    dists = np.concatenate(rows, axis=0)      # (n_speech_frames, 2*vshift+1)
    mdist = dists.mean(axis=0)                # mean distance per offset
    lse_d = float(mdist.min())                # LSE-D
    lse_c = float(np.median(mdist) - mdist.min())  # LSE-C (== returned conf)
    print(f"video={os.path.basename(args.video)}  AV offset(s)={offsets}")
    print(f"LSE-C = {lse_c:.3f}  (paper target {TARGETS['LSE-C'][0]}, higher better) "
          f"meets={meets_paper('LSE-C', lse_c)}")
    print(f"LSE-D = {lse_d:.3f}  (paper target {TARGETS['LSE-D'][0]}, lower better) "
          f"meets={meets_paper('LSE-D', lse_d)}")
    print("NOTE: the paper targets were measured on 1000 dialogues under the paper's "
          "protocol. Do not place this number beside a paper number; see DOCS.md.")

    if args.paper_lsed:
        if not lip_feats:
            print("PAPER-LSED: no lip features captured")
        else:
            import torch as _t
            f = _t.cat(lip_feats, 0)
            f = f.reshape(f.shape[0], -1).numpy()
            if f.shape[0] < 2:
                print("PAPER-LSED: fewer than 2 frames, cannot difference")
            else:
                d_raw = np.linalg.norm(np.diff(f, axis=0), axis=1)
                fn = f / np.clip(np.linalg.norm(f, axis=1, keepdims=True), 1e-8, None)
                d_unit = np.linalg.norm(np.diff(fn, axis=0), axis=1)
                d_cos = 1.0 - (fn[:-1] * fn[1:]).sum(axis=1)
                print(f"PAPER-LSED frames={f.shape[0]} dim={f.shape[1]}")
                print(f"PAPER-LSED-RAW = {d_raw.mean():.4f}")
                print(f"PAPER-LSED-UNIT = {d_unit.mean():.4f}")
                print(f"PAPER-LSED-COS = {d_cos.mean():.4f}")
    print("LSE_DONE")


if __name__ == "__main__":
    main()
