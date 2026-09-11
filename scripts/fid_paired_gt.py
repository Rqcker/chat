"""FID under the field's paired ground-truth protocol, alongside our existing one.

Why this exists. Our headline FID scores 186 generated sides against a frozen reference
of 2,493 *disjoint* HDTF clips at 4 sparse frames each. That configuration showed that this
construction, not generation quality, dominates the number: real HDTF video drawn to the
same shape (186 identities, 4 frames each) scores 31.24 against the generated set's 31.53,
while the same real footage from 582 identities scores 7.58.

The published protocol removes that term by pairing. SadTalker, EDTalk and ARIG each report
a ground-truth row at FID 0.000, which is only possible when the reference is the ground
truth of the very clips being evaluated, so identity coverage, background and framing cancel
by construction. SadTalker is the one fully specified protocol in the literature: "the 346
videos' first 8-second video (around 70k frames in total)", FOMM face crop.

This script builds that reference for our eval set. Every identity in eval_set2 came from a
real HDTF clip, and `x1_faces.py` logged which one ("HDTF identities: <i>.mp4, <j>.mp4").
So the paired ground truth is recoverable exactly: for each generated side, take its own
source HDTF clip, put it through the same reframe and the same Inception extractor, and
score against that.

Three subcommands:

  manifest  recover the side -> source-clip mapping from the x1 array logs, and verify it
            against the identity portraits actually on disk before trusting it.
  build     featurise the paired ground-truth clips into a reference .npz.
  score     score generated video against that reference.

The result is a SECOND protocol reported alongside the existing one. It does not restate
the existing 31.53, which remains a valid in-protocol measurement.

Run in the `video` conda env:

  python scripts/fid_paired_gt.py manifest --eval-dir eval_set2 --logs-dir eval_set2/logs \
      --clips-dir hdtf/clips_extracted/clips --out paired_gt_manifest.json
  python scripts/fid_paired_gt.py build --manifest paired_gt_manifest.json \
      --out fid_ref_heldout.npz
  python scripts/fid_paired_gt.py score --ref fid_ref_heldout.npz \
      --eval-dir eval_set2 --stem video_refined

Two references are possible and they are not interchangeable. The held-out reference,
built from footage of the same speakers that the pipeline never saw, is the one every
reported identity-matched number uses: real floor 7.72, `video` 18.13, `video_refined`
20.10. A strictly paired reference, built from the source clips the identity frames were
cut from, reads 19.97 on `video_refined`; it exists only to size the circularity term at
0.13 and is not the default. Scoring against the wrong one silently breaks comparability,
so keep the two `.npz` files under distinct names.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.dirname(HERE)
sys.path.insert(0, CODE_ROOT)
sys.path.insert(0, HERE)

from chat.eval import frechet_distance  # noqa: E402
from chat.eval.extractors import InceptionExtractor  # noqa: E402
from reframe_utils import apply_reframe, measure_reframe  # noqa: E402

# x1_faces.py prints exactly this before saving the two identity portraits.
IDENT_RE = re.compile(r"HDTF identities:\s*(\S+\.mp4),\s*(\S+\.mp4)")
# Two ways to learn which clip a log belongs to. The array driver prints CID=, but the
# per-clip chain logs do not, so fall back to the identity save path, which every log
# carries because x1_faces.py prints it unconditionally. Using only CID= recovers 15 of
# 93 clips; the save path recovers all of them.
CID_RE = re.compile(r"CID=(\S+)")
SAVED_RE = re.compile(r"saved\s+\S*?/(clip_[A-Za-z0-9]+)/identity_[ij]\.png")


def clip_id_from_log(text: str) -> str | None:
    """Which eval clip this log belongs to, preferring the explicit marker."""
    m = CID_RE.search(text)
    if m:
        return m.group(1)
    ids = set(SAVED_RE.findall(text))
    # A log covering two clips cannot be attributed safely, so decline rather than guess.
    return ids.pop() if len(ids) == 1 else None


# --------------------------------------------------------------------------- manifest


def recover_manifest(logs_dir: str, eval_dir: str, clips_dir: str) -> dict:
    """Map each eval side to the HDTF clip its identity portrait was cut from.

    A log is only trusted when it carries both a CID and an identity line, the clip
    directory still exists, and both source clips are still on disk. Anything else is
    reported as a gap rather than guessed at, because a wrong pairing would silently
    turn this protocol back into a disjoint one.
    """
    pairs, gaps = {}, []
    for log in sorted(glob.glob(os.path.join(logs_dir, "*.log"))):
        text = open(log, encoding="utf-8", errors="replace").read()
        id_m = IDENT_RE.search(text)
        if not id_m:
            continue
        cid = clip_id_from_log(text)
        if cid is None:
            gaps.append({"why": "identity line but no attributable clip id",
                         "log": os.path.basename(log)})
            continue
        clip_dir = os.path.join(eval_dir, cid)
        if not os.path.isdir(clip_dir):
            gaps.append({"cid": cid, "why": "no clip dir", "log": os.path.basename(log)})
            continue
        entry = {}
        for side, name in (("i", id_m.group(1)), ("j", id_m.group(2))):
            src = os.path.join(clips_dir, name)
            if not os.path.isfile(src):
                gaps.append({"cid": cid, "side": side, "why": f"source clip missing: {name}"})
                continue
            entry[side] = name
        if len(entry) == 2:
            # A later re-run of the same clip would append a second log; keep the first
            # and flag any disagreement rather than letting order decide silently.
            if cid in pairs and pairs[cid] != entry:
                gaps.append({"cid": cid, "why": "conflicting identity lines across logs"})
            pairs.setdefault(cid, entry)
    return {"pairs": pairs, "gaps": gaps}


def verify_manifest(man: dict, eval_dir: str, clips_dir: str, insightface_root: str,
                    n_check: int) -> list:
    """Confirm the recovered mapping by re-deriving identity portraits from the sources.

    `x1_faces.py` saved the raw source frame as `ref_{side}.png` next to the portrait it
    made from it. If the mapping is right, the middle frame of the named source clip is
    that same frame, so a direct pixel comparison settles it. This is the check that
    stops a mislabelled pairing from quietly producing a flattering FID.
    """
    import av
    from PIL import Image

    checks = []
    for cid in sorted(man["pairs"])[:n_check]:
        for side, name in man["pairs"][cid].items():
            ref_png = os.path.join(eval_dir, cid, f"ref_{side}.png")
            if not os.path.isfile(ref_png):
                checks.append({"cid": cid, "side": side, "verdict": "no ref png"})
                continue
            saved = np.array(Image.open(ref_png).convert("RGB"))
            with av.open(os.path.join(clips_dir, name)) as c:
                frames = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
            mid = frames[len(frames) // 2]
            if mid.shape != saved.shape:
                checks.append({"cid": cid, "side": side, "verdict": "shape mismatch",
                               "saved": list(saved.shape), "source": list(mid.shape)})
                continue
            mae = float(np.abs(mid.astype(np.float32) - saved.astype(np.float32)).mean())
            checks.append({"cid": cid, "side": side, "clip": name,
                           "mae": mae, "verdict": "match" if mae < 1.0 else "MISMATCH"})
    return checks


def cmd_manifest(args) -> None:
    man = recover_manifest(args.logs_dir, args.eval_dir, args.clips_dir)
    n_sides = 2 * len(man["pairs"])
    print(f"recovered {len(man['pairs'])} clips ({n_sides} sides) from {args.logs_dir}")
    if man["gaps"]:
        print(f"{len(man['gaps'])} gap(s):")
        for g in man["gaps"][:10]:
            print(f"  {g}")

    if args.verify:
        checks = verify_manifest(man, args.eval_dir, args.clips_dir,
                                 args.insightface_root, args.n_verify)
        bad = [c for c in checks if c["verdict"] != "match"]
        ok = len(checks) - len(bad)
        print(f"\nverification: {ok}/{len(checks)} portraits trace back to the named clip")
        for c in bad[:10]:
            print(f"  {c}")
        man["verification"] = checks
        if bad:
            print("\nWARNING: the mapping did not verify cleanly. Do NOT build a reference "
                  "from it; a wrong pairing makes this protocol silently disjoint again.")

    man["eval_dir"] = os.path.abspath(args.eval_dir)
    man["clips_dir"] = os.path.abspath(args.clips_dir)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(man, f, indent=2)
    print(f"wrote {args.out}")


# ------------------------------------------------------------------------------ build


def clip_frames(path: str, k: int, insightface_root: str, reframe: bool):
    """Up to k frames of one clip, under the same reframe the eval videos carry.

    Frames are taken uniformly so a short ground-truth clip and a long one contribute
    comparably, matching how the existing frozen reference samples.
    """
    import av

    with av.open(path) as container:
        frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
    if not frames:
        return None
    if k and len(frames) > k:
        idx = np.linspace(0, len(frames) - 1, k).round().astype(int)
        frames = [frames[i] for i in idx]
    if reframe:
        params = measure_reframe(frames[0], insightface_root)
        if params is None:
            return None
        frames = [apply_reframe(f, params) for f in frames]
    arr = np.stack(frames).astype(np.float32) / 255.0
    return arr.transpose(0, 3, 1, 2)


def speaker_key(clip_name: str) -> str:
    """HDTF clip names are <speaker>_<take>_<start>_<end>.mp4; strip to the speaker."""
    return re.sub(r"_\d+_\d+_\d+\.mp4$", "", clip_name)


def heldout_sources(man: dict, clips_dir: str, per_side: int, seed: int):
    """For each eval side, other clips of the SAME speaker, excluding every source clip.

    The strictly paired reference has a circularity: each identity portrait is literally a
    frame of its own source clip, so scoring against that clip flatters the generator in a
    way the number cannot separate from generation quality. This variant keeps what the
    pairing was for, cancelling identity coverage, while removing that circularity: the
    reference is the same 110 speakers in footage the pipeline never saw.
    """
    import glob as _glob

    bank = [os.path.basename(p) for p in _glob.glob(os.path.join(clips_dir, "*.mp4"))]
    by_speaker: dict[str, list[str]] = {}
    for name in sorted(bank):
        by_speaker.setdefault(speaker_key(name), []).append(name)

    used = {n for c in man["pairs"].values() for n in c.values()}
    rng = np.random.default_rng(seed)
    out, short = [], []
    for cid in sorted(man["pairs"]):
        for side, name in sorted(man["pairs"][cid].items()):
            pool = [c for c in by_speaker.get(speaker_key(name), []) if c not in used]
            if not pool:
                short.append({"cid": cid, "side": side, "speaker": speaker_key(name)})
                continue
            k = min(per_side, len(pool))
            for pick in rng.choice(pool, size=k, replace=False):
                out.append((cid, side, str(pick)))
    return out, short


def cmd_build(args) -> None:
    man = json.load(open(args.manifest, encoding="utf-8"))
    if man.get("verification"):
        bad = [c for c in man["verification"] if c["verdict"] != "match"]
        if bad and not args.allow_unverified:
            raise SystemExit(
                f"manifest has {len(bad)} failed verification(s); refusing to build. "
                "Re-run `manifest --verify`, or pass --allow-unverified deliberately.")

    clips_dir = args.clips_dir or man["clips_dir"]
    if args.mode == "paired":
        # One reference entry per generated SIDE, so the paired set has exactly the identity
        # composition of the evaluated set. A clip appearing twice contributes twice, which
        # is correct: two sides means two evaluated identities.
        sources = [(cid, side, name)
                   for cid in sorted(man["pairs"])
                   for side, name in sorted(man["pairs"][cid].items())]
        short = []
        print(f"{len(sources)} paired ground-truth sides; "
              f"{len({n for _, _, n in sources})} distinct source clips")
    else:
        sources, short = heldout_sources(man, clips_dir, args.per_side, args.seed)
        print(f"{len(sources)} held-out entries over "
              f"{len({n for _, _, n in sources})} distinct clips, "
              f"{args.per_side} per side, same speakers, no source clip reused")
        if short:
            print(f"WARNING: {len(short)} side(s) had no held-out clip for their speaker")

    ext = InceptionExtractor(device=args.device, batch_size=args.batch_size)
    feats, used, failed = [], [], []
    for cid, side, name in sources:
        try:
            arr = clip_frames(os.path.join(clips_dir, name), args.frames_per_clip,
                              args.insightface_root, not args.no_reframe)
        except Exception as exc:                       # a corrupt clip must not end the build
            failed.append({"cid": cid, "side": side, "clip": name, "error": str(exc)})
            continue
        if arr is None:
            failed.append({"cid": cid, "side": side, "clip": name, "error": "no usable frames"})
            continue
        feats.append(ext.features(arr))
        used.append({"cid": cid, "side": side, "clip": name, "frames": int(arr.shape[0])})
        if len(used) % 20 == 0:
            print(f"  {len(used)}/{len(sources)} sides ({len(failed)} failed)", flush=True)

    if not feats:
        raise SystemExit("no usable paired ground-truth frames")
    allf = np.concatenate(feats, axis=0)
    mu, cov = allf.mean(axis=0), np.cov(allf, rowvar=False)
    np.savez(args.out, mu=mu, cov=cov, n_frames=allf.shape[0])
    manifest_out = os.path.splitext(args.out)[0] + "_manifest.json"
    protocol = ("paired ground truth (per-side source HDTF clip)" if args.mode == "paired"
                else "held-out same-speaker HDTF clips (no source clip reused)")
    with open(manifest_out, "w", encoding="utf-8") as f:
        json.dump({"protocol": protocol,
                   "mode": args.mode,
                   "source_manifest": os.path.abspath(args.manifest),
                   "clips_dir": clips_dir,
                   "frames_per_clip": args.frames_per_clip,
                   "per_side": args.per_side if args.mode == "heldout" else None,
                   "seed": args.seed,
                   "reframe": not args.no_reframe,
                   "n_entries": len(used), "n_frames": int(allf.shape[0]),
                   "sides_without_heldout": short,
                   "used": used, "failed": failed}, f, indent=2)
    print(f"\nentries={len(used)} frames={allf.shape[0]} failed={len(failed)}")
    print(f"wrote {args.out} and {manifest_out}")
    if args.mode == "paired":
        print("NOTE: strictly paired. Each identity portrait is a frame of its own source "
              "clip, so this reference is partly circular and will read optimistically. "
              "The held-out mode is the defensible number; report both.")
    if allf.shape[0] < 10000:
        print(f"NOTE: {allf.shape[0]} reference frames. Jayasumana et al. (CVPR 2024) put the "
              "threshold for a stable 2048-d covariance above 20,000, so read this number "
              "with the finite-sample bias table in metric_ledger.md.")


# ------------------------------------------------------------------------------ score


def cmd_control(args) -> None:
    """Score REAL held-out video of the same speakers against the same reference.

    Without this the protocol is uninterpretable. A low generated FID against an
    identity-matched reference could mean the generations are good, or merely that
    matching identities collapses the number regardless of quality. This control is the
    floor: real footage of the same speakers, disjoint from both the reference and the
    identity sources, put through the same reframe and extractor. The generated number is
    only meaningful as a distance above this floor.
    """
    man = json.load(open(args.manifest, encoding="utf-8"))
    clips_dir = args.clips_dir or man["clips_dir"]
    ref = np.load(args.ref)
    mu_r, cov_r = ref["mu"], ref["cov"]
    print(f"reference: {args.ref} ({int(ref['n_frames'])} frames)")

    ref_man_path = os.path.splitext(args.ref)[0] + "_manifest.json"
    in_ref = set()
    if os.path.isfile(ref_man_path):
        in_ref = {u["clip"] for u in json.load(open(ref_man_path, encoding="utf-8"))["used"]}
    used_src = {n for c in man["pairs"].values() for n in c.values()}
    exclude = in_ref | used_src
    print(f"excluding {len(exclude)} clips already used as reference or identity source")

    picks, short = heldout_sources(man, clips_dir, args.per_side, args.seed)
    picks = [(cid, side, name) for cid, side, name in picks if name not in exclude]
    if not picks:
        raise SystemExit("no held-out clips left after exclusion; lower --per-side on the build")
    print(f"{len(picks)} control clips over {len({n for _, _, n in picks})} distinct files")

    ext = InceptionExtractor(device=args.device, batch_size=args.batch_size)
    feats, n, failed = [], 0, 0
    for cid, side, name in picks:
        try:
            arr = clip_frames(os.path.join(clips_dir, name), args.frames_per_clip,
                              args.insightface_root, True)
        except Exception:                              # a corrupt clip must not end the run
            failed += 1
            continue
        if arr is None:
            failed += 1
            continue
        feats.append(ext.features(arr))
        n += int(arr.shape[0])
        if len(feats) % 40 == 0:
            print(f"  {len(feats)}/{len(picks)} clips ({failed} failed)", flush=True)

    allf = np.concatenate(feats, axis=0)
    fid = frechet_distance(mu_r, cov_r, allf.mean(axis=0), np.cov(allf, rowvar=False))
    print(f"\nclips={len(feats)} frames={n} failed={failed}")
    print(f"FID (REAL held-out video, same protocol) = {fid:.2f}")
    print("This is the floor. Read any generated number as its distance above this.")

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump({"protocol": "real held-out control", "ref": args.ref,
                       "n_clips": len(feats), "n_frames": n, "failed": failed,
                       "fid": fid}, f, indent=2)
        print(f"wrote {args.out_json}")


def cmd_heldout_dir(args) -> None:
    """Materialise the held-out same-speaker clips as a directory of symlinks.

    FVD takes directories of mp4s rather than a feature npz, so the same held-out draw that
    backs the FID protocol has to exist on disk to be reusable there. Symlinks keep it free.
    """
    man = json.load(open(args.manifest, encoding="utf-8"))
    clips_dir = args.clips_dir or man["clips_dir"]
    picks, short = heldout_sources(man, clips_dir, args.per_side, args.seed)
    names = sorted({n for _, _, n in picks})
    os.makedirs(args.out_dir, exist_ok=True)
    made = 0
    for name in names:
        dst = os.path.join(args.out_dir, name)
        if os.path.lexists(dst):
            os.unlink(dst)
        os.symlink(os.path.join(clips_dir, name), dst)
        made += 1
    print(f"linked {made} held-out clips into {args.out_dir}")
    if short:
        print(f"WARNING: {len(short)} side(s) had no held-out clip for their speaker")


def collect_mp4s(paths):
    """Every .mp4 under the given files or directories, deduplicated and sorted.

    Archived generated sets use four different layouts (`clip_X/video_i.mp4`,
    `clip_01__i.mp4`, `gen/clip_01_i/*.mp4`, flat), so scoring them needs a collector that
    does not assume one.
    """
    out = []
    for p in paths:
        if os.path.isdir(p):
            out.extend(os.path.join(r, f)
                       for r, _, fs in os.walk(p) for f in fs if f.endswith(".mp4"))
        elif p.endswith(".mp4"):
            out.append(p)
    return sorted(set(out))


def cmd_score(args) -> None:
    ref = np.load(args.ref)
    mu_r, cov_r = ref["mu"], ref["cov"]
    print(f"reference: {args.ref} ({int(ref['n_frames'])} frames, identity-matched)")

    if args.videos:
        vids = collect_mp4s(args.videos)
        if not vids:
            raise SystemExit(f"no .mp4 found under {args.videos}")
    else:
        vids = []
        for cid in sorted(os.listdir(args.eval_dir)):
            d = os.path.join(args.eval_dir, cid)
            if not os.path.isdir(d):
                continue
            for side in ("i", "j"):
                p = os.path.join(d, f"{args.stem}_{side}.mp4")
                if os.path.isfile(p):
                    vids.append(p)
        if not vids:
            raise SystemExit(f"no {args.stem}_[ij].mp4 under {args.eval_dir}")
    print(f"{len(vids)} generated sides")

    import av

    ext = InceptionExtractor(device=args.device, batch_size=args.batch_size)
    feats, n = [], 0
    for p in vids:
        with av.open(p) as c:
            frames = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
        if not frames:
            continue
        if args.frames_per_video and len(frames) > args.frames_per_video:
            idx = np.linspace(0, len(frames) - 1, args.frames_per_video).round().astype(int)
            frames = [frames[i] for i in idx]
        arr = np.stack(frames).astype(np.float32).transpose(0, 3, 1, 2) / 255.0
        feats.append(ext.features(arr))
        n += len(frames)
        if len(feats) % 20 == 0:
            print(f"  {len(feats)}/{len(vids)} sides", flush=True)

    allf = np.concatenate(feats, axis=0)
    fid = frechet_distance(mu_r, cov_r, allf.mean(axis=0), np.cov(allf, rowvar=False))
    print(f"\nsides={len(feats)} frames={n}")
    print(f"FID (identity-matched reference) = {fid:.2f}")
    print("This is a SECOND protocol. It does not restate the disjoint-reference 31.53,")
    print("which remains a valid in-protocol measurement. Report both, labelled.")

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump({"protocol": "identity-matched reference", "ref": args.ref,
                       "stem": None if args.videos else args.stem,
                       "videos": args.videos or None,
                       "label": args.label,
                       "n_sides": len(feats), "n_frames": n,
                       "fid": fid}, f, indent=2)
        print(f"wrote {args.out_json}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("manifest", help="recover and verify side -> source-clip mapping")
    m.add_argument("--eval-dir", required=True)
    m.add_argument("--logs-dir", required=True, help="x1 array logs with 'HDTF identities:' lines")
    m.add_argument("--clips-dir", required=True, help="HDTF clip bank the identities came from")
    m.add_argument("--out", required=True)
    m.add_argument("--verify", action="store_true",
                   help="re-derive portraits from the named clips and compare pixels")
    m.add_argument("--n-verify", type=int, default=10)
    m.add_argument("--insightface-root", default=os.path.expanduser("~/.insightface"))
    m.set_defaults(func=cmd_manifest)

    b = sub.add_parser("build", help="featurise the paired ground truth into a reference npz")
    b.add_argument("--manifest", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--mode", choices=("paired", "heldout"), default="heldout",
                   help="paired: each side's own source clip, matching the SadTalker-style "
                        "ground-truth reference but partly circular here because the identity "
                        "portrait is a frame of that clip. heldout (default): other clips of "
                        "the same speakers, which cancels identity coverage without the "
                        "circularity")
    b.add_argument("--per-side", type=int, default=2,
                   help="heldout mode: how many held-out clips per evaluated side")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--clips-dir", default=None, help="override the manifest's clips_dir")
    b.add_argument("--frames-per-clip", type=int, default=0, help="0 = every frame")
    b.add_argument("--no-reframe", action="store_true",
                   help="skip the identity reframe; only for a framing-sensitivity control")
    b.add_argument("--allow-unverified", action="store_true")
    b.add_argument("--device", default=None)
    b.add_argument("--batch-size", type=int, default=64)
    b.add_argument("--insightface-root", default=os.path.expanduser("~/.insightface"))
    b.set_defaults(func=cmd_build)

    s = sub.add_parser("score", help="score generated video against the paired reference")
    s.add_argument("--ref", required=True)
    s.add_argument("--eval-dir", default=None,
                   help="eval_set2-style tree; scored via --stem. Ignored when --videos is given")
    s.add_argument("--videos", nargs="+", default=None,
                   help="explicit mp4s or directories, walked recursively. Use for archived "
                        "sets whose layout is not eval_set2's")
    s.add_argument("--label", default=None, help="free-text tag recorded in the out-json")
    s.add_argument("--stem", default="video_refined")
    s.add_argument("--frames-per-video", type=int, default=0, help="0 = every frame")
    s.add_argument("--device", default=None)
    s.add_argument("--batch-size", type=int, default=64)
    s.add_argument("--out-json", default=None)
    s.set_defaults(func=cmd_score)

    c = sub.add_parser("control",
                       help="score REAL held-out video the same way: the interpretive floor")
    c.add_argument("--ref", required=True)
    c.add_argument("--manifest", required=True, help="the side -> source-clip manifest")
    c.add_argument("--clips-dir", default=None)
    c.add_argument("--per-side", type=int, default=2)
    c.add_argument("--seed", type=int, default=101,
                   help="different from the build seed so the control draws other clips")
    c.add_argument("--frames-per-clip", type=int, default=0)
    c.add_argument("--device", default=None)
    c.add_argument("--batch-size", type=int, default=64)
    c.add_argument("--insightface-root", default=os.path.expanduser("~/.insightface"))
    c.add_argument("--out-json", default=None)
    c.set_defaults(func=cmd_control)

    h = sub.add_parser("heldout-dir",
                       help="symlink the held-out clips into a directory, for FVD reuse")
    h.add_argument("--manifest", required=True)
    h.add_argument("--out-dir", required=True)
    h.add_argument("--clips-dir", default=None)
    h.add_argument("--per-side", type=int, default=2)
    h.add_argument("--seed", type=int, default=0)
    h.set_defaults(func=cmd_heldout_dir)

    args = ap.parse_args()
    if args.cmd == "score" and not args.videos and not args.eval_dir:
        ap.error("score needs --eval-dir or --videos")
    args.func(args)


if __name__ == "__main__":
    main()
