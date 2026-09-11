"""X1 step 2 (tts env): plan.json -> per-unit synthesis -> timed dual tracks.

Synthesises every sentence / interactive word with XTTS-v2 by default (voice
cloned from a per-speaker reference wav; emotion rate maps to XTTS speed),
measures real durations, assigns the timeline (chat.dadg.iar.timeline), and lays
out one 16 kHz track per speaker.

Usage:
  python scripts/x1_tts.py --plan x1_out/plan.json --out x1_out \
      --voice-i <ref_i.wav> --voice-j <ref_j.wav>

Optional opt-in (default OFF, behaviour unchanged unless both are given): clone
the voice from an emotion-matched reference clip instead of the fixed per-speaker
--voice-i/--voice-j.
  --emo-ref-dir <dir>   dir with ref_neutral.wav/ref_happy.wav/ref_angry.wav/ref_sad.wav
  --emo-map <path>      JSON mapping emotion category string -> neu/hap/ang/sad

Optional beyond-paper routing (default OFF): route target hap/ang/sad units to
F5-TTS-Emotional-CFG via a subprocess in the f5tts env, while keeping neutral and
unmapped units on the normal XTTS path. This is a mixture-of-experts choice keyed
on the target IAR emotion, not classifier output.
  --emo-route
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.dirname(HERE)
PROJECT_ROOT = os.path.dirname(CODE_ROOT)
sys.path.insert(0, CODE_ROOT)

# XTTS-v2 is distributed under the Coqui Public Model Licence, which its loader
# gates behind an interactive prompt. Accepting a third-party licence on the user's
# behalf is not this script's call, so the variable is read, never set here: export
# COQUI_TOS_AGREED=1 yourself once you have read the licence.
if not os.environ.get("COQUI_TOS_AGREED"):
    sys.exit(
        "COQUI_TOS_AGREED is not set.\n"
        "scripts/x1_tts.py synthesises with XTTS-v2, which is released under the Coqui "
        "Public Model Licence (https://huggingface.co/coqui/XTTS-v2/blob/main/LICENSE.txt). Read it, and if you accept it run:\n"
        "    export COQUI_TOS_AGREED=1\n"
        "This script does not accept the licence for you."
    )

from chat.dadg.iar.timeline import assign_timeline  # noqa: E402
from chat.schema import AudioUnit, Emotion, Prosody  # noqa: E402

SR = 16000  # Hallo2's expected input rate
# Emotion routing is an optional extra that needs a separate checkout of
# F5-TTS-Emotional-CFG and its own environment. Nothing on the default path touches
# any of this. Point CHAT_F5EMO_REPO (or --emo-route-repo) at the checkout; the
# default below is where `third_party/` would put it in a fresh clone.
DEFAULT_F5_ROOT = os.path.join(PROJECT_ROOT, "third_party", "F5-TTS-Emotional-CFG")
# The Emotion-CFG repo ships its own reference clips (data/0015_<emo>.wav, 2.1-2.8 s) and they
# are what the validated run used. Separately built reference clips of roughly twice that
# duration measurably reduce emotion accuracy on routed units, with almost all of the loss in
# happy. Keep this pointed at the repo's own data directory.
DEFAULT_F5_REF_DIR = os.path.join(DEFAULT_F5_ROOT, "data")
# Override with CHAT_F5EMO_PYTHON or --emo-route-python. F5 needs its own environment, so this
# deliberately does not fall back to the current interpreter.
DEFAULT_F5_PYTHON = os.path.expanduser(os.path.join("~", "envs", "f5tts", "bin", "python"))


def check_emo_route_assets(python_bin, repo, ckpt, ref_dir, batch_script):
    """Fail early, and with an actionable message, when routing assets are absent.

    The previous check listed the missing paths but not which flag supplies each one,
    and the default repo path pointed inside a working directory that a fresh clone
    does not have, so the common failure read as a bug rather than as a missing
    optional dependency.
    """
    missing = []
    if not os.path.isfile(python_bin):
        missing.append(f"  interpreter   {python_bin}   (--emo-route-python / CHAT_F5EMO_PYTHON)")
    if not os.path.isdir(repo):
        missing.append(f"  checkout      {repo}   (--emo-route-repo / CHAT_F5EMO_REPO)")
    if not os.path.isfile(ckpt):
        missing.append(f"  checkpoint    {ckpt}   (--emo-route-ckpt / CHAT_F5EMO_CKPT)")
    if not os.path.isdir(ref_dir):
        missing.append(f"  reference dir {ref_dir}   (--emo-route-ref-dir / CHAT_F5EMO_REF_DIR)")
    if not os.path.isfile(batch_script):
        missing.append(f"  batch driver  {batch_script}   (ships with this repository)")
    if not missing:
        return
    raise SystemExit(
        "--emo-route needs F5-TTS-Emotional-CFG, and these are missing:\n"
        + "\n".join(missing)
        + "\n\nEmotion routing is optional and off by default; drop --emo-route to run the\n"
          "default XTTS path. To enable it, clone RaduBolbo/F5-TTS-Emotional-CFG, create an\n"
          "environment for it, and point the flags or CHAT_F5EMO_* variables above at them."
    )


DEFAULT_EMO_CLASS_MAP = {
    "neu": "neu", "neutral": "neu", "calm": "neu",
    "hap": "hap", "happy": "hap", "happiness": "hap", "joy": "hap", "excited": "hap",
    "ang": "ang", "angry": "ang", "anger": "ang", "frustrated": "ang",
    "sad": "sad", "sadness": "sad", "unhappy": "sad",
}


def load_units(plan_path: str):
    with open(plan_path, encoding="utf-8") as f:
        data = json.load(f)

    def mk(u):
        emo = u.get("emotion")
        emotion = Emotion(emo["category"], Prosody(**emo["prosody"])) if emo else None
        return AudioUnit(turn=u["turn"], speaker=u["speaker"], kind=u["kind"],
                         text=u.get("text", ""), emotion=emotion)

    return data, [mk(u) for u in data["units_i"]], [mk(u) for u in data["units_j"]]


def load_emo_class_map(path: str | None) -> dict[str, str]:
    out = dict(DEFAULT_EMO_CLASS_MAP)
    if path:
        raw = json.load(open(path, encoding="utf-8"))
        out.update({str(k).lower().strip(): str(v).lower().strip() for k, v in raw.items()})
    bad = {k: v for k, v in out.items() if v not in {"neu", "hap", "ang", "sad"}}
    if bad:
        raise ValueError(f"emotion map values must be neu/hap/ang/sad, got {bad}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--voice-i", required=True)
    ap.add_argument("--voice-j", required=True)
    ap.add_argument("--iw-gain", type=float, default=0.6, help="back-channel loudness factor")
    ap.add_argument("--emo-ref-dir", default=None,
                    help="opt-in: dir with ref_{neutral,happy,angry,sad}.wav for "
                         "emotion-matched voice cloning; requires --emo-map")
    ap.add_argument("--emo-map", default=None,
                    help="JSON mapping emotion category -> neu/hap/ang/sad. Required by "
                         "--emo-ref-dir; optional for --emo-route")
    ap.add_argument("--emo-route", action="store_true",
                    help="beyond-paper opt-in: route target hap/ang/sad units to "
                         "F5-TTS-Emotional-CFG; default OFF")
    ap.add_argument("--emo-route-classes", default="hap,ang,sad",
                    help="comma-separated classes to route to F5 when --emo-route is set")
    ap.add_argument("--emo-route-python", default=os.environ.get("CHAT_F5EMO_PYTHON", DEFAULT_F5_PYTHON),
                    help="python executable from the f5tts env")
    ap.add_argument("--emo-route-repo", default=os.environ.get("CHAT_F5EMO_REPO", DEFAULT_F5_ROOT),
                    help="F5-TTS-Emotional-CFG checkout")
    ap.add_argument("--emo-route-ckpt", default=os.environ.get(
        "CHAT_F5EMO_CKPT", os.path.join(DEFAULT_F5_ROOT, "ckpts", "model_emo.pt")),
                    help="F5 emotion checkpoint")
    ap.add_argument("--emo-route-ref-dir", default=os.environ.get("CHAT_F5EMO_REF_DIR", DEFAULT_F5_REF_DIR),
                    help="F5 emotion reference dir")
    ap.add_argument("--emo-route-cfg-strength2", type=float, default=30.0)
    args = ap.parse_args()
    if args.emo_ref_dir and not args.emo_map:
        ap.error("--emo-ref-dir requires --emo-map")
    if args.emo_map and not (args.emo_ref_dir or args.emo_route):
        ap.error("--emo-map is only used with --emo-ref-dir and/or --emo-route")

    import soundfile as sf
    import torch
    import torchaudio.functional as AF
    from TTS.api import TTS

    data, units_i, units_j = load_units(args.plan)
    units_dir = os.path.join(args.out, "units")
    os.makedirs(units_dir, exist_ok=True)

    emo_class_map = load_emo_class_map(args.emo_map)
    route_classes = {c.strip() for c in args.emo_route_classes.split(",") if c.strip()}
    bad_route = route_classes - {"neu", "hap", "ang", "sad"}
    if bad_route:
        ap.error(f"--emo-route-classes must use neu/hap/ang/sad, got {sorted(bad_route)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading XTTS-v2 on {device} ...")
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    voice = {"i": args.voice_i, "j": args.voice_j}

    emo_refs = None
    n_emo_mapped = n_emo_fallback = 0
    if args.emo_ref_dir:
        emo_refs = {c: os.path.join(args.emo_ref_dir, f"ref_{name}.wav")
                    for c, name in (("neu", "neutral"), ("hap", "happy"),
                                    ("ang", "angry"), ("sad", "sad"))}
        missing = [p for p in emo_refs.values() if not os.path.isfile(p)]
        if missing:
            raise FileNotFoundError(f"--emo-ref-dir missing reference wav(s): {missing}")
        print(f"emotion-ref cloning ENABLED: {args.emo_ref_dir}")

    if args.emo_route:
        check_emo_route_assets(
            args.emo_route_python, args.emo_route_repo, args.emo_route_ckpt,
            args.emo_route_ref_dir, os.path.join(HERE, "f5emo_synth_batch.py"))
        print("emotion routing ENABLED: "
              f"classes={sorted(route_classes)} repo={args.emo_route_repo}")

    def cls_for(unit: AudioUnit) -> str | None:
        if unit.emotion is None:
            return None
        return emo_class_map.get(unit.emotion.category.lower().strip())

    def pick_ref(unit: AudioUnit) -> str:
        """Default (emo_refs=None): identical to the always-on fixed per-speaker
        voice. Opt-in: clone from the emotion-matched reference when the unit's
        category maps to one of the 4 classes, else fall back to the fixed voice."""
        nonlocal n_emo_mapped, n_emo_fallback
        cls = cls_for(unit)
        if emo_refs is not None and cls in emo_refs:
            n_emo_mapped += 1
            return emo_refs[cls]
        if emo_refs is not None:
            n_emo_fallback += 1
        return voice[unit.speaker]

    def prosody_controls(unit: AudioUnit) -> tuple[float, float, float]:
        speed, n_steps, gain = 1.0, 0.0, 1.0
        if unit.emotion is not None:
            p = unit.emotion.prosody
            speed = float(np.clip(p.rate, 0.7, 1.4))
            n_steps = float(np.clip(p.pitch, -1.0, 1.0)) * 2.0
            gain = float(np.clip(p.energy, 0.6, 1.4))
        return speed, n_steps, gain

    def pitch_gain(wav: np.ndarray, sr: int, n_steps: float, gain: float) -> np.ndarray:
        if abs(n_steps) > 0.1:
            wav = AF.pitch_shift(torch.tensor(wav), sr, n_steps=n_steps).numpy()
        return (wav * gain).astype(np.float32)

    def synth_xtts(unit: AudioUnit, tag: str) -> tuple[np.ndarray, np.ndarray]:
        """Synthesise one unit -> (mono float32 @ SR for Hallo2, native 24k for mix)."""
        speed, n_steps, gain = prosody_controls(unit)
        wav24 = np.asarray(
            tts.tts(text=unit.text, speaker_wav=pick_ref(unit), language="en", speed=speed),
            dtype=np.float32,
        )
        wav24 = pitch_gain(wav24, 24000, n_steps, gain)
        wav16 = AF.resample(torch.tensor(wav24), 24000, SR).numpy().astype(np.float32)
        sf.write(os.path.join(units_dir, f"{tag}.wav"), wav16, SR)
        return wav16, wav24

    def run_f5_batch(pending: list[tuple[AudioUnit, str, str]]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        nonlocal tts
        # F5 runs as a separate process in the f5tts env. Release XTTS first;
        # otherwise both models can occupy the same GPU and trigger OOM.
        del tts
        tts = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        manifest = [{"tag": tag, "text": unit.text, "cls": cls} for unit, tag, cls in pending]
        f5_dir = os.path.join(args.out, "units_f5emo")
        os.makedirs(f5_dir, exist_ok=True)
        manifest_path = os.path.join(args.out, "emo_route_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        cmd = [
            args.emo_route_python,
            os.path.join(HERE, "f5emo_synth_batch.py"),
            "--manifest", manifest_path,
            "--out-dir", f5_dir,
            "--repo", args.emo_route_repo,
            "--ckpt", args.emo_route_ckpt,
            "--ref-dir", args.emo_route_ref_dir,
            "--cfg-strength2", str(args.emo_route_cfg_strength2),
            "--out-sr", str(SR),
            "--device", device,
        ]
        print(f"running F5 emotion batch: {len(pending)} units")
        subprocess.run(cmd, check=True)

        out = {}
        for unit, tag, _cls in pending:
            path = os.path.join(f5_dir, f"{tag}.wav")
            wav16, sr = sf.read(path, dtype="float32")
            if wav16.ndim > 1:
                wav16 = wav16.mean(axis=1)
            if sr != SR:
                wav16 = AF.resample(torch.tensor(wav16), sr, SR).numpy()
            _speed, n_steps, gain = prosody_controls(unit)
            wav16 = pitch_gain(np.asarray(wav16, dtype=np.float32), SR, n_steps, gain)
            sf.write(os.path.join(units_dir, f"{tag}.wav"), wav16, SR)
            wav24 = AF.resample(torch.tensor(wav16), SR, 24000).numpy().astype(np.float32)
            out[tag] = (wav16, wav24)
        return out

    # synthesise everything; measure durations per turn (the slot = active sentence length)
    audio = {}
    audio24 = {}
    durations = {}
    f5_pending: list[tuple[AudioUnit, str, str]] = []
    f5_tags = set()
    for units, side in ((units_i, "i"), (units_j, "j")):
        for u in units:
            if u.kind == "silence" or not u.text:
                continue
            tag = f"{side}_{u.turn:02d}_{u.kind}"
            cls = cls_for(u)
            if args.emo_route and cls in route_classes:
                print(f"  route-F5 {tag} cls={cls}: {u.text[:50]!r}")
                f5_pending.append((u, tag, cls))
                f5_tags.add(tag)
                continue
            print(f"  synth-XTTS {tag}: {u.text[:50]!r}")
            w16, w24 = synth_xtts(u, tag)
            audio[id(u)] = w16
            audio24[id(u)] = w24
            if u.kind == "sentence":
                durations[u.turn] = len(w16) / SR

    if f5_pending:
        f5_audio = run_f5_batch(f5_pending)
        for unit, tag, _cls in f5_pending:
            w16, w24 = f5_audio[tag]
            audio[id(unit)] = w16
            audio24[id(unit)] = w24
            if unit.kind == "sentence":
                durations[unit.turn] = len(w16) / SR
        print(f"emotion routing summary: f5={len(f5_pending)} xtts={len(audio) - len(f5_pending)}")

    if emo_refs is not None:
        print(f"emotion-ref mapping: mapped={n_emo_mapped} fallback={n_emo_fallback}")

    n_turns = len(units_i)
    turn_durations = [durations.get(k + 1, 0.8) for k in range(n_turns)]  # 0.8s floor for silent turns
    # natural turn-taking gap after each turn: ~150 ms base + the active speaker's
    # `pauses` prosody (paper IAR output (ii)) scaling up to ~+350 ms
    gaps = []
    for k in range(n_turns):
        active = units_i[k] if units_i[k].kind == "sentence" else units_j[k]
        pauses = float(active.emotion.prosody.pauses) if active.emotion else 0.0
        gaps.append(0.15 + 0.35 * float(np.clip(pauses, 0.0, 1.0)))
    total = assign_timeline(units_i, units_j, turn_durations, iw_duration=0.5, turn_gaps=gaps)
    print(f"timeline total: {total:.2f}s over {n_turns} turns (gaps {['%.2f' % g for g in gaps[:-1]]})")

    # lay out per-speaker tracks: 16 kHz (drives Hallo2) and native 24 kHz (final mix)
    def layout(bank, sr):
        n = int(np.ceil(total * sr)) + sr // 4
        tracks = {"i": np.zeros(n, np.float32), "j": np.zeros(n, np.float32)}
        for units in (units_i, units_j):
            for u in units:
                w = bank.get(id(u))
                if w is None or u.timespan is None:
                    continue
                gain = args.iw_gain if u.kind == "interactive_word" else 1.0
                start = int(u.timespan.start * sr)
                end = min(start + len(w), n)
                tracks[u.speaker][start:end] += gain * w[: end - start]
        return tracks

    def apply_se(track: np.ndarray, sr: int) -> np.ndarray:
        """Render the sound environment (paper IAR output (iv)) as a light synthetic
        room reverb: exponential-decay noise impulse response, wet level by SE
        label (studio = near-dry, open/outdoor = most reverberant)."""
        se = ((data.get("sound_environment") or {}).get("name") or "studio").lower()
        wet = 0.04
        if any(k in se for k in ("open", "outdoor", "street", "park")):
            wet = 0.10
        elif any(k in se for k in ("cafe", "restaurant", "bar", "office")):
            wet = 0.07
        from scipy.signal import fftconvolve

        rng = np.random.default_rng(0)
        n_ir = int(0.25 * sr)
        t = np.arange(n_ir) / sr
        ir = (rng.standard_normal(n_ir) * np.exp(-t / 0.08)).astype(np.float32)
        ir /= np.abs(ir).sum() ** 0.5
        rev = fftconvolve(track, ir)[: len(track)].astype(np.float32)
        return track + wet * rev

    for suffix, bank, sr in (("", audio, SR), ("_24k", audio24, 24000)):
        for side, tr in layout(bank, sr).items():
            if suffix == "_24k":  # SE colour only on the deliverable mix track
                tr = apply_se(tr, sr)
            peak = float(np.abs(tr).max())
            if peak > 0.99:
                tr = tr * (0.99 / peak)
            sf.write(os.path.join(args.out, f"track_{side}{suffix}.wav"), tr, sr)
            print(f"track_{side}{suffix}.wav: {len(tr)/sr:.2f}s @ {sr}Hz")

    # persist the timed plan for downstream steps
    def dump(units):
        return [
            {
                "turn": u.turn, "speaker": u.speaker, "kind": u.kind, "text": u.text,
                "timespan": None if u.timespan is None else
                {"start": u.timespan.start, "end": u.timespan.end},
            }
            for u in units
        ]

    with open(os.path.join(args.out, "plan_timed.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"sound_environment": data.get("sound_environment"), "total_seconds": total,
             "units_i": dump(units_i), "units_j": dump(units_j),
             "emo_route": {"enabled": bool(args.emo_route), "f5_tags": sorted(f5_tags)}},
            f, ensure_ascii=False, indent=2,
        )
    print("saved plan_timed.json")


if __name__ == "__main__":
    main()
