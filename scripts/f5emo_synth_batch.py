#!/usr/bin/env python3
"""Batch-synthesise tagged units with F5-TTS-Emotional-CFG (beyond-paper IAR engine).

Called as a subprocess from `x1_tts.py --emo-route` so the F5 model is loaded once
per pipeline run rather than once per unit. Runs in the `f5tts` env (its own torch);
`x1_tts.py` runs in the `tts` env, so an in-process import is not possible.

Input  manifest: [{"tag": "i_01_sentence", "text": "...", "cls": "hap"}, ...]
Output: <out-dir>/<tag>.wav, mono 16 kHz (the rate Hallo2 consumes).
Exit non-zero if any unit fails, so the caller can fall back loudly, not silently.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

CLS_TO_EMO = {"neu": "Neutral", "hap": "Happy", "ang": "Angry", "sad": "Sad"}
EMO_TO_FILE = {"Neutral": "neutral", "Happy": "happy", "Angry": "angry", "Sad": "sad"}


def build_tts(repo, ckpt, device):
    from f5_tts.infer.infer_emotion import (
        CFMConditioned, DiTConditioned, TTSModel, get_tokenizer,
        n_fft, hop_length, win_length, n_mel_channels, target_sample_rate, mel_spec_type,
    )
    mel_kwargs = dict(n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                      n_mel_channels=n_mel_channels,
                      target_sample_rate=target_sample_rate, mel_spec_type=mel_spec_type)
    vocab_char_map, vocab_size = get_tokenizer("EmiliaPetite_dataset_ZH_EN", "pinyin")
    emo_cond = {"emotion_condition_type": "text_mirror", "init_type": "xavier_reduced",
                "weight_reduction_scale": 1.0, "emotion_dim": 128,
                "emotion_conv_layers": 4, "load_emotion_weights": False}
    transformer = DiTConditioned(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512,
                                 emotion_dim=128, conv_layers=4,
                                 text_num_embeds=vocab_size, mel_dim=n_mel_channels,
                                 emotion_conditioning=emo_cond)
    model = CFMConditioned(transformer=transformer, mel_spec_kwargs=mel_kwargs,
                           vocab_char_map=vocab_char_map)
    tts = TTSModel(model=model, vocoder_name=mel_spec_type, checkpoint_path=ckpt,
                   emotion_conditioning_parameters=emo_cond, device=device)
    return tts, mel_kwargs, target_sample_rate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ref-dir", required=True,
                    help="dir holding 0015_<emo>.wav (or ref_<emo>.wav) reference clips")
    ap.add_argument("--cfg-strength2", type=float, default=30.0)
    ap.add_argument("--out-sr", type=int, default=16000)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    # Emotion-CFG lives in its own checkout; its package must be importable BEFORE
    # any `f5_tts.*` import below (it is not pip-installed into the env).
    src = os.path.join(args.repo, "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    # get_tokenizer() resolves its vocab via the RELATIVE path
    # "data/<dataset>_pinyin/vocab.txt", so the repo root must be the cwd.
    # Resolve caller-supplied paths to absolute first, since we are about to move.
    args.manifest = os.path.abspath(args.manifest)
    args.out_dir = os.path.abspath(args.out_dir)
    args.ckpt = os.path.abspath(args.ckpt)
    args.ref_dir = os.path.abspath(args.ref_dir)
    os.chdir(args.repo)

    units = json.loads(Path(args.manifest).read_text())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_dir = Path(args.ref_dir)
    ref_paths = {}
    for emo, short in EMO_TO_FILE.items():
        hit = next((p for p in (ref_dir / f"0015_{short}.wav",
                                ref_dir / f"0011_{short}.wav",
                                ref_dir / f"ref_{short}.wav") if p.is_file()), None)
        if hit is None:
            raise SystemExit(f"missing F5 reference clip for {emo} in {ref_dir}")
        ref_paths[emo] = hit

    import whisper
    wmodel = whisper.load_model("base", device=args.device)
    ref_text = {}
    for emo, path in ref_paths.items():
        t = (wmodel.transcribe(str(path), language="en").get("text") or "").strip()
        if t and t[-1] not in ".!?":
            t += "."
        ref_text[emo] = t
    del wmodel
    torch.cuda.empty_cache()

    from f5_tts.infer.infer_emotion import compute_mel_from_wav, nfe_step, cfg_strength, sway_sampling_coef
    tts, mel_kwargs, native_sr = build_tts(args.repo, args.ckpt, args.device)
    ref_mel = {e: compute_mel_from_wav(str(p), mel_kwargs, device=args.device)
               for e, p in ref_paths.items()}

    done, errors = 0, []
    for u in units:
        cls, tag = u.get("cls"), u.get("tag")
        text = (u.get("text") or "").strip()
        if cls not in CLS_TO_EMO or not text or not tag:
            errors.append({"tag": tag, "error": f"bad unit cls={cls}"})
            continue
        if text[-1] not in ".!?":
            text += "."
        emo = CLS_TO_EMO[cls]
        dst = out_dir / f"{tag}.wav"
        try:
            t0 = time.time()
            _mel, gen = tts.infer(
                inference_text=text, inference_emotion=emo,
                ref_mel=ref_mel[emo], ref_text=ref_text[emo], ref_emotion=emo,
                steps=nfe_step, cfg_strength=cfg_strength,
                cfg_strength2=args.cfg_strength2, sway_sampling_coef=sway_sampling_coef,
            )
            wav = gen.detach().cpu()
            if wav.ndim == 1:
                wav = wav.unsqueeze(0)
            if native_sr != args.out_sr:
                wav = torchaudio.functional.resample(wav, native_sr, args.out_sr)
            torchaudio.save(str(dst), wav, args.out_sr)
            if dst.stat().st_size <= 0:
                raise RuntimeError("empty output")
            done += 1
            print(f"F5EMO_OK {tag} cls={cls} {time.time()-t0:.1f}s", flush=True)
        except Exception as exc:
            errors.append({"tag": tag, "error": repr(exc)})
            print(f"F5EMO_ERR {tag} {exc!r}", flush=True)

    print(f"F5EMO_BATCH_DONE done={done}/{len(units)} errors={len(errors)}", flush=True)
    if errors:
        print("F5EMO_ERRORS " + json.dumps(errors[:5], ensure_ascii=False), flush=True)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
