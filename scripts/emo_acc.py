"""Emotion accuracy of generated speech (paper ablation Table 3, target 78.4%, higher better).

Runs a pretrained SpeechBrain emotion classifier (wav2vec2 / IEMOCAP, the same
family the paper cites, ravanelli2021speechbrain) on each generated audio clip and
compares the predicted emotion against the emotion the IAR block intended for that
clip. Emo-Acc = fraction of clips whose predicted emotion matches the intended one.

IEMOCAP has 4 classes {neu, ang, hap, sad}; our IAR emotion categories are richer,
so `--map` folds them onto the 4 classes (a sensible default mapping is built in).

Offline note: the classifier's hyperparams.yaml sets `save_path: wav2vec2_checkpoints`, a
RELATIVE path that SpeechBrain hands to transformers as `cache_dir`. HF_HOME is therefore
ignored for the `facebook/wav2vec2-base` config, and under TRANSFORMERS_OFFLINE=1 the load
fails unless `./wav2vec2_checkpoints` exists in the current working directory. Populate it
once with the model online, then symlink that directory into the job's cwd before scoring.
The SpeechBrain download location itself is `$CHAT_SPEECHBRAIN_DIR`, defaulting to
`~/.cache/speechbrain_emo`.

Run in the `video` conda env (SpeechBrain model caches in HF_HOME):
  python scripts/emo_acc.py --pairs pairs.json [--device cuda]
      pairs.json: [{"audio": "clip1.wav", "emotion": "happy"}, ...]
  or just probe one file:
  python scripts/emo_acc.py --audio clip.wav
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.dirname(HERE)
sys.path.insert(0, CODE_ROOT)

from chat.eval import TARGETS, meets_paper  # noqa: E402

# our IAR emotion categories -> IEMOCAP 4-class {neu, ang, hap, sad}
DEFAULT_MAP = {
    "neutral": "neu", "calm": "neu", "serious": "neu", "thoughtful": "neu",
    "happy": "hap", "joyful": "hap", "excited": "hap", "cheerful": "hap", "amused": "hap",
    "vibrant": "hap", "surprised": "hap",
    "angry": "ang", "frustrated": "ang", "annoyed": "ang", "disgusted": "ang",
    "sad": "sad", "fearful": "sad", "anxious": "sad", "disappointed": "sad", "melancholy": "sad",
}


def load_classifier(device):
    from speechbrain.inference.interfaces import foreign_class

    savedir = os.environ.get(
        "CHAT_SPEECHBRAIN_DIR", os.path.expanduser("~/.cache/speechbrain_emo")
    )
    return foreign_class(
        source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
        pymodule_file="custom_interface.py",
        classname="CustomEncoderWav2vec2Classifier",
        savedir=savedir,
        run_opts={"device": device},
    )


def predict(classifier, wav_path):
    out_prob, score, index, text_lab = classifier.classify_file(wav_path)
    lab = text_lab[0] if isinstance(text_lab, list) else text_lab
    return str(lab).lower()  # 'neu' | 'ang' | 'hap' | 'sad'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", help="json list of {audio, emotion} (intended IAR emotion)")
    ap.add_argument("--audio", help="single wav to probe (no accuracy)")
    ap.add_argument("--map", help="json overriding the category->4class map")
    ap.add_argument("--out-json", help="optional: write per-unit predictions and the "
                                       "per-class breakdown here")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import torch
    dev = args.device if torch.cuda.is_available() else "cpu"
    clf = load_classifier(dev)
    cat_map = dict(DEFAULT_MAP)
    if args.map:
        cat_map.update(json.load(open(args.map)))

    if args.audio:
        print(f"{args.audio}: predicted emotion = {predict(clf, args.audio)}")
        print("EMO_ACC_DONE")
        return

    pairs = json.load(open(args.pairs))
    correct = total = 0
    unknown = 0
    records = []
    per_class = {}
    for p in pairs:
        intended = cat_map.get(str(p["emotion"]).lower())
        if intended is None:
            unknown += 1
            continue
        pred = predict(clf, p["audio"])
        hit = int(pred == intended)
        correct += hit
        total += 1
        c = per_class.setdefault(intended, [0, 0])
        c[0] += hit
        c[1] += 1
        records.append({"audio": p["audio"], "emotion": p["emotion"],
                        "intended": intended, "pred": pred})
    acc = 100.0 * correct / total if total else 0.0
    print(f"Emo-Acc = {acc:.1f}%  over {total} clips ({unknown} unmapped emotions skipped)")
    for cls in sorted(per_class):
        hits, n = per_class[cls]
        print(f"  {cls}: {100.0 * hits / n:.1f}% ({hits}/{n})")
    print(f"  (paper ablation Full CHAT target: {TARGETS['Emo-Acc'][0]}%, higher better)")
    print("meets paper (never worse):", meets_paper("Emo-Acc", acc))
    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump({"acc": acc, "total": total, "unmapped_skipped": unknown,
                       "per_class": {k: {"correct": v[0], "n": v[1],
                                         "acc": 100.0 * v[0] / v[1]}
                                     for k, v in sorted(per_class.items())},
                       "predictions": records}, f, indent=2)
        print(f"wrote {args.out_json}")
    print("EMO_ACC_DONE")


if __name__ == "__main__":
    main()
