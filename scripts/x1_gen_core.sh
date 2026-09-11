#!/bin/bash
# X1 core generator: stages 1-4 of the pipeline plus stage 5 (RFBG+TCR refine, on by
# default -- see REFINE below). No super-resolution pass, no dyadic composition.
# Produces per-side talking-face video + audio + identity for eval-set metric scoring:
#   <out>/video_i.mp4 <out>/video_j.mp4  (base Hallo2 512x512, pre-SR, pre-refine)
#   <out>/video_refined_i.mp4 <out>/video_refined_j.mp4  (RFBG+TCR, REFINE=1 default)
#   <out>/track_i.wav <out>/track_j.wav  <out>/identity_i.png <out>/identity_j.png
#   <out>/plan.json   <out>/script.json
# Skipping the super-resolution pass roughly halves per-clip time, which doubles the
# number of evaluation samples for FID and FVD. It is a cosmetic post-process: every
# metric in DOCS.md is computed on the raw model output, without it.
#
# REFINE (default 1): run x1_refine.py (RFBG+TCR, paper Sec. 4.3/IFBR) after Hallo2.
# video_i.mp4/video_j.mp4 alone are the talking-face backbone (TF) output only --
# the paper's headline numbers are measured with the complete refinement pipeline, and
# its own ablation reports a large gap between all three blocks disabled and all three
# enabled. RFBG and TCR are implemented in chat/ifbg/ifbr/{rfbg,tcr}.py and run by
# default; SFBG (chat/ifbg/ifbr/sfbg.py) is opt-in via SFBG=1 (default OFF, uses
# checkpoints/sfbg_pre_020000.pt at --sfbg-strength 0.3). Strength 1.0 generates from
# noise and decodes to colour noise; do not override SFBG_STRENGTH unless you have
# measured the new value. The block order here is RFBG -> SFBG -> TCR; the paper orders
# SFBG -> RFBG -> TCR (SFBG feeds RFBG there) — see DOCS.md.
# Set REFINE=0 to run the talking-face backbone alone, for comparison.
#
# Usage: bash scripts/x1_gen_core.sh "<scene prompt>" <out_dir> [seed]
# Locations come from the environment, each with a fresh-clone default:
#   CHAT_CODE        this repository            (default: the script's own repo root)
#   CHAT_ASSETS      third-party checkouts/data (default: CHAT_CODE)
#   CHAT_CONDA_SH    conda profile script       (default: ~/miniconda3/.../conda.sh)
#   CHAT_HALLO2      Hallo2 checkout            (default: $CHAT_CODE/hallo2)
#   CHAT_HDTF_DIR    HDTF clips for identities  (default: $CHAT_ASSETS/hdtf/clips_extracted)
#   ARC2FACE_REPO / SD15_PATH / INSIGHTFACE_ROOT  stage 3 third-party paths
#   CHAT_SECRETS     file with API keys, sourced if set
#   RFBG_CKPT / SFBG_CKPT   default to the released weights in checkpoints/
set -euo pipefail

# Every machine-specific location is an environment variable with a default that
# works from a fresh clone. CODE defaults to this script's own repository, so the
# script runs from wherever it is checked out.
CODE="${CHAT_CODE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
P="${CHAT_ASSETS:-$CODE}"                # third-party checkouts, datasets, weights
CONDA_SH="${CHAT_CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
PROMPT="${1:?usage: x1_gen_core.sh \"<scene prompt>\" <out_dir> [seed]}"
OUT="${2:?usage: x1_gen_core.sh \"<scene prompt>\" <out_dir> [seed]}"
SEED="${3:-0}"                 # identity-sampling seed; MUST differ per clip or every clip gets the same two identities
CLIP_SECS="${CLIP_SECS:-10}"   # standard talking-head eval-clip length; full 8-turn tracks are ~70s and make Hallo2 ~7x slower
mkdir -p "$OUT"

# shellcheck disable=SC1090
source "$CONDA_SH"
# XTTS-v2 is under the Coqui Public Model Licence (https://huggingface.co/coqui/XTTS-v2/blob/main/LICENSE.txt). Stage 2
# refuses to run until you have read it and exported COQUI_TOS_AGREED=1 yourself;
# this script deliberately does not accept it on your behalf.
# Caches default to the system location; set these when the home directory is
# quota-limited, as it is on the cluster this was developed on.
[ -n "${CHAT_TMPDIR:-}" ] && export TMPDIR="$CHAT_TMPDIR"
[ -n "${CHAT_HF_HOME:-}" ] && export HF_HOME="$CHAT_HF_HOME"
[ -n "${CHAT_XDG_DATA_HOME:-}" ] && export XDG_DATA_HOME="$CHAT_XDG_DATA_HOME"
# API keys. Supply them however you like; this sources a file if you point at one.
[ -n "${CHAT_SECRETS:-}" ] && source "$CHAT_SECRETS"

echo "=========== [1/5] text (chat env) ==========="
conda activate chat
python "$CODE/scripts/x1_text.py" --prompt "$PROMPT" --out "$OUT" --turns 8

echo "=========== [2/5] tts (tts env) ==========="
conda deactivate; conda activate tts
NVLIBS=$(python -c "import glob,os,site; print(':'.join(sorted(glob.glob(os.path.join(site.getsitepackages()[0],'nvidia','*','lib')))))")
export LD_LIBRARY_PATH="$NVLIBS:${LD_LIBRARY_PATH:-}"
HALLO2="${CHAT_HALLO2:-$CODE/hallo2}"   # Hallo2 checkout (stage 4 and its bundled VAE)
TTS_ARGS=(
  --plan "$OUT/plan.json" --out "$OUT"
  --voice-i "${CHAT_VOICE_I:-$HALLO2/examples/driving_audios/1.wav}"
  --voice-j "${CHAT_VOICE_J:-$HALLO2/examples/driving_audios/3.wav}"
)
if [ -n "${EMO_REF_DIR:-}" ]; then
  TTS_ARGS+=(--emo-ref-dir "$EMO_REF_DIR")
fi
if [ -n "${EMO_MAP:-}" ]; then
  TTS_ARGS+=(--emo-map "$EMO_MAP")
fi
if [ "${EMO_ROUTE:-0}" = "1" ]; then
  TTS_ARGS+=(--emo-route)
fi
python "$CODE/scripts/x1_tts.py" "${TTS_ARGS[@]}"

echo "=========== [3/5] identity portraits (video env) ==========="
conda deactivate; conda activate video
export ARC2FACE_REPO="${ARC2FACE_REPO:-$P/arc2face_repo}" SD15_PATH="${SD15_PATH:-$P/sd15}"
export INSIGHTFACE_ROOT="${INSIGHTFACE_ROOT:-$HOME/.insightface}"
# --hdtf-dir samples real HDTF frames as the identity portraits, which is what every
# reported number used. Add --generate (or drop --hdtf-dir) for the paper's
# descriptor-to-Arc2Face path; see DOCS.md (identity: ID-txt is not the default).
python "$CODE/scripts/x1_faces.py" --script "$OUT/script.json" --out "$OUT" \
  --hdtf-dir "${CHAT_HDTF_DIR:-$P/hdtf/clips_extracted}" --seed "$SEED"

echo "=========== truncate per-side tracks to ${CLIP_SECS}s (standard eval-clip length) ==========="
for s in i j; do
  for f in "$OUT/track_${s}.wav" "$OUT/track_${s}_24k.wav"; do
    if [ -f "$f" ]; then
      ffmpeg -y -loglevel error -i "$f" -t "$CLIP_SECS" "$f.cut.wav" && mv "$f.cut.wav" "$f"
      echo "  truncated $(basename "$f") -> ${CLIP_SECS}s"
    fi
  done
done

echo "=========== [4/5] Hallo2 talking heads (video env, 40 steps) ==========="
python "$CODE/scripts/x1_video.py" --out "$OUT" --hallo2 "$HALLO2" --steps 40

echo "=========== [5/5] RFBG+TCR refine (video env, IFBR paper Sec 4.3) ==========="
if [ "${REFINE:-1}" = "1" ]; then
  # Released weights ship in checkpoints/; these are the exact files every reported
  # refined number was produced with.
  RFBG_CKPT="${RFBG_CKPT:-$CODE/checkpoints/rfbg_qknorm_ft_030500.pt}"
  VAE="${CHAT_VAE:-$HALLO2/pretrained_models/sd-vae-ft-mse}"
  SFBG_ARGS=""
  if [ "${SFBG:-0}" = "1" ]; then
    SFBG_CKPT="${SFBG_CKPT:-$CODE/checkpoints/sfbg_pre_020000.pt}"
    SFBG_STRENGTH="${SFBG_STRENGTH:-0.3}"
    SFBG_ARGS="--sfbg-ckpt $SFBG_CKPT --sfbg-strength $SFBG_STRENGTH --out-suffix _sfbg"
    echo "  SFBG enabled: $SFBG_CKPT strength=$SFBG_STRENGTH"
  fi
  python -u "$CODE/scripts/x1_refine.py" --x1-dir "$OUT" \
    --rfbg-ckpt "$RFBG_CKPT" --vae "$VAE" --sides i j $SFBG_ARGS
  if [ "${SFBG:-0}" = "1" ]; then
    echo "X1_REFINE_COMPLETE -> $OUT/video_refined_i_sfbg.mp4 + video_refined_j_sfbg.mp4 (full IFBR: RFBG+SFBG+TCR)"
  else
    echo "X1_REFINE_COMPLETE -> $OUT/video_refined_i.mp4 + video_refined_j.mp4"
  fi
else
  echo "REFINE=0, skipped -- video_i.mp4/j.mp4 are TF-only, NOT the paper's Table 1 protocol"
fi

echo "X1_CORE_COMPLETE -> $OUT/video_i.mp4 + video_j.mp4 (+ video_refined_* if REFINE=1)"

# ---------- Beyond-paper post-processing ----------
# The lip-sync post-processing reported separately in DOCS.md is an external tool
# applied to video_refined_{i,j}.mp4 after this script finishes. It is not part of the
# CHAT pipeline, it is not invoked here, and its figures are never merged into the
# default-path row. See DOCS.md for what is beyond the paper and why.
