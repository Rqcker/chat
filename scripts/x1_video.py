"""X1 step 4 (video env): per-speaker identity portrait + audio track -> talking-
head video. This implements the paper's **TF** (talking face model, Sec. 4.3),
which the paper describes as extending FaceTalk (`aneja2024facetalk`) without
claiming to use its released code or weights. FaceTalk has no public pretrained
checkpoint (verified at its official repo), so TF is realised here on the
**Hallo2** backbone. Writes a per-speaker inference config (based on the smoke
config) and invokes Hallo2's inference_long.py.

Usage:
  python scripts/x1_video.py --out x1_out --hallo2 <hallo2 repo dir> [--steps 20]
Requires x1_out/identity_i.png, identity_j.png, track_i.wav, track_j.wav.
"""

import argparse
import os
import subprocess
import sys

TEMPLATE = """source_image: {image}
driving_audio: {audio}

weight_dtype: fp16

data:
  n_motion_frames: 2
  n_sample_frames: 16
  source_image:
    width: 512
    height: 512
  driving_audio:
    sample_rate: 16000
  export_video:
    fps: 25

inference_steps: {steps}
cfg_scale: {cfg_scale}

use_mask: true
mask_rate: 0.25
use_cut: true

audio_ckpt_dir: pretrained_models/hallo2

save_path: {save_path}
cache_path: ./.cache

base_model_path: ./pretrained_models/stable-diffusion-v1-5
motion_module_path: ./pretrained_models/motion_module/mm_sd_v15_v2.ckpt

face_analysis:
  model_path: ./pretrained_models/face_analysis

wav2vec:
  model_path: ./pretrained_models/wav2vec/wav2vec2-base-960h
  features: all

audio_separator:
  model_path: ./pretrained_models/audio_separator/Kim_Vocal_2.onnx

vae:
  model_path: ./pretrained_models/sd-vae-ft-mse

face_expand_ratio: 1.2
pose_weight: {pose_weight}
face_weight: {face_weight}
lip_weight: {lip_weight}

unet_additional_kwargs:
  use_inflated_groupnorm: true
  unet_use_cross_frame_attention: false
  unet_use_temporal_attention: false
  use_motion_module: true
  use_audio_module: true
  motion_module_resolutions:
    - 1
    - 2
    - 4
    - 8
  motion_module_mid_block: true
  motion_module_decoder_only: false
  motion_module_type: Vanilla
  motion_module_kwargs:
    num_attention_heads: 8
    num_transformer_block: 1
    attention_block_types:
      - Temporal_Self
      - Temporal_Self
    temporal_position_encoding: true
    temporal_position_encoding_max_len: 32
    temporal_attention_dim_div: 1
  audio_attention_dim: 768
  stack_enable_blocks_name:
    - "up"
    - "down"
    - "mid"
  stack_enable_blocks_depth: [0,1,2,3]

enable_zero_snr: true

noise_scheduler_kwargs:
  beta_start: 0.00085
  beta_end: 0.012
  beta_schedule: "linear"
  clip_sample: false
  steps_offset: 1
  prediction_type: "v_prediction"
  rescale_betas_zero_snr: True
  timestep_spacing: "trailing"

sampler: DDIM
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="x1 output dir (absolute)")
    ap.add_argument("--hallo2", required=True, help="hallo2 repo dir")
    ap.add_argument("--steps", type=int, default=40)  # Hallo2's official inference_steps
    ap.add_argument("--cfg-scale", type=float, default=3.5)
    # 1.5 rather than Hallo2's default of 1.0: raising the lip weight improved
    # lip synchronisation without costing image quality in our own sweep.
    ap.add_argument("--lip-weight", type=float, default=1.5)
    ap.add_argument("--face-weight", type=float, default=1.0)
    ap.add_argument("--pose-weight", type=float, default=1.0)
    ap.add_argument("--sides", nargs="+", default=["i", "j"])
    args = ap.parse_args()

    out = os.path.abspath(args.out)
    for side in args.sides:
        image = os.path.join(out, f"identity_{side}.png")
        audio = os.path.join(out, f"track_{side}.wav")
        assert os.path.isfile(image) and os.path.isfile(audio), f"missing inputs for {side}"
        save_path = os.path.join(out, f"hallo2_{side}")
        cfg_path = os.path.join(out, f"hallo2_{side}.yaml")
        with open(cfg_path, "w", encoding="utf-8") as f:
            f.write(TEMPLATE.format(image=image, audio=audio, steps=args.steps, cfg_scale=args.cfg_scale,
                                    pose_weight=args.pose_weight, face_weight=args.face_weight,
                                    lip_weight=args.lip_weight, save_path=save_path))
        print(f"== Hallo2 for speaker {side} ==")
        proc = subprocess.run(
            [sys.executable, "scripts/inference_long.py", "--config", cfg_path],
            cwd=args.hallo2, text=True, capture_output=True,
        )
        tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-12:])
        print(tail)
        if proc.returncode != 0:
            raise SystemExit(f"hallo2 failed for {side} (rc={proc.returncode})")
        stem = f"identity_{side}"
        merged = os.path.join(save_path, stem, "merge_video.mp4")
        assert os.path.isfile(merged), f"expected {merged}"
        final = os.path.join(out, f"video_{side}.mp4")
        if os.path.exists(final):
            os.remove(final)
        os.link(merged, final)
        print(f"video_{side}.mp4 ready")


if __name__ == "__main__":
    main()
