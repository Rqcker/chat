# CHAT — implementation notes

Implementation notes for the CHAT reference implementation (ECCV 2026). The
default configuration follows the paper's method; the table below records each
place the implementation resolves something the paper states differently, and
everything beyond the paper is off unless you switch it on.

Measurements for a given version, and the protocol each one depends on, ship with
that [release](https://github.com/Rqcker/chat/releases).
[`CHANGELOG.md`](CHANGELOG.md) records what each version changes.

**Contents**

- [Implementation against the paper](#implementation-against-the-paper)
- [Beyond the paper](#beyond-the-paper)
- [Evaluation](#evaluation)
- [Third-party components](#third-party-components)
- [Not shipped](#not-shipped)

---

## Implementation against the paper

Twenty-two of the paper's technical statements about the method are implemented as
written. The table records how the implementation resolves each of the rest.

| Paper | This repository | Default |
|---|---|---|
| Talking-face model TF extends FaceTalk | Hallo2, used frozen | on |
| SFBG and RFBG pre-trained on HDTF and REACT 2024 | SFBG on HDTF; RFBG on HDTF cold start, then a NoXI dyadic fine-tune | on |
| TF, SFBG and RFBG jointly optimised end to end | trained separately; no joint stage | on |
| RFBG conditions on the partner's audio and emotion | pathway built, never trained, never supplied at inference | unexercised |
| IAR latent `z = TFM-C + TFM-E` conditions the synthesiser | implemented and unit-tested; no pipeline script calls it | unexercised |
| Speech features concatenated with emotion and injected into TF | not implemented; Hallo2 exposes no emotion input | absent |
| ID-txt converted to a synthetic face image | implemented; the default path uses a real HDTF frame | unexercised |
| IFBR order SFBG → RFBG → TCR | RFBG → SFBG → TCR | SFBG off by default |
| TCR with W = 10 over 100-frame segments | W = 4 over 16-frame segments | on |
| LLMs in TDG and DADG are Gemini | package default is Gemini; the pipeline script defaults to DeepSeek | on |
| Multi-scale conditions upsampled before injection | native-resolution tokens with a key-padding mask | on |
| Audio and emotion injected at the first denoising step | injected at every step, following the paper's equation | on |

Three of these change what the released model can do, so they are worth stating
plainly:

- **No joint training stage.** `train_rfbg.py` and `train_sfbg.py` train their
  modules separately. TF is a frozen third-party checkpoint.
- **RFBG's partner audio and emotion are never supplied.** `RFBGDenoiser` carries
  `audio_proj`, `emo_proj` and a per-block `aux_attn`, but nothing trains or feeds
  them. Responsiveness in this reproduction rests on the partner's visual window
  alone.
- **Identity comes from a real HDTF frame, not from ID-txt.** For the paper's
  descriptor path, run `x1_faces.py --script script.json` with no `--hdtf-dir`.
  With `--hdtf-dir` set, `--generate` still drives Arc2Face from a real embedding.
  That descriptor path generates six candidates per speaker and keeps the best
  under a hand-weighted suitability score, twice over. No number reported for this
  work came from it — the measured path takes a real frame and selects nothing.

Training scale: RFBG holds 1,707,836 trainable parameters and SFBG 1,375,492,
trained primarily on a single 16 GB GPU. Neither saw CHAT-AVD-50k.

To use the paper's language model, both the provider and the endpoint must be set:

```bash
python3 scripts/x1_text.py --prompt "..." --out x1_out \
    --provider gemini --api-key-env GEMINI_API_KEY --model gemini-2.5-flash \
    --endpoint https://generativelanguage.googleapis.com/v1beta/models
```

## Beyond the paper

| Addition | Flag | Default |
|---|---|---|
| QK-normalisation in the conditioning cross-attention | `qk_norm` | `False` in code, **`True` in the released RFBG weight** |
| Raised lip weight in the talking-face backbone | `--lip-weight` | **on**, 1.5 (Hallo2's own default is 1.0) |
| Weight EMA | `--ema-decay` | **on**, 0.999 |
| Classifier-free guidance on the partner condition | `--guidance` | 1.0, off |
| Contrastive partner objective | `--contrast-weight` | 0, off |
| SFBG at inference | `--sfbg-ckpt` | off |
| SFBG architecture: RFBG's block in place of Hallo3's | not a flag | always |
| Emotion-matched reference cloning | `--emo-ref-dir` | `None`, off |
| Per-class text-to-speech routing | `--emo-route` | off |
| Lip-sync post-processing (LatentSync-1.6) | external; no runner ships here | not applied |

Two of these need care:

- The released RFBG checkpoint stores `qk_norm=True` and the loader reads the
  configuration from the file, so the default pipeline with the released weight
  runs with QK-normalisation on. The code default of `False` applies to a model
  built from scratch.
- `--sfbg-strength` defaults to 0.3. At 1.0 the released 20k-step checkpoint
  decodes silent segments to colour noise.

## Evaluation

A metric here is not a single scalar. FID, FVD and lip-sync all move by more than
the differences under discussion when the reference set, the crop or the speech
rule changes. Quote a number with its settings or not at all.

The evaluation set is 93 dyadic clips, 186 speaking sides, generated by
`scripts/x1_gen_core.sh` with a different identity seed per clip, each truncated
to ten seconds (`CLIP_SECS=10`). `video_refined_{i,j}.mp4` is the default subject;
`video_{i,j}.mp4` is the backbone-only stem.

**The two sides of an FID comparison are not sampled the same way, and this
matters when you read the number.** Evaluation identities are screened for
animation suitability: `x1_faces.py` samples up to 30 HDTF clips per dyad and
keeps the first two whose middle frame scores above 0.8 on detection confidence,
frontal pose and a closed mouth (`anim_score`, `x1_faces.py:101-114`). Hallo2
needs such a frame to animate at all. The FID reference built by
`build_fid_reference.py` applies no such filter — it shuffles and takes the first
`--n-clips`. The generated side therefore comes from an easier slice of HDTF than
the reference does, which flatters any identity-sensitive metric to an extent this
repository has not quantified. Passing `--files-from` to `build_fid_reference.py`
lets you build a reference over a chosen clip list if you want to close the gap.

Three more asymmetries are worth knowing before quoting a number. CSIM is
averaged only over frames where a face was detected, and a frame too degraded to
detect is dropped rather than scored, so the metric is biased upward; the scripts
print the skip rate and warn above 5%. `fid_video.py --face-crop` drops a whole
clip when no face is found in its first frame, and now lists what it dropped.
`fvd_video.py --ref-reframe` also caps the reference at three clips per video
while the generated side is uncapped — `--ref-per-video` makes that explicit.

All 186 evaluation sides were synthesised from **two** speaker reference clips, one
per side of the dyad, so voice variation across the set is far smaller than the
side count suggests. Treat MCD and emotion accuracy as 186 measurements of two
voices, not of 186. `CHAT_VOICE_I` and `CHAT_VOICE_J` take your own references.

**FID**, disjoint reference. `--face-crop` must match on both sides.

```bash
python3 scripts/build_fid_reference.py --clips <real_dir> --out fid_ref.npz --face-crop
python3 scripts/fid_video.py --ref fid_ref.npz --videos <eval_dir> --face-crop
```

**FID**, identity-matched reference. Read the result against the real-video floor
measured under the identical protocol, never against the paper's target. Needs
HDTF.

```bash
python scripts/fid_paired_gt.py manifest --eval-dir eval_set2 \
    --logs-dir eval_set2/logs --clips-dir "$CHAT_HDTF_DIR" --out paired_gt_manifest.json
python scripts/fid_paired_gt.py build --manifest paired_gt_manifest.json --out fid_ref_heldout.npz
python scripts/fid_paired_gt.py score --ref fid_ref_heldout.npz --eval-dir eval_set2 --stem video_refined
```

**FVD.** `--ref-reframe` applies the identity-framing transform to the reference
clips so both sides share a framing distribution.

```bash
python3 scripts/fvd_video.py --ref-videos <real_dir> --videos <eval_dir> [--ref-reframe]
```

**LSE-C and LSE-D.** Needs a SyncNet checkout holding `syncnet_v2.model`, via
`--syncnet-dir` or `$CHAT_SYNCNET_DIR`.

```bash
python3 scripts/lse_video.py --video <clip>.mp4 --audio <track>.wav --speech-only \
    --syncnet-dir <syncnet_python checkout>
```

Three settings decide the number: the crop (`--crop s3fd --crop-scale 1.4
--crop-vbias 0.1429`, the geometry of the implementation the paper cites), speech
gating (`--speech-only`, an RMS gate at `thresh_frac=0.15` of the 95th-percentile
RMS), and the offset search (`--vshift 15`). `scripts/x1_refine.py` reuses
`lse_video.speech_segments`, so the SFBG split and the lip-sync figures share one
definition of speech.

**CSIM.**

```bash
python3 scripts/csim_video.py --ref-image <identity>.png --videos <eval_dir> \
    --insightface-root ~/.insightface --frames-per-video 16
```

**Emotion accuracy.** `pairs.json` is a list of `{"audio": ..., "emotion": ...}`
where the emotion is the intended category from the audio-refinement stage, so the
metric scores the pipeline rather than agreement between two classifiers.

```bash
python3 scripts/emo_acc.py --pairs pairs.json --out-json emo_acc.json
```

The classifier caches under `$CHAT_SPEECHBRAIN_DIR` (default
`~/.cache/speechbrain_emo`). Its `hyperparams.yaml` sets a relative `save_path`, so
`HF_HOME` is ignored for the underlying wav2vec2 config and an offline run needs
`./wav2vec2_checkpoints` in the working directory.

**MCD.** `chat/eval/mcd.py` implements the distortion; alignment and cepstral
extraction are the caller's. The reported value excludes the zeroth coefficient,
following Kominek et al., so a plain loudness difference does not move it. Any MCD
claim in this repository uses at least 300 pairs.

**Not reported here.** ViSQOL compares a signal against a reference recording,
which this pipeline does not produce. FRCorr and FRDiv use REACT 2024's
25-dimensional per-frame annotation and its speaker-neighbour matrix, distributed
by the challenge organisers under their own licence; request access from them to
run the official protocol.

## Third-party components

None is redistributed here. Install each from its own source, under its own
licence.

| Component | Used by | Source |
|---|---|---|
| Hallo2 | stage 4 | https://github.com/fudan-generative-vision/hallo2 |
| Arc2Face | stage 3 | https://github.com/foivospar/Arc2Face |
| XTTS-v2 | stage 2 | https://github.com/coqui-ai/TTS |
| insightface | stages 3 and 6 | https://github.com/deepinsight/insightface |
| SyncNet | lip-sync evaluation | https://github.com/joonson/syncnet_python |
| F5-TTS-Emotional-CFG | stage 2, only with `--emo-route` | https://github.com/RaduBolbo/F5-TTS-Emotional-CFG |
| Stable Diffusion 1.5 | stage 3, via Arc2Face | https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5 (CreativeML OpenRAIL-M, carries use restrictions) |
| CLIP ViT-B/32 | stage 3, descriptor encoding | https://github.com/openai/CLIP |
| sd-vae-ft-mse | stages 4 and 5 | https://huggingface.co/stabilityai/sd-vae-ft-mse |
| SpeechBrain wav2vec2 IEMOCAP | emotion-accuracy evaluation | https://huggingface.co/speechbrain/emotion-recognition-wav2vec2-IEMOCAP |
| pytorch-fid InceptionV3 | FID evaluation | https://github.com/mseitzer/pytorch-fid |
| cd-fvd Kinetics-I3D | FVD evaluation | https://github.com/JunyaoHu/common_metrics_on_video_quality |
| Whisper | `--emo-route` transcription only | https://github.com/openai/whisper |

XTTS-v2 is released under the Coqui Public Model Licence
([text](https://huggingface.co/coqui/XTTS-v2/blob/main/LICENSE.txt)). Stage 2 will
not run until you have read it and set `COQUI_TOS_AGREED=1` yourself. This
repository never sets it for you.

## Not shipped

- **HDTF** and **NoXI** are licensed corpora. Obtain them from their own sources;
  the training scripts document the expected layout.
- **REACT 2024** annotations come from the challenge organisers under their own
  licence.
- **CHAT-AVD-50k** is released separately, with provenance metadata and bias
  auditing.
- **Third-party weights** are installed from their own sources.
