# Pinned environments

The pipeline runs in three separate environments because their dependencies conflict:
stage 1 needs only the language-model client, stage 2 needs the speech stack, and
stages 3 to 6 need the video stack. Do not merge them into one environment.

Every version below was read from the machine that produced the reported numbers
(`pip list --format=freeze`, 2026-08-17). They are the versions the results were
measured under, not a minimum-supported range.

| file | env | stages | Python |
|---|---|---|---|
| `chat.txt` | `chat` | 1 (`x1_text.py`) | 3.11.15 |
| `tts.txt` | `tts` | 2 (`x1_tts.py`) | 3.11.15 |
| `video.txt` | `video` | 3-6 (`x1_faces`, `x1_video`, `x1_refine`, `x1_compose`) and every metric script under `scripts/` | 3.10.20 |

```bash
conda create -n chat  python=3.11.15 && conda activate chat  && pip install -r requirements/chat.txt
conda create -n tts   python=3.11.15 && conda activate tts   && pip install -r requirements/tts.txt
conda create -n video python=3.10.20 && conda activate video && pip install -r requirements/video.txt
```

`pytest` is not in any of the three files either. The unit tests need no network
and no GPU, so install it separately when you want to run them.

## What is deliberately not pinned here

`torch` is listed as the `+cu128` build actually used. Install the wheel that matches
your CUDA version from https://pytorch.org rather than taking the pin literally, then
install the rest of the file.

The third-party checkouts (Hallo2, Arc2Face, XTTS-v2, and optionally
F5-TTS-Emotional-CFG) are not pip dependencies. Install them from their own repositories
as the README describes; each carries its own licence.

## Hardware the numbers were measured on

NVIDIA A16 (16 GB) and RTX 5070 Ti (16 GB), CUDA 12.8. Results are reproducible
run-to-run on the same machine when `--seed` is passed, but diffusion sampling is not
bit-exact across different GPU models, so small numerical differences are expected.
