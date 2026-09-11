"""X1 step 3 (video env): speaker identity portraits for Hallo2.

Identity sources, in order of preference:
  --hdtf-dir       sample REAL frames from HDTF clips (animation-suitability gated:
                   frontal, calm, confident detection). Default: the real frame IS
                   the identity portrait (photorealism guaranteed). With
                   --generate, the frame's ArcFace embedding drives Arc2Face
                   instead (paper's synthetic-identity path; best-of-N with an
                   anim + CLIP semantic gate — its quality gate is still maturing,
                   four artefact modes seen: sketch, laughing, statue, sunglasses).
  --script         paper path: TDG ID-txt -> SD1.5 txt2img reference -> ArcFace
                   emb -> Arc2Face portrait.
  --face-i/-j      photo mode: embeddings from given photos -> Arc2Face.

All paths end with the same recomposition: face to ~55% of frame height (Hallo2
wants 50-70%), blur-extend background, feathered paste (no seam).

Usage examples:
  python scripts/x1_faces.py --hdtf-dir .../clips_extracted --out x1_out
  python scripts/x1_faces.py --script x1_out/script.json --out x1_out
"""

import argparse
import json
import os
import sys

import numpy as np

# Filled in `main` after argparse so `--help` works in a fresh clone that has
# not set the third-party paths. Reading them at import time used to raise
# KeyError before the parser ran.
ARC2FACE_REPO = ""
SD15_PATH = ""
INSIGHTFACE_ROOT = os.environ.get(
    "INSIGHTFACE_ROOT", os.path.expanduser("~/.insightface")
)

POSITIVE = ("professional close-up portrait photograph of {desc}, facing the camera directly, "
            "calm neutral expression, mouth closed, eyes open and clearly visible, "
            "natural lighting, sharp focus, photorealistic, high detail, plain background")
NEGATIVE = ("sunglasses, glasses, hat, cap, hand, microphone, occlusion, open mouth, shouting, "
            "laughing, teeth, extreme expression, side profile, tilted head, looking away, "
            "sketch, drawing, cartoon, anime, painting, illustration, monochrome, "
            "black and white, deformed, disfigured, blurry, multiple people")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdtf-dir", help="real-face bank: sample identities from HDTF clips (recommended)")
    ap.add_argument("--generate", action="store_true",
                    help="with --hdtf-dir: run Arc2Face on the sampled embedding instead of "
                         "using the real frame (synthetic-identity path)")
    ap.add_argument("--script", help="script.json with identity_i/identity_j descriptions")
    ap.add_argument("--face-i", help="photo mode: source photo for speaker i")
    ap.add_argument("--face-j", help="photo mode: source photo for speaker j")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tries", type=int, default=6, help="seeds tried per stage, best kept")
    args = ap.parse_args()

    global ARC2FACE_REPO, SD15_PATH
    try:
        ARC2FACE_REPO = os.environ["ARC2FACE_REPO"]
        SD15_PATH = os.environ["SD15_PATH"]
    except KeyError as exc:
        raise SystemExit(
            f"x1_faces.py needs {exc.args[0]} in the environment. Stage 3 drives "
            "Arc2Face and SD 1.5 from their own checkouts; set ARC2FACE_REPO and "
            "SD15_PATH to those directories. INSIGHTFACE_ROOT is optional and "
            "defaults to ~/.insightface. `python scripts/x1_faces.py --help` "
            "does not require any of them."
        ) from exc
    sys.path.insert(0, ARC2FACE_REPO)

    import torch
    from insightface.app import FaceAnalysis
    from PIL import Image, ImageFilter

    os.makedirs(args.out, exist_ok=True)
    dtype = torch.float16
    app = FaceAnalysis(name="antelopev2", root=INSIGHTFACE_ROOT,
                       providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))

    def detect(rgb: np.ndarray):
        """Largest face + padded retry for close-ups; returns (face|None, pad_offset)."""
        faces = app.get(rgb[:, :, ::-1])
        off = 0
        if not faces:
            h, w = rgb.shape[:2]
            off = h // 2
            padded = np.zeros((h + 2 * off, w + 2 * off, 3), dtype=rgb.dtype)
            padded[off : off + h, off : off + w] = rgb
            faces = app.get(padded[:, :, ::-1])
        if not faces:
            return None, 0
        return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])), off

    def anim_score(face) -> float:
        """Suitability for talking-head animation: detection confidence, frontal
        pose, and a CLOSED mouth (Hallo2 needs a neutral, unoccluded, frontal face)."""
        s = float(face.det_score)
        pose = getattr(face, "pose", None)
        if pose is not None:  # (pitch, yaw, roll) degrees from the 3d68 model
            s -= abs(float(pose[1])) / 60.0
        lmk = getattr(face, "landmark_3d_68", None)
        if lmk is not None:
            bbox_h = max(float(face.bbox[3] - face.bbox[1]), 1.0)
            mouth_open = float(np.linalg.norm(lmk[66][:2] - lmk[62][:2])) / bbox_h
            if mouth_open > 0.06:
                s -= (mouth_open - 0.06) * 5.0
        return s

    def feathered_reframe(img: "Image.Image", face_frac: float = 0.55, size: int = 512,
                          feather: int = 24) -> "Image.Image":
        """Face to ~face_frac of frame height; blur-extend background; feathered
        paste (linear alpha ramp) so there is no picture-in-picture seam."""
        arr = np.array(img)
        face, off = detect(arr)
        assert face is not None, "no face in portrait candidate"
        x1, y1, x2, y2 = (np.asarray(face.bbox) - off).tolist()
        scale = (face_frac * size) / max(y2 - y1, 1.0)
        img_s = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))),
                           Image.LANCZOS)
        cx, cy = (x1 + x2) / 2 * scale, (y1 + y2) / 2 * scale
        cover = max(size / img.width, size / img.height)
        bg = img.resize((round(img.width * cover), round(img.height * cover)), Image.LANCZOS)
        bx, by = (bg.width - size) // 2, (bg.height - size) // 2
        canvas = bg.crop((bx, by, bx + size, by + size)).filter(ImageFilter.GaussianBlur(24))
        h, w = img_s.height, img_s.width
        yy = np.minimum(np.arange(h), np.arange(h)[::-1])[:, None]
        xx = np.minimum(np.arange(w), np.arange(w)[::-1])[None, :]
        alpha = (np.clip(np.minimum(yy, xx) / max(feather, 1), 0.0, 1.0) * 255).astype(np.uint8)
        canvas.paste(img_s, (int(round(size / 2 - cx)), int(round(size * 0.45 - cy))),
                     Image.fromarray(alpha))
        return canvas

    # ---- HDTF real-frame identities (recommended default) ----
    ref_embs = {}
    if args.hdtf_dir:
        import random

        import av

        clips = sorted(
            os.path.join(r, f) for r, _, fs in os.walk(args.hdtf_dir) for f in fs if f.endswith(".mp4")
        )
        assert clips, f"no .mp4 under {args.hdtf_dir}"
        rng = random.Random(args.seed)
        picks = rng.sample(clips, k=min(30, len(clips)))
        found = []
        for path in picks:  # first two clips with a confident, animation-suitable face
            with av.open(path) as c:
                frames = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
            frame = frames[len(frames) // 2]
            face, _ = detect(frame)
            if face is not None and anim_score(face) > 0.8:
                found.append((path, face.embedding, frame))
            if len(found) == 2:
                break
        assert len(found) == 2, "could not find two suitable HDTF identity frames"
        print(f"HDTF identities: {os.path.basename(found[0][0])}, {os.path.basename(found[1][0])}")
        for side, (_, emb, frame) in zip(("i", "j"), found):
            Image.fromarray(frame).save(os.path.join(args.out, f"ref_{side}.png"))
            ref_embs[side] = emb
        if not args.generate:
            # real frame IS the identity portrait: photorealism + animatability guaranteed
            for side, (_, _, frame) in zip(("i", "j"), found):
                out_p = os.path.join(args.out, f"identity_{side}.png")
                feathered_reframe(Image.fromarray(frame)).save(out_p)
                print(f"saved {out_p} (real HDTF frame, reframed ~55% face, feathered)")
            return

    # ---- generative paths below (Arc2Face; CLIP-gated best-of-N) ----
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline, UNet2DConditionModel
    from transformers import CLIPModel, CLIPProcessor

    from arc2face import CLIPTextModelWrapper, project_face_embs

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32",
                                           torch_dtype=torch.float32).to("cuda").eval()
    clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    CLIP_TEXTS = [
        "a photograph of a real human face with a calm neutral expression, no glasses",
        "a sketch, drawing, statue, painting or cartoon of a face",
        "a person laughing or shouting with mouth wide open",
        "a person wearing sunglasses or with an occluded face",
    ]

    def clip_gate(img) -> float:
        with torch.no_grad():
            inputs = clip_proc(text=CLIP_TEXTS, images=img, return_tensors="pt", padding=True).to("cuda")
            probs = clip_model(**inputs).logits_per_image.softmax(-1)[0]
        return float(probs[0] - probs[1] - probs[2] - probs[3])

    def quality(face, img) -> float:
        return anim_score(face) + 1.5 * clip_gate(img)

    # identity sources for the generative paths
    sources = {}
    if ref_embs:  # --hdtf-dir --generate
        sources = {"i": ("emb", None), "j": ("emb", None)}
    elif args.script:
        with open(args.script, encoding="utf-8") as f:
            sc = json.load(f)
        for side in ("i", "j"):
            ident = sc.get(f"identity_{side}") or {}
            desc = ident.get("description") or ident.get("text") or ""
            assert desc, f"script.json has no identity_{side} description"
            sources[side] = ("desc", desc)
    else:
        assert args.face_i and args.face_j, "need --hdtf-dir, --script, or both --face-i/--face-j"
        sources = {"i": ("photo", args.face_i), "j": ("photo", args.face_j)}

    # stage A: reference embeddings (txt2img only for desc sources)
    need_txt2img = any(kind == "desc" for kind, _ in sources.values())
    if need_txt2img:
        print("== stage A: SD1.5 text-to-image reference faces ==")
        ref_pipe = StableDiffusionPipeline.from_pretrained(
            SD15_PATH, torch_dtype=dtype, safety_checker=None
        ).to("cuda")
        ref_pipe.scheduler = DPMSolverMultistepScheduler.from_config(ref_pipe.scheduler.config)
    for side, (kind, src) in sources.items():
        if kind == "emb":
            continue
        if kind == "photo":
            rgb = np.array(Image.open(src).convert("RGB"))
            face, _ = detect(rgb)
            assert face is not None, f"no face in {src}"
            ref_embs[side] = face.embedding
            continue
        best = None
        for k in range(args.tries):
            g = torch.Generator(device="cuda").manual_seed(args.seed + 100 * k + (0 if side == "i" else 7))
            img = ref_pipe(POSITIVE.format(desc=src), negative_prompt=NEGATIVE,
                           num_inference_steps=args.steps, guidance_scale=7.0, generator=g).images[0]
            face, _ = detect(np.array(img))
            score = quality(face, img) if face is not None else -10.0
            print(f"  ref {side} seed#{k}: quality={score:.3f}")
            if face is not None and (best is None or score > best[0]):
                best = (score, face.embedding, img)
        assert best is not None, f"txt2img produced no detectable face for {side}"
        ref_embs[side] = best[1]
        best[2].save(os.path.join(args.out, f"ref_{side}.png"))
    if need_txt2img:
        del ref_pipe
        torch.cuda.empty_cache()

    # stage B: Arc2Face portraits (best-of-N by anim + CLIP quality)
    print("== stage B: Arc2Face portraits ==")
    models = os.path.join(ARC2FACE_REPO, "models")
    encoder = CLIPTextModelWrapper.from_pretrained(models, subfolder="encoder", torch_dtype=dtype)
    unet = UNet2DConditionModel.from_pretrained(models, subfolder="arc2face", torch_dtype=dtype)
    pipe = StableDiffusionPipeline.from_pretrained(
        SD15_PATH, text_encoder=encoder, unet=unet, torch_dtype=dtype, safety_checker=None
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")

    for side in ("i", "j"):
        emb = torch.tensor(ref_embs[side], dtype=dtype)[None].cuda()
        emb = emb / torch.norm(emb, dim=1, keepdim=True)
        emb = project_face_embs(pipe, emb)
        best = None
        for k in range(args.tries):
            g = torch.Generator(device="cuda").manual_seed(args.seed + 1000 + 100 * k + (0 if side == "i" else 7))
            img = pipe(prompt_embeds=emb, num_inference_steps=args.steps,
                       guidance_scale=3.0, generator=g).images[0]
            face, _ = detect(np.array(img))
            score = quality(face, img) if face is not None else -10.0
            print(f"  portrait {side} seed#{k}: quality={score:.3f}")
            if face is not None and (best is None or score > best[0]):
                best = (score, img)
        assert best is not None, f"Arc2Face produced no detectable face for {side}"
        out = os.path.join(args.out, f"identity_{side}.png")
        feathered_reframe(best[1]).save(out)
        print(f"saved {out} (reframed ~55% face, feathered)")


if __name__ == "__main__":
    main()
