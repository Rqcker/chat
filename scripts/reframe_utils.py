"""Identity-framing transform for FID/FVD reference building.

Faithful standalone copy of `feathered_reframe` from x1_faces.py (kept separate so the
proven generation script stays untouched): face scaled to ~55% of a 512 frame, blur-
extended background, feathered paste. Generated eval videos inherit this framing from
their identity images, so a fair FID/FVD reference must apply the SAME transform to the
real reference frames (standard practice: e.g. FFHQ alignment is applied to both real
and generated sides). `measure_reframe` + `apply_reframe` split detection from geometry
so one detection per clip yields a temporally stable transform for FVD reference clips.

Uses insightface antelopev2 (CPU providers fine; ~100 ms/detect).
"""
import os

import numpy as np
from PIL import Image, ImageFilter

_APP = None


def _app(insightface_root: str):
    global _APP
    if _APP is None:
        from insightface.app import FaceAnalysis

        _APP = FaceAnalysis(
            name="antelopev2", root=insightface_root,
            providers=["CPUExecutionProvider"],
        )
        _APP.prepare(ctx_id=-1, det_size=(640, 640))
    return _APP


def detect_largest(rgb: np.ndarray, insightface_root: str):
    """Largest face + padded retry for close-ups (mirrors x1_faces.detect)."""
    app = _app(insightface_root)
    faces = app.get(rgb[:, :, ::-1])
    off = 0
    if not faces:
        h, w = rgb.shape[:2]
        off = h // 2
        padded = np.zeros((h + 2 * off, w + 2 * off, 3), dtype=rgb.dtype)
        padded[off: off + h, off: off + w] = rgb
        faces = app.get(padded[:, :, ::-1])
    if not faces:
        return None, 0
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])), off


def measure_reframe(rgb: np.ndarray, insightface_root: str,
                    face_frac: float = 0.55, size: int = 512):
    """Detect once -> geometry params dict, or None if no face."""
    face, off = detect_largest(rgb, insightface_root)
    if face is None:
        return None
    x1, y1, x2, y2 = (np.asarray(face.bbox) - off).tolist()
    scale = (face_frac * size) / max(y2 - y1, 1.0)
    return {"scale": scale, "cx": (x1 + x2) / 2 * scale, "cy": (y1 + y2) / 2 * scale,
            "size": size}


def apply_reframe(rgb: np.ndarray, params: dict, feather: int = 24) -> np.ndarray:
    """Apply the measured transform to one RGB frame -> (size, size, 3) uint8.

    Geometry identical to x1_faces.feathered_reframe: scaled face paste at
    (size/2 - cx, size*0.45 - cy) over a blur-extended centre-crop background.
    """
    size = params["size"]
    scale = params["scale"]
    img = Image.fromarray(rgb)
    img_s = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))),
                       Image.LANCZOS)
    cover = max(size / img.width, size / img.height)
    bg = img.resize((round(img.width * cover), round(img.height * cover)), Image.LANCZOS)
    bx, by = (bg.width - size) // 2, (bg.height - size) // 2
    canvas = bg.crop((bx, by, bx + size, by + size)).filter(ImageFilter.GaussianBlur(24))
    h, w = img_s.height, img_s.width
    yy = np.minimum(np.arange(h), np.arange(h)[::-1])[:, None]
    xx = np.minimum(np.arange(w), np.arange(w)[::-1])[None, :]
    alpha = (np.clip(np.minimum(yy, xx) / max(feather, 1), 0.0, 1.0) * 255).astype(np.uint8)
    canvas.paste(img_s, (int(round(size / 2 - params["cx"])), int(round(size * 0.45 - params["cy"]))),
                 Image.fromarray(alpha))
    return np.array(canvas)



def measure_face_crop(rgb: np.ndarray, insightface_root: str,
                      size: int = 299, pad: float = 0.35):
    """Detect once -> stable square face-crop geometry, or None.

    One detection per video avoids framewise detector jitter and makes the face-FID
    transform temporally stable and tractable for 46k-frame evals.
    """
    face, off = detect_largest(rgb, insightface_root)
    if face is None:
        return None
    x1, y1, x2, y2 = (np.asarray(face.bbox) - off).tolist()
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * (1.0 + pad)
    return {"cx": float(cx), "cy": float(cy), "side": float(side), "size": int(size)}


def apply_face_crop(rgb: np.ndarray, params: dict) -> np.ndarray:
    """Apply stable face-crop geometry -> (size,size,3) uint8."""
    from PIL import Image

    cx, cy, side = params["cx"], params["cy"], params["side"]
    size = params["size"]
    x1, y1 = int(cx - side / 2), int(cy - side / 2)
    x2, y2 = int(cx + side / 2), int(cy + side / 2)
    h, w = rgb.shape[:2]
    pad_l, pad_t = max(0, -x1), max(0, -y1)
    pad_r, pad_b = max(0, x2 - w), max(0, y2 - h)
    if pad_l or pad_t or pad_r or pad_b:
        rgb = np.pad(rgb, ((pad_t, pad_b), (pad_l, pad_r), (0, 0)), mode="edge")
        x1, x2, y1, y2 = x1 + pad_l, x2 + pad_l, y1 + pad_t, y2 + pad_t
    crop = rgb[y1:y2, x1:x2]
    if crop.size == 0:
        raise ValueError("empty face crop")
    return np.array(Image.fromarray(crop).resize((size, size), Image.LANCZOS))


def face_tight_crop(rgb: np.ndarray, insightface_root: str,
                    size: int = 299, pad: float = 0.35):
    """Single-frame convenience wrapper; long videos should measure once then apply."""
    params = measure_face_crop(rgb, insightface_root, size=size, pad=pad)
    return apply_face_crop(rgb, params) if params is not None else None
