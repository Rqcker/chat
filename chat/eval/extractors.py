"""Pluggable feature extractors for the evaluation harness (GPU required).

Kept OUT of ``chat.eval.__init__`` so ``import chat.eval`` stays numpy-only; this
module pulls torch + model weights and is imported explicitly where a GPU exists.

FID uses the standard pytorch-fid InceptionV3 pool3 (2048-d) features, so the FID
values are comparable with the literature. Feed the extracted features to
``chat.eval.frechet_from_features``; compare the result against ``TARGETS["FID"]``.
"""

from __future__ import annotations

import numpy as np


class InceptionExtractor:
    """Standard FID InceptionV3 pool3 features (2048-d) via pytorch-fid."""

    def __init__(self, device: str | None = None, dims: int = 2048, batch_size: int = 32):
        import torch
        from pytorch_fid.inception import InceptionV3

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.dims = dims
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        self.model = InceptionV3([block_idx]).to(self.device).eval()

    def features(self, images) -> np.ndarray:
        """images: (N, 3, H, W) tensor/array, float in [0, 1] (or uint8 0-255)
        -> (N, dims) numpy features. RGB, 3 channels required."""
        torch = self.torch
        x = images if torch.is_tensor(images) else torch.as_tensor(np.asarray(images))
        x = x.float()
        if float(x.max()) > 1.5:  # tolerate 0-255 inputs
            x = x / 255.0
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f"expected (N,3,H,W) RGB images, got {tuple(x.shape)}")
        x = x.to(self.device)
        out = []
        with torch.no_grad():
            for i in range(0, x.shape[0], self.batch_size):
                pred = self.model(x[i : i + self.batch_size])[0]  # (b, dims, 1, 1)
                out.append(pred.squeeze(-1).squeeze(-1).cpu().numpy())
        return np.concatenate(out, axis=0)


def fid(real_images, gen_images, extractor: InceptionExtractor | None = None) -> float:
    """End-to-end FID from two image batches (real vs generated)."""
    from .frechet import frechet_from_features

    ext = extractor or InceptionExtractor()
    return frechet_from_features(ext.features(real_images), ext.features(gen_images))


class ArcFaceExtractor:
    """Identity embeddings for CSIM via insightface antelopev2 (glintr100).

    insightface ignores the INSIGHTFACE_HOME environment variable, so the model
    root has to be passed explicitly, for example ``~/.insightface``."""

    def __init__(self, root: str, det_size: int = 640, device: str = "cuda"):
        from insightface.app import FaceAnalysis

        if device == "cpu":
            providers, ctx_id = ["CPUExecutionProvider"], -1
        else:
            providers, ctx_id = ["CUDAExecutionProvider", "CPUExecutionProvider"], 0
        self.app = FaceAnalysis(name="antelopev2", root=root, providers=providers)
        self.app.prepare(ctx_id=ctx_id, det_size=(det_size, det_size))

    def embed_frame(self, rgb: np.ndarray):
        """One RGB uint8 frame (H, W, 3) -> 512-d embedding of the LARGEST face,
        or None if no face is detected."""
        rgb = np.asarray(rgb)
        faces = self.app.get(rgb[:, :, ::-1])  # insightface expects BGR
        if not faces:
            # Close-up faces filling the frame fall outside SCRFD's training
            # distribution; retry with 50% padding so the face looks smaller.
            h, w = rgb.shape[:2]
            ph, pw = h // 2, w // 2
            padded = np.zeros((h + 2 * ph, w + 2 * pw, 3), dtype=rgb.dtype)
            padded[ph : ph + h, pw : pw + w] = rgb
            faces = self.app.get(padded[:, :, ::-1])
        if not faces:
            return None
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        return np.asarray(face.embedding, dtype=np.float64)

    def embed_frames(self, frames) -> tuple:
        """frames: iterable of RGB uint8 (H, W, 3) -> ((M, 512) embeddings, skipped)."""
        out, skipped = [], 0
        for f in frames:
            emb = self.embed_frame(np.asarray(f))
            if emb is None:
                skipped += 1
            else:
                out.append(emb)
        if not out:
            raise ValueError("no faces detected in any frame")
        return np.stack(out), skipped
