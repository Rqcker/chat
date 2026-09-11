"""Trainable content/emotion encoders for IAR (paper Sec. 4.2, TFM-C / TFM-E).

Each sentence / interactive word is encoded by a content transformer (TFM-C) and
its emotion descriptor by an emotion transformer (TFM-E); the two latents are
summed element-wise into z:

    z_n^i(k) = Sum( TFM-C(dg_n^i(k)), TFM-E(e_n^i(k)) ).

z then, together with the timestamps and the sound environment, conditions an
emotional speech synthesiser (XTTS) that produces the waveform (paper 2026-07-01;
the earlier mel-decoder + HiFi-GAN vocoder was removed). This module is the
trainable-encoder part only; it is torch-based and kept out of ``dadg.iar``'s
package init so the text/logic pipeline stays import-light. Shape-verified on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class TFMConfig:
    vocab_size: int = 8192       # content token vocabulary
    num_emotions: int = 16       # discrete emotion categories
    embed_dim: int = 256
    num_heads: int = 4
    num_layers: int = 2
    ff_dim: int = 1024
    prosody_dim: int = 4         # rate, pitch, energy, pauses
    max_content_len: int = 128
    dropout: float = 0.0


def _sinusoidal(n: int, dim: int, device, dtype=torch.float32) -> torch.Tensor:
    pos = torch.arange(n, device=device).float()[:, None]
    idx = torch.arange(dim, device=device).float()[None, :]
    ang = pos / (10000.0 ** (2.0 * (idx // 2) / dim))
    pe = torch.zeros(n, dim, device=device, dtype=dtype)
    pe[:, 0::2] = torch.sin(ang[:, 0::2]).to(dtype)
    pe[:, 1::2] = torch.cos(ang[:, 1::2]).to(dtype)
    return pe


def _encoder(cfg: TFMConfig) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=cfg.embed_dim, nhead=cfg.num_heads, dim_feedforward=cfg.ff_dim,
        dropout=cfg.dropout, batch_first=True,
    )
    return nn.TransformerEncoder(layer, cfg.num_layers)


class TFMContent(nn.Module):
    """Content transformer encoder (TFM-C): token ids -> pooled latent (B, D)."""

    def __init__(self, cfg: TFMConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.encoder = _encoder(cfg)

    def forward(self, content_ids: torch.Tensor, pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.embed(content_ids)                                          # (B, L, D)
        x = x + _sinusoidal(x.shape[1], self.cfg.embed_dim, x.device, x.dtype).unsqueeze(0)
        h = self.encoder(x, src_key_padding_mask=pad_mask)                   # (B, L, D)
        if pad_mask is not None:
            keep = (~pad_mask).to(h.dtype).unsqueeze(-1)                     # (B, L, 1)
            return (h * keep).sum(1) / keep.sum(1).clamp(min=1.0)
        return h.mean(dim=1)


class TFMEmotion(nn.Module):
    """Emotion transformer encoder (TFM-E): category + prosody -> latent (B, D)."""

    def __init__(self, cfg: TFMConfig):
        super().__init__()
        self.cfg = cfg
        self.cat_embed = nn.Embedding(cfg.num_emotions, cfg.embed_dim)
        self.prosody_proj = nn.Linear(cfg.prosody_dim, cfg.embed_dim)
        self.encoder = _encoder(cfg)

    def forward(self, category_ids: torch.Tensor, prosody: torch.Tensor) -> torch.Tensor:
        cat = self.cat_embed(category_ids)                                   # (B, D)
        pro = self.prosody_proj(prosody)                                     # (B, D)
        tokens = torch.stack([cat, pro], dim=1)                              # (B, 2, D)
        return self.encoder(tokens).mean(dim=1)                              # (B, D)


class IARLatentEncoder(nn.Module):
    """z = TFM-C(content) + TFM-E(emotion) (paper Sec. 4.2 element-wise sum)."""

    def __init__(self, cfg: Optional[TFMConfig] = None):
        super().__init__()
        self.cfg = cfg or TFMConfig()
        self.tfm_c = TFMContent(self.cfg)
        self.tfm_e = TFMEmotion(self.cfg)

    def forward(
        self,
        content_ids: torch.Tensor,
        category_ids: torch.Tensor,
        prosody: torch.Tensor,
        content_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert content_ids.shape[0] == category_ids.shape[0] == prosody.shape[0], (
            "content, category, and prosody must share the batch size"
        )
        return self.tfm_c(content_ids, content_mask) + self.tfm_e(category_ids, prosody)
