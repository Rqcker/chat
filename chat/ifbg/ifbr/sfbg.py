"""Silent Facial Behaviour Generation (SFBG), paper Sec. 4.3.

Generates the initial silent facial-behaviour segment for a turn where a speaker is
listening (e.g. nods, blinks), conditioned on the preceding and succeeding talking
segments and the speaker's identity:

    bar_v_n^i(k) = SilentDiff(bar_v_n^i(k-1), bar_v_n^i(k+1), ID_n^i).

The paper states SilentDiff "inherits the Hallo3 architecture" without giving the
conditioning details, so the following design choices are ours (documented for the
reproduction):
  - SilentDiff is a conditional video diffusion (epsilon-prediction) that reuses
    the RFBG spatio-temporally factorised block (`rfbg.STBlock`) and schedule —
    global self-attention over F*H*W latent tokens OOMs at real resolution.
  - The two neighbour segments and the identity image are tokenised with distinct
    learnable role embeddings (prev / next / identity) plus 2D spatial and
    per-frame temporal position codes (so frame ORDER is visible: the prev tail /
    next head — what boundary continuity actually needs — are identifiable).
    Neighbour frames are spatially pooled (``ctx_pool``); identity keeps full
    resolution.
It is shape-verified on CPU; end-to-end training needs a CUDA machine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rfbg import STBlock, cosine_betas, spatial_embedding, timestep_embedding


@dataclass
class SFBGConfig:
    latent_channels: int = 4
    embed_dim: int = 128
    num_heads: int = 4
    num_layers: int = 4
    num_timesteps: int = 50
    mlp_ratio: int = 4
    dropout: float = 0.0
    ctx_pool: int = 2   # spatial pooling of the neighbour-segment condition tokens
    x0_clamp: float = 3.0  # sampling-time x0 clamp; <= 0 disables


class SFBGDenoiser(nn.Module):
    """Predicts the noise on a silent-segment latent, conditioned on the (pooled)
    neighbour talking segments and the full-resolution identity."""

    def __init__(self, cfg: SFBGConfig):
        super().__init__()
        assert cfg.embed_dim % cfg.num_heads == 0, "embed_dim must be divisible by num_heads"
        self.cfg = cfg
        C, D = cfg.latent_channels, cfg.embed_dim
        self.in_proj = nn.Linear(C, D)
        self.cond_proj = nn.Linear(C, D)
        self.role_embed = nn.Embedding(3, D)  # 0=prev, 1=next, 2=identity
        self.t_mlp = nn.Sequential(nn.Linear(D, D), nn.SiLU(), nn.Linear(D, D))
        self.time_proj = nn.Linear(D, D)  # distinguishes the temporal basis from the spatial one
        self.blocks = nn.ModuleList([STBlock(cfg, use_aux=False) for _ in range(cfg.num_layers)])
        self.out_norm = nn.LayerNorm(D)
        self.out_proj = nn.Linear(D, C)

    def _seg_tokens(self, seg: torch.Tensor, role: int) -> torch.Tensor:
        """(B, F, C, H, W) -> pooled (B, F*h*w, D) with role + spatial + FRAME-ORDER codes."""
        B, Fr, C, H, W = seg.shape
        D = self.cfg.embed_dim
        p = self.cfg.ctx_pool
        x = seg.reshape(B * Fr, C, H, W)
        if p > 1:
            x = F.avg_pool2d(x, p)
        h, w = x.shape[-2:]
        tok = self.cond_proj(x.reshape(B, Fr, C, h * w).permute(0, 1, 3, 2))  # (B, F, hw, D)
        tok = tok + spatial_embedding(h, w, D, seg.device).to(tok.dtype).view(1, 1, h * w, D)
        tok = tok + timestep_embedding(torch.arange(Fr, device=seg.device), D).to(tok.dtype).view(1, Fr, 1, D)
        return tok.reshape(B, Fr * h * w, D) + self.role_embed.weight[role]

    def build_condition(self, prev, next_seg, identity) -> torch.Tensor:
        """Condition memory (B, M, D): pooled prev + pooled next + full-res identity."""
        B, C, H, W = identity.shape
        D = self.cfg.embed_dim
        pt = self._seg_tokens(prev, 0)
        nt = self._seg_tokens(next_seg, 1)
        it = self.cond_proj(identity.reshape(B, C, H * W).permute(0, 2, 1))
        it = it + spatial_embedding(H, W, D, identity.device).to(it.dtype)
        it = it + self.role_embed.weight[2]
        return torch.cat([pt, nt, it], dim=1)

    def forward(self, x, t, cond) -> torch.Tensor:
        B, Fr, C, H, W = x.shape
        N = H * W
        D = self.cfg.embed_dim
        tok = self.in_proj(x.reshape(B, Fr, C, N).permute(0, 1, 3, 2))  # (B, F, N, D)
        pos_s = spatial_embedding(H, W, D, x.device).to(tok.dtype)
        pos_t = self.time_proj(timestep_embedding(torch.arange(Fr, device=x.device), D).to(tok.dtype))
        tok = tok + pos_s.view(1, 1, N, D) + pos_t.view(1, Fr, 1, D)
        tok = tok + self.t_mlp(timestep_embedding(t, D).to(tok.dtype)).view(B, 1, 1, D)
        for blk in self.blocks:
            tok = blk(tok, cond)  # STBlock broadcasts the shared memory per frame
        out = self.out_proj(self.out_norm(tok))  # (B, F, N, C)
        return out.permute(0, 1, 3, 2).reshape(B, Fr, C, H, W)


class SFBG(nn.Module):
    """Silent-segment video diffusion (paper Sec. 4.3, SilentDiff)."""

    def __init__(self, cfg: Optional[SFBGConfig] = None):
        super().__init__()
        self.cfg = cfg or SFBGConfig()
        self.denoiser = SFBGDenoiser(self.cfg)
        betas = cosine_betas(self.cfg.num_timesteps)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", torch.cumprod(1.0 - betas, dim=0))

    def q_sample(self, x0, t, noise):
        ac = self.alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        return ac.sqrt() * x0 + (1.0 - ac).sqrt() * noise

    def training_loss(self, silent_gt, prev, next_seg, identity):
        """Epsilon-prediction loss. silent_gt is the target silent segment; prev and
        next_seg are the neighbouring talking segments; identity is the face image."""
        B = silent_gt.shape[0]
        t = torch.randint(0, self.cfg.num_timesteps, (B,), device=silent_gt.device)
        noise = torch.randn_like(silent_gt)
        xt = self.q_sample(silent_gt, t, noise)
        cond = self.denoiser.build_condition(prev, next_seg, identity)
        pred = self.denoiser(xt, t, cond)
        return F.mse_loss(pred, noise)

    @torch.no_grad()
    def sample(self, prev, next_seg, identity, num_frames, init=None, strength=1.0):
        """Generate a silent segment of `num_frames` frames. DDPM ancestral sampling
        (posterior mirrors `rfbg.RFBG.sample`).

        strength >= 1.0 (default): full generation from noise, the original behaviour.
        strength < 1.0 with `init` supplied: SDEdit, matching how `rfbg.RFBG.sample`
        is actually run in production. The init segment is renoised to
        ``round(strength * T) - 1`` and denoised from there, so the result stays on the
        manifold the init came from instead of being synthesised from scratch.

        Why the option exists: at 20k steps and 1.38M parameters this model cannot
        generate a decodable latent segment from pure noise. The 2026-08-14 pilot
        measured +128 FID and the replaced frames decode to colour noise rather than
        faces. Anchoring on the RFBG output keeps identity and background intact and
        limits SFBG to adjusting behaviour, which is what the module is for.
        """
        B, C, H, W = prev.shape[0], prev.shape[2], prev.shape[3], prev.shape[4]
        device = prev.device
        dtype = prev.dtype
        cond = self.denoiser.build_condition(prev, next_seg, identity)
        T = self.cfg.num_timesteps
        if init is not None and strength < 1.0:
            if init.shape[1] != num_frames:
                raise ValueError(
                    f"init has {init.shape[1]} frames, expected {num_frames}"
                )
            start_step = max(0, min(T - 1, int(round(strength * T)) - 1))
            t0 = torch.full((B,), start_step, device=device, dtype=torch.long)
            x = self.q_sample(init.to(device=device, dtype=dtype), t0,
                              torch.randn(B, num_frames, C, H, W, device=device, dtype=dtype))
        else:
            start_step = T - 1
            x = torch.randn(B, num_frames, C, H, W, device=device, dtype=dtype)
        for step in reversed(range(start_step + 1)):
            t = torch.full((B,), step, device=device, dtype=torch.long)
            pred = self.denoiser(x, t, cond)
            ac = self.alphas_cumprod[step]
            ac_prev = self.alphas_cumprod[step - 1] if step > 0 else x.new_tensor(1.0)
            beta = self.betas[step]
            x0 = (x - (1.0 - ac).sqrt() * pred) / ac.sqrt()
            if self.cfg.x0_clamp > 0:
                x0 = x0.clamp(-self.cfg.x0_clamp, self.cfg.x0_clamp)
            coef_x0 = (ac_prev.sqrt() * beta) / (1.0 - ac)
            coef_xt = ((1.0 - beta).sqrt() * (1.0 - ac_prev)) / (1.0 - ac)
            mean = coef_x0 * x0 + coef_xt * x
            if step > 0:
                var = beta * (1.0 - ac_prev) / (1.0 - ac)
                x = mean + var.sqrt() * torch.randn_like(x)
            else:
                x = mean
        return x
