"""Responsive Facial Behaviour Generation (RFBG), paper Sec. 4.3.

A conditional video diffusion transformer that refines an initial facial-behaviour
segment of S^i into one that is responsive to the conversational partner S^j's
audio-visual behaviour over a 3-turn window (k-1:k+1).

Conditioning (paper Eq.):
  - The partner behaviour v_bar_n^j(k-1:k+1) and the identity ID_n^i are spatially
    downsampled to L scales and fused per scale by its OWN cross-attention block:
        C_n^j(k-1:k+1 | l) = CA_l(v_bar_n^j(k-1:k+1 | l), ID_n^i(l)),   l = 1..L
    (l=1 finest, l=L coarsest.) Each scale keeps its NATIVE resolution tokens and
    gets a learnable per-scale embedding.
  - Coarse-to-fine injection: Phi(t) = union_{l=l(t)}^{L} C(l), l(t)=max(1, ceil(L*t/T)).
    The union is realised as the CONCATENATION of the selected scales' token
    sequences; per sample, unselected scales are removed with a key-padding mask.
  - Every latent frame cross-attends to the WHOLE condition window (all partner
    frames, all selected scales) — the partner window length F_p is independent of
    the target segment length F_x. Condition tokens carry 2D spatial + temporal
    position codes; `cond_spatial_pool` / `cond_temporal_stride` bound the memory.
  - The partner audio a_n^j and emotion e_n^j condition a DEDICATED cross-attention
    in every block (separate stream, so they are not diluted by the visual tokens),
    at all denoising steps in both training and sampling. When training WITHOUT
    audio/emotion the aux parameters receive no gradient (DDP: set
    find_unused_parameters=True, or always provide aux — the CHAT pipeline does).

Attention is SPATIO-TEMPORALLY FACTORISED (spatial per-frame + temporal
per-position), because global self-attention over F*H*W latent tokens OOMs at the
real SD1.5 latent resolution (64x64 -> 4096 tokens/frame; measured on a 16GB card
even F=2 fails). All nn.MultiheadAttention calls pass need_weights=False (the
default True materialises the NxN matrix and bypasses fused SDPA — 8x VRAM).

Sampling: the module supports both full generation from noise (strength >= 1.0) and
SDEdit-style refinement (strength < 1.0 renoises the init segment). The paper uses
full generation; the release inference path (x1_refine.py) operates it in SDEdit mode
by default (strength=0.5) to preserve the backbone's quality while adding the
conditioning pathway's effect. `sample_strength < 1` renoises the init segment to
`round(strength*T)-1` then denoises.

Architecture implementation for GPU training; shape-verified on CPU and at
real latent resolution on the GPU.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RFBGConfig:
    latent_channels: int = 4     # channels of the (VAE) latent video
    embed_dim: int = 128         # denoiser token dim
    cond_dim: int = 64           # conditioner attention dim
    num_heads: int = 4
    num_layers: int = 4
    num_scales: int = 3          # L
    num_timesteps: int = 50      # T (paper: 50 denoising steps)
    mlp_ratio: int = 4
    audio_dim: int = 78          # partner audio feature dim
    emotion_dim: int = 32        # partner emotion feature dim
    dropout: float = 0.0
    sample_strength: float = 1.0   # 1.0 = full generation from noise (faithful); <1 = opt-in SDEdit
    cond_spatial_pool: int = 2     # extra spatial pooling of condition tokens in the denoiser memory
    cond_temporal_stride: int = 1  # temporal stride over partner frames in the denoiser memory
    x0_clamp: float = 3.0          # sampling-time x0 clamp; <= 0 disables
    qk_norm: bool = False          # QK-normalise the conditioning cross-attention (bounds attention
                                   # logits -> prevents the entropy-collapse attractor seen in the
                                   # real-dyadic fine-tune). Off by default: keeps nn.MultiheadAttention
                                   # so existing checkpoints/tests are byte-identical.


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Sinusoidal embedding. t: (N,) -> (N, dim), fp32 (cast at the call site)."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device) / max(half, 1))
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


def spatial_embedding(h: int, w: int, dim: int, device) -> torch.Tensor:
    """Factorised 2D position code: rows on the first dim/2 channels, columns on the
    rest -> (h*w, dim). Row/column structure survives resolution changes."""
    dr = dim // 2
    rows = timestep_embedding(torch.arange(h, device=device), dr)          # (h, dr)
    cols = timestep_embedding(torch.arange(w, device=device), dim - dr)    # (w, dim-dr)
    grid = torch.cat(
        [rows.unsqueeze(1).expand(h, w, dr), cols.unsqueeze(0).expand(h, w, dim - dr)], dim=-1
    )
    return grid.reshape(h * w, dim)


def cosine_betas(num_timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Nichol & Dhariwal cosine noise schedule."""
    steps = num_timesteps + 1
    x = torch.linspace(0, num_timesteps, steps)
    ac = torch.cos(((x / num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    ac = ac / ac[0]
    betas = 1 - (ac[1:] / ac[:-1])
    return betas.clamp(1e-4, 0.999)


class MultiScaleConditioner(nn.Module):
    """Downsample partner behaviour + identity to L scales, fuse each by a per-scale
    cross-attention block (partner queries identity), and add a learnable per-scale
    embedding. Each scale KEEPS its native resolution: scale l returns
    (B, F_p, C, H/2^l, W/2^l); index 0 = finest."""

    def __init__(self, cfg: RFBGConfig):
        super().__init__()
        self.cfg = cfg
        C, D, L = cfg.latent_channels, cfg.cond_dim, cfg.num_scales
        self.q_proj = nn.ModuleList([nn.Linear(C, D) for _ in range(L)])
        self.kv_proj = nn.ModuleList([nn.Linear(C, D) for _ in range(L)])
        self.attn = nn.ModuleList(
            [nn.MultiheadAttention(D, cfg.num_heads, batch_first=True, dropout=cfg.dropout) for _ in range(L)]
        )
        self.out_proj = nn.ModuleList([nn.Linear(D, C) for _ in range(L)])
        self.scale_embed = nn.Parameter(torch.zeros(L, C))  # per-scale tag

    def _fuse_scale(self, partner: torch.Tensor, identity: torch.Tensor, l: int) -> torch.Tensor:
        scale = 2 ** l
        B, Fr, C, H, W = partner.shape
        if scale > 1:
            p = F.avg_pool2d(partner.reshape(B * Fr, C, H, W), scale)
            idn = F.avg_pool2d(identity, scale)
        else:
            p = partner.reshape(B * Fr, C, H, W)
            idn = identity
        h, w = p.shape[-2:]
        hi, wi = idn.shape[-2:]  # identity may differ in resolution from the partner
        q = self.q_proj[l](p.reshape(B * Fr, C, h * w).transpose(1, 2))
        kv = self.kv_proj[l](idn.reshape(B, C, hi * wi).transpose(1, 2))
        kv = kv.repeat_interleave(Fr, dim=0)
        fused, _ = self.attn[l](q, kv, kv, need_weights=False)  # fused SDPA, no NxN materialisation
        fused = self.out_proj[l](fused).transpose(1, 2).reshape(B * Fr, C, h, w)
        fused = fused + p
        fused = fused.reshape(B, Fr, C, h, w)
        return fused + self.scale_embed[l].view(1, 1, C, 1, 1)

    def forward(self, partner: torch.Tensor, identity: torch.Tensor) -> List[torch.Tensor]:
        min_hw = min(partner.shape[-2:])
        need = 2 ** (self.cfg.num_scales - 1)
        assert min_hw >= need, (
            f"spatial size {min_hw} too small for {self.cfg.num_scales} scales (need >= {need})"
        )
        return [self._fuse_scale(partner, identity, l) for l in range(self.cfg.num_scales)]


class QKNormCrossAttention(nn.Module):
    """Cross-attention with per-head L2 QK-normalisation on Q and K (Gemma-2 /
    ViT-22B style), the standard fix for attention-logit blow-up / entropy
    collapse. Weight-compatible with nn.MultiheadAttention: q/k/v projections
    concatenate to `in_proj_weight` and out_proj is shared-named, so a model
    pretrained with nn.MultiheadAttention grafts in via `graft_from_mha_state`.

    Logits become cos(q,k) * q_scale * k_scale / sqrt(head_dim) — bounded
    regardless of how large the projection weights grow. The learnable per-head
    scales are init to the pretrained ||q||,||k|| magnitudes at graft time so the
    graft is near-identity (short heal); default init 1.0 for from-scratch.

    Signature/return match nn.MultiheadAttention(need_weights=False): returns
    (out, None) so STBlock.forward is unchanged."""

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.q_scale = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.k_scale = nn.Parameter(torch.ones(num_heads, 1, 1))

    def forward(self, query, key, value, key_padding_mask=None, need_weights=False):
        B, Lq, D = query.shape
        Lk = key.shape[1]
        H, hd = self.num_heads, self.head_dim
        q = self.q_proj(query).view(B, Lq, H, hd).transpose(1, 2)  # (B,H,Lq,hd)
        k = self.k_proj(key).view(B, Lk, H, hd).transpose(1, 2)
        v = self.v_proj(value).view(B, Lk, H, hd).transpose(1, 2)
        # unit-normalise per head, then rescale by learnable per-head scales; the
        # sqrt(hd) folds q_scale so the effective logit scale matches SDPA's 1/sqrt(hd).
        q = F.normalize(q, dim=-1) * (self.q_scale * (hd ** 0.5))
        k = F.normalize(k, dim=-1) * self.k_scale
        attn_mask = None
        if key_padding_mask is not None:
            # nn.MHA convention: True = ignore. SDPA boolean mask: True = keep.
            attn_mask = (~key_padding_mask).view(B, 1, 1, Lk)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)  # (B,H,Lq,hd)
        out = out.transpose(1, 2).reshape(B, Lq, D)
        return self.out_proj(out), None

    @torch.no_grad()
    def graft_from_mha_state(self, in_proj_weight, in_proj_bias, out_w, out_b):
        """Copy an nn.MultiheadAttention's packed QKV projection into the split
        q/k/v projections (+ out_proj). Leaves q_scale/k_scale at their current
        value — call `set_magnitude_scales` afterwards to make the graft identity."""
        D = self.embed_dim
        self.q_proj.weight.copy_(in_proj_weight[0:D]); self.q_proj.bias.copy_(in_proj_bias[0:D])
        self.k_proj.weight.copy_(in_proj_weight[D:2 * D]); self.k_proj.bias.copy_(in_proj_bias[D:2 * D])
        self.v_proj.weight.copy_(in_proj_weight[2 * D:3 * D]); self.v_proj.bias.copy_(in_proj_bias[2 * D:3 * D])
        self.out_proj.weight.copy_(out_w); self.out_proj.bias.copy_(out_b)

    @torch.no_grad()
    def set_magnitude_scales(self, query, key):
        """Init q_scale/k_scale to the per-head RMS magnitude of the projected
        Q,K on a sample batch, so at init the logit scale matches the pre-graft
        model (near-identity graft)."""
        B, Lq, _ = query.shape
        Lk = key.shape[1]
        H, hd = self.num_heads, self.head_dim
        q = self.q_proj(query).view(B, Lq, H, hd).transpose(1, 2)
        k = self.k_proj(key).view(B, Lk, H, hd).transpose(1, 2)
        qn = q.norm(dim=-1).mean(dim=(0, 2)).view(H, 1, 1) / (hd ** 0.5)  # undo the sqrt(hd) fold
        kn = k.norm(dim=-1).mean(dim=(0, 2)).view(H, 1, 1)
        self.q_scale.copy_(qn.clamp_min(1e-3))
        self.k_scale.copy_(kn.clamp_min(1e-3))


class STBlock(nn.Module):
    """Spatio-temporally factorised transformer block:
    spatial self-attention (per frame) -> temporal self-attention (per position) ->
    cross-attention from every frame to the SHARED condition memory ->
    optional dedicated aux (audio/emotion) cross-attention -> MLP.
    Shared by RFBG and SFBG (SFBG passes no aux)."""

    def __init__(self, cfg, use_aux: bool = True):
        super().__init__()
        D = cfg.embed_dim
        self.ns = nn.LayerNorm(D)
        self.spatial_attn = nn.MultiheadAttention(D, cfg.num_heads, batch_first=True, dropout=cfg.dropout)
        self.nt = nn.LayerNorm(D)
        self.temporal_attn = nn.MultiheadAttention(D, cfg.num_heads, batch_first=True, dropout=cfg.dropout)
        self.nc = nn.LayerNorm(D)
        # qk_norm swaps ONLY the conditioning cross-attention (the pathway that
        # collapses); everything else is untouched so qk_norm=False is identical.
        if getattr(cfg, "qk_norm", False):
            self.cross_attn = QKNormCrossAttention(D, cfg.num_heads)
        else:
            self.cross_attn = nn.MultiheadAttention(D, cfg.num_heads, batch_first=True, dropout=cfg.dropout)
        if use_aux:
            self.na = nn.LayerNorm(D)
            self.aux_attn = nn.MultiheadAttention(D, cfg.num_heads, batch_first=True, dropout=cfg.dropout)
        else:
            self.na = self.aux_attn = None
        self.nm = nn.LayerNorm(D)
        self.mlp = nn.Sequential(
            nn.Linear(D, D * cfg.mlp_ratio), nn.GELU(), nn.Linear(D * cfg.mlp_ratio, D)
        )

    def forward(self, x, cond, cond_key_padding_mask=None, aux=None):
        """x: (B, F, N, D). cond: (B, M, D) shared condition memory (every frame sees
        the whole window). cond_key_padding_mask: (B, M) True = ignore.
        aux: (B, Ta, D) or None."""
        B, Fr, N, D = x.shape
        # spatial self-attention within each frame
        h = self.ns(x).reshape(B * Fr, N, D)
        x = x + self.spatial_attn(h, h, h, need_weights=False)[0].reshape(B, Fr, N, D)
        # temporal self-attention across frames at each spatial position
        h = self.nt(x).permute(0, 2, 1, 3).reshape(B * N, Fr, D)
        x = x + self.temporal_attn(h, h, h, need_weights=False)[0].reshape(B, N, Fr, D).permute(0, 2, 1, 3)
        # cross-attention: every frame's tokens query the shared window memory
        h = self.nc(x).reshape(B * Fr, N, D)
        mem = cond.repeat_interleave(Fr, dim=0)
        mask = cond_key_padding_mask.repeat_interleave(Fr, dim=0) if cond_key_padding_mask is not None else None
        x = x + self.cross_attn(h, mem, mem, key_padding_mask=mask, need_weights=False)[0].reshape(B, Fr, N, D)
        # dedicated audio/emotion stream
        if aux is not None:
            h = self.na(x).reshape(B * Fr, N, D)
            a = aux.repeat_interleave(Fr, dim=0)
            x = x + self.aux_attn(h, a, a, need_weights=False)[0].reshape(B, Fr, N, D)
        x = x + self.mlp(self.nm(x))
        return x


class RFBGDenoiser(nn.Module):
    """Predicts the noise on a noised video latent. Factorised attention; every frame
    cross-attends to the full (masked) multi-scale window memory; dedicated
    audio/emotion cross-attention."""

    def __init__(self, cfg: RFBGConfig):
        super().__init__()
        assert cfg.embed_dim % cfg.num_heads == 0, "embed_dim must be divisible by num_heads"
        self.cfg = cfg
        C, D = cfg.latent_channels, cfg.embed_dim
        self.in_proj = nn.Linear(2 * C, D)  # noised latent + initial segment v_n^i(k)
        self.cond_proj = nn.Linear(C, D)
        self.t_mlp = nn.Sequential(nn.Linear(D, D), nn.SiLU(), nn.Linear(D, D))
        self.time_proj = nn.Linear(D, D)  # distinguishes the temporal basis from the spatial one
        self.audio_proj = nn.Linear(cfg.audio_dim, D)
        self.emo_proj = nn.Linear(cfg.emotion_dim, D)
        self.blocks = nn.ModuleList([STBlock(cfg) for _ in range(cfg.num_layers)])
        self.out_norm = nn.LayerNorm(D)
        self.out_proj = nn.Linear(D, C)

    def _cond_tokens(self, cond_maps, scale_mask):
        """cond_maps: list of L (B, F_p, C, h_l, w_l) native-resolution scale maps.
        Builds the shared window memory (B, M, D) + key-padding mask (B, M): all
        partner frames (strided) x all scales (pooled), each token tagged with 2D
        spatial + temporal position codes. F_p is independent of the target length."""
        cfg = self.cfg
        D = cfg.embed_dim
        toks, masks = [], []
        for l, c in enumerate(cond_maps):
            if cfg.cond_temporal_stride > 1:
                c = c[:, :: cfg.cond_temporal_stride]
            B, Fp, C, h, w = c.shape
            if cfg.cond_spatial_pool > 1 and min(h, w) >= cfg.cond_spatial_pool * 2:
                c = F.avg_pool2d(c.reshape(B * Fp, C, h, w), cfg.cond_spatial_pool)
                h, w = c.shape[-2:]
                c = c.reshape(B, Fp, C, h, w)
            t = self.cond_proj(c.reshape(B, Fp, C, h * w).permute(0, 1, 3, 2))  # (B, Fp, hw, D)
            t = t + spatial_embedding(h, w, D, c.device).to(t.dtype).view(1, 1, h * w, D)
            t = t + timestep_embedding(torch.arange(Fp, device=c.device), D).to(t.dtype).view(1, Fp, 1, D)
            toks.append(t.reshape(B, Fp * h * w, D))
            m = (~scale_mask[:, l]).unsqueeze(1).expand(B, Fp * h * w)
            masks.append(m)
        return torch.cat(toks, dim=1), torch.cat(masks, dim=1)

    def forward(self, x, t, cond_maps, scale_mask, init, audio=None, emotion=None):
        """x, init: (B, F, C, H, W). t: (B,). scale_mask: (B, L) bool, True = scale
        selected at this step (from the coarse-to-fine schedule). The partner window
        length in cond_maps need not equal F."""
        B, Fr, C, H, W = x.shape
        N = H * W
        D = self.cfg.embed_dim
        tok = self.in_proj(
            torch.cat([x, init], dim=2).reshape(B, Fr, 2 * C, N).permute(0, 1, 3, 2)
        )  # (B, F, N, D)
        pos_s = spatial_embedding(H, W, D, x.device).to(tok.dtype)
        pos_t = self.time_proj(timestep_embedding(torch.arange(Fr, device=x.device), D).to(tok.dtype))
        tok = tok + pos_s.view(1, 1, N, D) + pos_t.view(1, Fr, 1, D)
        tok = tok + self.t_mlp(timestep_embedding(t, D).to(tok.dtype)).view(B, 1, 1, D)

        memory, key_padding_mask = self._cond_tokens(cond_maps, scale_mask)

        aux = []
        if audio is not None:
            aux.append(self.audio_proj(audio))
        if emotion is not None:
            aux.append(self.emo_proj(emotion))
        aux = torch.cat(aux, dim=1) if aux else None

        for blk in self.blocks:
            tok = blk(tok, memory, cond_key_padding_mask=key_padding_mask, aux=aux)
        out = self.out_proj(self.out_norm(tok))  # (B, F, N, C)
        return out.permute(0, 1, 3, 2).reshape(B, Fr, C, H, W)


class RFBG(nn.Module):
    """Responsive Facial Behaviour Generation diffusion module (paper Sec. 4.3)."""

    def __init__(self, cfg: Optional[RFBGConfig] = None):
        super().__init__()
        self.cfg = cfg or RFBGConfig()
        self.conditioner = MultiScaleConditioner(self.cfg)
        self.denoiser = RFBGDenoiser(self.cfg)
        betas = cosine_betas(self.cfg.num_timesteps)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", torch.cumprod(1.0 - betas, dim=0))
        # (T, L) scale-selection table for the coarse-to-fine schedule.
        table = torch.zeros(self.cfg.num_timesteps, self.cfg.num_scales, dtype=torch.bool)
        for step in range(self.cfg.num_timesteps):
            for idx in self.scale_indices_for_t(step):
                table[step, idx] = True
        self.register_buffer("scale_table", table)

    def scale_indices_for_t(self, t_val: int) -> List[int]:
        """0-based scale indices in the union Phi(t) = {l(t)..L} (paper Eq.)."""
        L, T = self.cfg.num_scales, self.cfg.num_timesteps
        lt = max(1, math.ceil(L * t_val / T))
        return list(range(lt - 1, L))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        ac = self.alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        return ac.sqrt() * x0 + (1.0 - ac).sqrt() * noise

    def training_loss(self, x0, init_segment, partner, audio, emotion, identity,
                      wrong_partner=None, contrast_weight=0.0, contrast_margin=0.02):
        """Epsilon-prediction diffusion loss. x0 is the refined target segment
        tilde_v_n^i(k); init_segment is the initial segment v_n^i(k) being refined.
        partner may have a different frame count (3-turn window).

        If wrong_partner is given and contrast_weight>0, add a hinge that forces the
        model to USE the partner: the correct partner must predict the noise at least
        `contrast_margin` better (lower MSE) than a wrong (mismatched) partner. Under
        plain reconstruction the partner is nearly ignorable (a person's face is mostly
        self-driven), so gradient has little reason to use it; this term rewards
        partner-discrimination directly. Doubles the forward cost when active."""
        B = x0.shape[0]
        t = torch.randint(0, self.cfg.num_timesteps, (B,), device=x0.device)
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        conds = self.conditioner(partner, identity)
        pred = self.denoiser(xt, t, conds, self.scale_table[t], init_segment, audio=audio, emotion=emotion)
        loss = F.mse_loss(pred, noise)
        if wrong_partner is not None and contrast_weight > 0:
            conds_w = self.conditioner(wrong_partner, identity)
            pred_w = self.denoiser(xt, t, conds_w, self.scale_table[t], init_segment, audio=audio, emotion=emotion)
            loss_wrong = F.mse_loss(pred_w, noise)
            hinge = torch.clamp(loss + contrast_margin - loss_wrong, min=0.0)
            loss = loss + contrast_weight * hinge
        return loss

    def _ddpm_step(self, x, pred, step):
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
            return mean + var.sqrt() * torch.randn_like(x)
        return mean

    @torch.no_grad()
    def sample(self, init_segment, partner, audio, emotion, identity, strength=None, guidance_weight=1.0):
        """DDPM ancestral sampling with coarse-to-fine + audio/emotion conditioning.
        guidance_weight>1 applies classifier-free guidance on the partner conditioning.
        strength>=1.0 (default): full generation from noise (the init segment is still
        supplied as channel-concat conditioning). strength<1.0: opt-in SDEdit (renoise
        the init segment to `round(strength*T)-1` then denoise)."""
        strength = self.cfg.sample_strength if strength is None else strength
        shape = init_segment.shape
        B = shape[0]
        device = init_segment.device
        dtype = init_segment.dtype
        T = self.cfg.num_timesteps
        conds = self.conditioner(partner, identity)
        # Classifier-free guidance on the PARTNER conditioning: amplify the partner's
        # effect at inference (guidance_weight>1). The unconditional branch zeroes the
        # partner window (matching the partner-dropout used in training) while keeping
        # identity + init_segment, so guidance amplifies exactly the responsive direction.
        do_cfg = guidance_weight != 1.0
        conds_u = self.conditioner(torch.zeros_like(partner), identity) if do_cfg else None
        if strength >= 1.0:
            start_step = T - 1
            x = torch.randn(shape, device=device, dtype=dtype)
        else:
            start_step = max(0, min(T - 1, int(round(strength * T)) - 1))
            t0 = torch.full((B,), start_step, device=device, dtype=torch.long)
            x = self.q_sample(init_segment, t0, torch.randn(shape, device=device, dtype=dtype))
        for step in reversed(range(start_step + 1)):
            t = torch.full((B,), step, device=device, dtype=torch.long)
            pred = self.denoiser(x, t, conds, self.scale_table[t], init_segment, audio=audio, emotion=emotion)
            if do_cfg:
                pred_u = self.denoiser(x, t, conds_u, self.scale_table[t], init_segment, audio=audio, emotion=emotion)
                pred = pred_u + guidance_weight * (pred - pred_u)
            x = self._ddpm_step(x, pred, step)
        return x
