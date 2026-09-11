"""VAE bridge between pixel video and the SD1.5 latent space (paper Sec. 4.2/4.3).

Hallo2 (the FaceTalk stand-in) generates in the Stable Diffusion 1.5 VAE latent
space; SFBG/RFBG refine 4-channel latent videos of exactly that shape. This module
encodes pixel frames -> latents for refinement and decodes refined latents back to
frames, using the same `sd-vae-ft-mse` weights Hallo2 ships.

diffusers/torch are imported lazily inside the functions so `import chat.ifbg`
stays light; run these on the GPU in the `video` conda env.
"""

from __future__ import annotations

# SD1.5 latent scaling factor (diffusers AutoencoderKL config value).
LATENT_SCALE = 0.18215


def load_vae(path: str, device: str = "cuda", dtype=None):
    """Load the sd-vae-ft-mse AutoencoderKL (e.g. hallo2/pretrained_models/sd-vae-ft-mse)."""
    import torch
    from diffusers import AutoencoderKL

    vae = AutoencoderKL.from_pretrained(path, torch_dtype=dtype or torch.float16)
    return vae.to(device).eval()


def encode_video(vae, frames, batch_size: int = 8):
    """Pixel frames -> scaled latents.

    frames: (F, 3, H, W) float in [0, 1] (H, W divisible by 8)
    returns (F, 4, H/8, W/8) latents (mean of the posterior, scaled by LATENT_SCALE).
    """
    import torch

    dev = next(vae.parameters()).device
    dtype = next(vae.parameters()).dtype
    x = (frames.to(dev, dtype) * 2.0 - 1.0)  # [0,1] -> [-1,1]
    outs = []
    with torch.no_grad():
        for i in range(0, x.shape[0], batch_size):
            posterior = vae.encode(x[i : i + batch_size]).latent_dist
            outs.append(posterior.mean * LATENT_SCALE)
    return torch.cat(outs, dim=0)


def decode_video(vae, latents, batch_size: int = 8):
    """Scaled latents (F, 4, h, w) -> pixel frames (F, 3, 8h, 8w) float in [0, 1]."""
    import torch

    dev = next(vae.parameters()).device
    dtype = next(vae.parameters()).dtype
    z = latents.to(dev, dtype) / LATENT_SCALE
    outs = []
    with torch.no_grad():
        for i in range(0, z.shape[0], batch_size):
            img = vae.decode(z[i : i + batch_size]).sample
            outs.append(((img + 1.0) / 2.0).clamp(0.0, 1.0))
    return torch.cat(outs, dim=0)
