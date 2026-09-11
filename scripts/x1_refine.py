"""X1 step: IFBR refinement — apply a trained RFBG to the Hallo2 talking heads.

Per speaker: video -> SD1.5-VAE latents -> 16-frame segments -> each segment
refined by RFBG conditioned on the PARTNER's 3-segment window (k-1:k+1, paper
Sec. 4.3) and the speaker identity -> optional SFBG replacement of silent segments
(paper Sec. 4.3, SilentDiff generates listening-face behaviour conditioned on the
neighbouring talking segments and identity) -> TCR half-Gaussian boundary cross-fade
-> decode -> refined video. Default sampling is SDEdit (strength<1): the pre-trained
RFBG refines Hallo2's segment instead of regenerating from pure noise, preserving
the backbone's quality while adding the conditioning pathway's effect.

The paper's IFBR order is SFBG -> RFBG -> TCR (Sec. 4.3: SFBG generates the initial
silent segments, RFBG then refines those and the talking segments, TCR blends the
boundaries). This release runs RFBG -> SFBG -> TCR when SFBG is enabled: the backbone
already renders silent frames, so RFBG refines the whole video first and SFBG then
replaces the silent segments anchored on that output (the released SFBG checkpoint
cannot pre-generate decodable segments from noise). See DOCS.md.
SFBG is opt-in via ``--sfbg-ckpt``; when omitted, the pipeline is RFBG+TCR only.

Run in the `video` conda env (GPU):
  python scripts/x1_refine.py --x1-dir <dir with video_i/j.mp4, identity_*.png> \
      --rfbg-ckpt runs/rfbg_pre/rfbg_020000.pt --vae <sd-vae-ft-mse> \
      [--frames 16] [--strength 0.5] [--tcr-window 4] \
      [--sfbg-ckpt runs/sfbg_pre/sfbg_020000.pt]
"""

import argparse
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.dirname(HERE)
sys.path.insert(0, CODE_ROOT)

from chat.ifbg.ifbr.rfbg import RFBG, RFBGConfig  # noqa: E402
from chat.ifbg.ifbr.sfbg import SFBG, SFBGConfig  # noqa: E402
from chat.ifbg.ifbr.tcr import apply_tcr  # noqa: E402


def load_rfbg(ckpt_path: str, device: str, weights: str = "auto"):
    import torch

    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    cfg = RFBGConfig(**state["cfg"]) if "cfg" in state else RFBGConfig()
    model = RFBG(cfg).to(device).eval()
    if weights == "raw":
        sd = state["model"]
    elif weights == "ema":
        sd = state["model_ema"]  # KeyError if this ckpt has no EMA -- intended, don't silently fall back
    else:  # auto: prefer EMA when present (smoother), else raw
        sd = state.get("model_ema", state["model"])
    model.load_state_dict(sd)
    return model


def load_sfbg(ckpt_path: str, device: str, weights: str = "auto"):
    """Load a trained SFBG model from checkpoint (same format as RFBG)."""
    import torch

    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    cfg = SFBGConfig(**state["cfg"]) if "cfg" in state else SFBGConfig()
    model = SFBG(cfg).to(device).eval()
    sd = state.get("model_ema", state["model"]) if weights == "auto" else state[weights]
    model.load_state_dict(sd)
    return model


def _silent_segment_mask(audio_path, n_frames, fps=25.0, frames_per_seg=16,
                         thresh_frac=0.15, silence_ratio=0.5):
    """Boolean array (n_seg,) marking segments that are predominantly silent.

    Uses the same RMS speech gate as ``lse_video.speech_segments`` so the split
    matches every LSE figure in this project. A segment is "silent" if >= silence_ratio
    of its frames fall in the gate's complement.
    """
    from lse_video import speech_segments

    duration = n_frames / fps if fps > 0 else 0.0
    speech = speech_segments(audio_path, thresh_frac=thresh_frac)

    # Build a per-frame speech boolean
    is_speech = np.zeros(n_frames, dtype=bool)
    for a, b in speech:
        lo, hi = int(round(a * fps)), int(round(b * fps))
        is_speech[lo:hi] = True

    n_seg = -(-n_frames // frames_per_seg)
    mask = np.zeros(n_seg, dtype=bool)
    for k in range(n_seg):
        lo, hi = k * frames_per_seg, min((k + 1) * frames_per_seg, n_frames)
        seg_frames = is_speech[lo:hi]
        if seg_frames.size > 0 and seg_frames.mean() < (1.0 - silence_ratio):
            mask[k] = True
    return mask


def apply_sfbg(refined_segs, sfbg_model, identity, silent_mask, frames=16,
               ctx_frames=8, strength=1.0):
    """Replace silent segments with SFBG-generated listening-face latents.

    refined_segs: list of (frames, 4, h, w) tensors (RFBG output, one per segment).
    sfbg_model: loaded SFBG.
    identity: (4, h, w) identity latent.
    silent_mask: boolean (n_seg,) — which segments are silent.
    strength: <1.0 anchors generation on the RFBG output for that segment (SDEdit,
        the way RFBG itself is run); 1.0 generates from pure noise.

    For each silent segment, searches outward for the nearest non-silent segment
    to use as prev/next conditioning (ctx_frames frames extracted from it). If no
    talking neighbour exists on one side, repeats the available one.

    Returns the same list with silent entries replaced.
    """
    import torch

    dev = next(sfbg_model.parameters()).device
    n_seg = len(refined_segs)
    identity = identity.unsqueeze(0).to(dev)
    out = list(refined_segs)

    def _ctx_from_seg(k, direction):
        """Extract ctx_frames latents from segment k, padded if shorter."""
        seg = refined_segs[k]
        if direction > 0:  # next: take first ctx_frames
            ctx = seg[:ctx_frames]
        else:  # prev: take last ctx_frames
            ctx = seg[-ctx_frames:]
        if ctx.shape[0] < ctx_frames:
            reps = -(-ctx_frames // ctx.shape[0])
            ctx = ctx.repeat(reps, 1, 1, 1)[:ctx_frames]
        return ctx

    def _find_talking(k, direction):
        """Nearest non-silent segment index in given direction, or None."""
        idx = k + direction
        while 0 <= idx < n_seg:
            if not silent_mask[idx]:
                return idx
            idx += direction
        return None

    with torch.no_grad():
        for k in range(n_seg):
            if not silent_mask[k]:
                continue
            prev_idx = _find_talking(k, -1)
            next_idx = _find_talking(k, +1)

            # Build conditioning contexts
            if prev_idx is not None:
                prev_ctx = _ctx_from_seg(prev_idx, -1).unsqueeze(0).to(dev)
            elif next_idx is not None:
                prev_ctx = _ctx_from_seg(next_idx, +1).unsqueeze(0).to(dev)  # fallback
            else:
                # No talking segments at all — skip SFBG, keep RFBG output
                continue

            if next_idx is not None:
                next_ctx = _ctx_from_seg(next_idx, +1).unsqueeze(0).to(dev)
            else:
                next_ctx = _ctx_from_seg(prev_idx, -1).unsqueeze(0).to(dev)  # fallback

            num_frames = refined_segs[k].shape[0]
            init = refined_segs[k].unsqueeze(0).to(dev) if strength < 1.0 else None
            generated = sfbg_model.sample(prev_ctx, next_ctx, identity, num_frames,
                                          init=init, strength=strength)
            out[k] = generated[0].float().cpu()

    return out


def refine_latents(model, lat_s, lat_p, identity, frames: int = 16,
                   strength: float = 0.5, tcr_window: int = 4, guidance: float = 1.0,
                   sfbg_model=None, audio_path=None, sfbg_ctx_frames: int = 8,
                   sfbg_strength: float = 0.3):
    """Segment-wise RFBG refinement of one speaker's latent video.

    lat_s, lat_p: (F, 4, h, w) latents of the speaker and the partner.
    identity: (4, h, w) identity latent. Returns (F', 4, h, w) refined latents
    (F' = full segments only) with TCR-blended boundaries.
    guidance>1 = classifier-free guidance on the partner conditioning.
    sfbg_model + audio_path: if both provided, silent segments are replaced by
    SFBG-generated listening-face latents after RFBG and before TCR (release order;
    the paper generates silent segments with SFBG before RFBG refines them — see
    DOCS.md).
    """
    import torch

    orig_len = lat_s.shape[0]
    assert orig_len >= 1, "empty video"
    # Ceil, not floor: floor division silently dropped up to `frames-1` trailing
    # frames (e.g. 10 of 250 at frames=16), desyncing the refined video's frame
    # count from its untouched audio track for every clip in this pipeline (a fixed
    # 10s/250-frame clip length never divides evenly by 16). Pad the trailing
    # partial segment by repeating the last real frame, refine it like any other
    # full segment, then trim the padding back off the final output below so the
    # returned length always equals the input length.
    n_seg = -(-orig_len // frames)
    padded_len = n_seg * frames
    if lat_s.shape[0] < padded_len:
        lat_s = torch.cat([lat_s, lat_s[-1:].expand(padded_len - lat_s.shape[0], -1, -1, -1)], dim=0)
    if lat_p.shape[0] < padded_len:
        # The partner window below (lo:hi) assumes the partner covers the same
        # range as the speaker; nothing upstream asserts the two sides have equal
        # length, and `hi` was already clamped to lat_p's length while `lo` was
        # not, so a shorter partner could put lo > hi and silently hand the model
        # a 0-frame conditioning tensor. Pad defensively (repeat the partner's last
        # real frame) so conditioning always degrades gracefully instead.
        lat_p = torch.cat([lat_p, lat_p[-1:].expand(padded_len - lat_p.shape[0], -1, -1, -1)], dim=0)
    dev = next(model.parameters()).device
    identity = identity.unsqueeze(0).to(dev)
    refined = []
    with torch.no_grad():
        for k in range(n_seg):
            init = lat_s[k * frames : (k + 1) * frames].unsqueeze(0).to(dev)
            lo = max(0, (k - 1) * frames)
            hi = min(lat_p.shape[0], (k + 2) * frames)
            partner = lat_p[lo:hi].unsqueeze(0).to(dev)  # 3-turn-style window (k-1:k+1)
            out = model.sample(init, partner, None, None, identity, strength=strength,
                               guidance_weight=guidance)
            refined.append(out[0].float().cpu())

    # SFBG: replace silent segments with listening-face generation. Release order is
    # RFBG -> SFBG -> TCR (the paper orders SFBG -> RFBG -> TCR, feeding SFBG's output
    # into RFBG; see DOCS.md). Running after RFBG means the
    # talking-segment context SFBG conditions on is already RFBG-refined.
    if sfbg_model is not None and audio_path is not None:
        if not os.path.exists(audio_path):
            print(f"  SFBG: audio not found at {audio_path}, skipping")
        else:
            orig_frames = orig_len  # use original frame count, NOT padded lat_s.shape[0]
            silent_mask = _silent_segment_mask(audio_path, orig_frames,
                                               frames_per_seg=frames)
            n_silent = int(silent_mask.sum())
            if n_silent > 0:
                mode = (f"SDEdit strength={sfbg_strength}" if sfbg_strength < 1.0
                        else "from noise")
                print(f"  SFBG: replacing {n_silent}/{len(silent_mask)} silent segments ({mode})")
                refined = apply_sfbg(refined, sfbg_model, identity[0] if identity.dim() == 4 else identity,
                                     silent_mask, frames=frames, ctx_frames=sfbg_ctx_frames,
                                     strength=sfbg_strength)
            else:
                print("  SFBG: no silent segments found, skipping")

    blended = apply_tcr([r.numpy() for r in refined], window=min(tcr_window, frames))
    import torch as _t

    return _t.from_numpy(np.concatenate(blended, axis=0))[:orig_len]


def main() -> None:
    import torch

    from chat.ifbg.bridge import decode_video, encode_video, load_vae

    ap = argparse.ArgumentParser()
    ap.add_argument("--x1-dir", required=True)
    ap.add_argument("--rfbg-ckpt", required=True)
    ap.add_argument("--vae", required=True)
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--strength", type=float, default=0.5)
    ap.add_argument("--tcr-window", type=int, default=4)
    ap.add_argument("--sides", nargs="+", default=["i", "j"])
    ap.add_argument("--weights", choices=["auto", "raw", "ema"], default="auto",
                    help="which stored weights to load from the checkpoint (default auto = EMA if present)")
    ap.add_argument("--out-suffix", default="",
                    help="append to output filenames: video_refined_{side}{suffix}.mp4 (avoids clobbering)")
    ap.add_argument("--guidance", type=float, default=1.0,
                    help="classifier-free guidance weight on the partner conditioning (1.0 = off; "
                         ">1 amplifies the partner's responsive effect, doubles sampling cost)")
    ap.add_argument("--sfbg-ckpt", default=None,
                    help="SFBG checkpoint for silent-segment generation (paper Sec. 4.3, SilentDiff). "
                         "When provided, silent segments are replaced by SFBG output after RFBG and "
                         "before TCR. Requires audio tracks at {x1-dir}/track_{side}.wav.")
    ap.add_argument("--sfbg-ctx-frames", type=int, default=8,
                    help="number of context frames from each neighbouring talking segment for "
                         "SFBG conditioning (default 8, matching the pretrained model)")
    ap.add_argument("--sfbg-strength", type=float, default=0.3,
                    help="SFBG sampling strength, only read when --sfbg-ckpt is given. Below 1.0 "
                         "the block anchors on the RFBG output for that segment (SDEdit), the way "
                         "RFBG itself is run; 1.0 generates each silent segment from pure noise "
                         "instead. The released 20k-step SFBG checkpoint cannot generate a "
                         "decodable segment from noise: at 1.0 a 10-side pilot measured +128 FID "
                         "and the segments decode to colour noise rather than faces. The default "
                         "is therefore 0.3, the value every reported SFBG number used; 0.5 was "
                         "measurably blurrier. Raise it only with a stronger checkpoint.")
    ap.add_argument("--seed", type=int, default=None,
                    help="seed the RNG that drives RFBG/SFBG diffusion sampling. Both draw "
                         "noise with torch.randn, so without this two runs of the same command "
                         "produce different video. Set it to make a run reproducible; leave it "
                         "unset for the previous non-deterministic behaviour.")
    args = ap.parse_args()

    if args.seed is not None:
        # Sampling calls torch.randn / randn_like on the default generator, so seeding
        # it (torch is imported at the top of main) is what makes a run repeatable.
        # Not a claim of bit-exactness across machines: cuDNN kernel selection and
        # GPU model still vary the last digits.
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        print(f"seed={args.seed} (same-machine reproducible; not bit-exact across GPUs)")

    import av
    from PIL import Image

    def read_frames(path):
        with av.open(path) as c:
            fr = [f.to_ndarray(format="rgb24") for f in c.decode(c.streams.video[0])]
        arr = np.stack(fr).astype(np.float32) / 255.0
        return torch.from_numpy(arr.transpose(0, 3, 1, 2))

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_rfbg(args.rfbg_ckpt, dev, weights=args.weights)
    vae = load_vae(args.vae, device=dev)
    sfbg_model = None
    if args.sfbg_ckpt:
        sfbg_model = load_sfbg(args.sfbg_ckpt, dev, weights=args.weights)
        print(f"SFBG loaded from {args.sfbg_ckpt}")
    out_dir = os.path.abspath(args.x1_dir)

    # encode every side we output PLUS its partner (refine conditions on the partner),
    # so `--sides j` still has i available as the conditioning window.
    partner_of = {"i": "j", "j": "i"}
    need = set(args.sides) | {partner_of[s] for s in args.sides}
    lats = {}
    for side in sorted(need):
        frames = read_frames(os.path.join(out_dir, f"video_{side}.mp4"))
        lats[side] = encode_video(vae, frames).float().cpu()
        print(f"{side}: {tuple(lats[side].shape)} latents")

    for side in args.sides:
        other = partner_of[side]
        ident_img = np.array(Image.open(os.path.join(out_dir, f"identity_{side}.png")).convert("RGB"))
        ident = torch.from_numpy(ident_img.astype(np.float32) / 255.0).permute(2, 0, 1)
        ident_lat = encode_video(vae, ident.unsqueeze(0)).float().cpu()[0]
        refined = refine_latents(model, lats[side], lats[other], ident_lat,
                                 frames=args.frames, strength=args.strength,
                                 tcr_window=args.tcr_window, guidance=args.guidance,
                                 sfbg_model=sfbg_model,
                                 audio_path=os.path.join(out_dir, f"track_{side}.wav"),
                                 sfbg_ctx_frames=args.sfbg_ctx_frames,
                                 sfbg_strength=args.sfbg_strength)
        pixels = decode_video(vae, refined).float().cpu()
        # write mp4 (video only; compose re-muxes audio)
        import av as _av

        # Write to a temp path and rename into place atomically: if this process is
        # killed mid-encode (SLURM preemption, OOM, ctrl-C), a partial/corrupt file
        # never lands at the final path, so a caller's "does video_refined_*.mp4
        # exist" completeness check (e.g. refine_eval_batch.sh) can't be fooled into
        # treating a truncated file as done.
        dst = os.path.join(out_dir, f"video_refined_{side}{args.out_suffix}.mp4")
        tmp_dst = dst + ".tmp"
        # av.open infers the container format from the path's extension when `format`
        # is omitted; ".tmp" isn't a format it recognises ("Could not determine output
        # format"), so it must be passed explicitly here now that the temp path no
        # longer ends in ".mp4".
        container = _av.open(tmp_dst, "w", format="mp4")
        stream = container.add_stream("libx264", rate=25)
        stream.width, stream.height = pixels.shape[-1], pixels.shape[-2]
        stream.pix_fmt = "yuv420p"
        for fr in (pixels.numpy().transpose(0, 2, 3, 1) * 255).clip(0, 255).astype(np.uint8):
            for packet in stream.encode(_av.VideoFrame.from_ndarray(fr, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        container.close()
        os.replace(tmp_dst, dst)
        print(f"saved {dst} ({pixels.shape[0]} frames)")


if __name__ == "__main__":
    main()
