"""
Run Stable Diffusion with periodic audits and tournament-based inpainting.

Config (YAML) keys expected:
  model: stable-diffusion model id or local path
  diffusion_steps: int
  n_checks: int    # number of stops in last 30% where we audit
  candidates: int  # number of candidate suggestions per tournament (N)
  adv_threshold: float
  inpainter_pth: path to inpainter weights (optional)
  policy_checkpoint: path to tspo policy .pth
  state_enc_checkpoint: path to state encoder .pth (optional, only if building state here)
  device: 'cpu'|'cuda'|'auto'

This script relies on the training modules under the Tournamnet-Training folder.
It performs a diffusion denoising loop and at specified timesteps decodes latents,
audits them, runs tournament suggestions, generates inpainted candidates, scores them,
selects a winner and blends it back into the latent via the inpainter.

This is an orchestration script — tune config for your setup. It assumes the
Tournamnet-Training code is available alongside this repo.
"""

import argparse
import os
import sys
import yaml
import math
import random
from typing import List

import torch
from PIL import Image

# Ensure training package is importable
TP_TRAIN = os.path.join(os.path.dirname(__file__), "..", "Tournamnet-Training")
TP_TRAIN = os.path.abspath(TP_TRAIN)
if TP_TRAIN not in sys.path:
    sys.path.insert(0, TP_TRAIN)

from diffusers import DDIMScheduler, StableDiffusionPipeline

from src.config import DEVICE as TRAIN_DEVICE, DTYPE
from src.models import AuditorWrapper, Inpainter, StateEncoderWrapper, StateEncoder, TSPOPolicy
from src.utils import encode_prompt_safe, seam_quality_vgg, compute_guard_utility
from tournament_inference import deterministic_knob, run_inference, load_policy
from policy_inference import compute_guard_utility as local_compute_guard_utility


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def decode_latents_to_pil(pipe, latents):
    zf = latents.to(DTYPE) / pipe.vae.config.scaling_factor
    with torch.no_grad():
        img = pipe.vae.decode(zf).sample
    img = (img.float().clamp(-1, 1) + 1) / 2
    arr = (img[0].permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
    return Image.fromarray(arr)


def build_timesteps(scheduler, num_steps: int):
    scheduler.set_timesteps(num_steps)
    return list(scheduler.timesteps)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="YAML config path")
    p.add_argument("--prompt", required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    cfg = load_yaml(args.config)
    model_id = cfg.get("model", "runwayml/stable-diffusion-v1-5")
    diffusion_steps = int(cfg.get("diffusion_steps", 50))
    n_checks = int(cfg.get("n_checks", 3))
    N = int(cfg.get("candidates", 5))
    adv_threshold = float(cfg.get("adv_threshold", 0.15))
    inpainter_pth = cfg.get("inpainter_pth")
    policy_ckpt = cfg.get("policy_checkpoint")
    state_enc_ckpt = cfg.get("state_enc_checkpoint")
    device_cfg = cfg.get("device", "auto")

    if device_cfg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_cfg)

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # Load pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=DTYPE, safety_checker=None, requires_safety_checker=False,
    )
    pipe.safety_checker = None
    pipe.feature_extractor = None
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(str(device))
    pipe.enable_attention_slicing()

    # Helper components from training repo
    auditor = AuditorWrapper(cfg.get("auditor_model", "complete_auditor_best.pth"), cfg.get("auditor_vocab", "vocab.json"))
    inpainter = Inpainter(inpainter_pth=inpainter_pth)

    # State encoder (optional) — if provided and you want to build state here
    state_encoder = None
    if state_enc_ckpt:
        text_encoder = auditor.model.text_encoder
        state_enc = StateEncoder()
        state_enc.load_state_dict(torch.load(state_enc_ckpt, map_location=device))
        state_encoder = StateEncoderWrapper(encoder=state_enc, tokenizer=auditor.tokenizer, text_encoder=text_encoder)

    # Load policy weights via tournament_inference loader (keeps flexibility)
    policy = load_policy(policy_ckpt, device)

    sched = pipe.scheduler
    sched.set_timesteps(diffusion_steps)
    timesteps = list(sched.timesteps)
    total = len(timesteps)
    start_at = int(0.7 * total)

    # Choose check indices evenly across last 30%
    last_indices = list(range(start_at, total))
    if n_checks <= 0:
        check_idxs = []
    else:
        if n_checks >= len(last_indices):
            check_idxs = last_indices
        else:
            step = max(1, len(last_indices) // n_checks)
            check_idxs = [last_indices[i] for i in range(0, len(last_indices), step)][:n_checks]

    # initialize latents
    gen = torch.Generator(device=str(device)).manual_seed(args.seed)
    latents = (
        torch.randn((1, pipe.unet.config.in_channels, 64, 64), generator=gen, device=str(device), dtype=DTYPE)
        * sched.init_noise_sigma
    )

    # Text embed once
    with torch.no_grad():
        text_emb = encode_prompt_safe(pipe, args.prompt, str(device)).to(dtype=DTYPE)

    for idx, t in enumerate(timesteps):
        t_norm = t.item() / 1000.0
        lat_in = sched.scale_model_input(torch.cat([latents] * 2), t)
        with torch.no_grad():
            noise_pred = pipe.unet(lat_in, t, encoder_hidden_states=text_emb).sample
        u, c = noise_pred.chunk(2)
        noise_pred = u + 7.5 * (c - u)
        latents = sched.step(noise_pred, t, latents, generator=gen).prev_sample

        if idx in check_idxs:
            # decode and audit
            image = decode_latents_to_pil(pipe, latents)
            ctrl = auditor.audit(image, args.prompt, t_norm=t_norm)
            if ctrl["adversarial_score"] >= adv_threshold:
                # build state (if available) or fallback to using ctrl features
                if state_encoder is not None:
                    state = state_encoder.encode(args.prompt, latents.float(), ctrl["img_embed"], ctrl["mask_512"], t_norm)
                    state_vec = state.cpu()
                else:
                    # fallback: assemble minimal state-like vector not used by policy loader directly
                    state_vec = None

                # get N candidate knobsets by calling tournament inference module
                knob_sets = []
                for _ in range(N):
                    # tournament_inference.run_inference expects a policy and state; reuse that
                    out = run_inference(policy, state_vec if state_vec is not None else torch.zeros(257), num_suggestions=1, deterministic=False)
                    # run_inference returns list of dicts
                    ks = out[0]
                    knob_sets.append(ks)

                # generate candidates via inpainter
                candidates = inpainter.generate_candidates(image, ctrl["mask_512"], args.prompt, knob_sets)

                # score candidates
                utilities = []
                cand_scores = []
                cand_embeds = []
                for cand in candidates:
                    B_i = seam_quality_vgg(image, cand, ctrl.get("ring_mask"))
                    cs = auditor.audit(cand, args.prompt, t_norm=t_norm, mask=ctrl.get("mask_512"))
                    u_i = local_compute_guard_utility(cs["P_R"], cs["F_R"], B_i, t_norm)
                    utilities.append(u_i)
                    cand_scores.append(cs)
                    cand_embeds.append(cs.get("img_embed"))

                best_idx = int(max(range(len(utilities)), key=lambda i: utilities[i]))
                accepted = utilities[best_idx] > 0.0

                if accepted:
                    wk = knob_sets[best_idx]
                    z_edit = inpainter.short_ddim_inversion(candidates[best_idx], wk.get("inversion_depth", 1), t_norm)
                    latents = inpainter.blend_latents(latents.float(), z_edit, ctrl["mask_512"], t_norm).to(DTYPE)

    # finish: decode final image and save
    final_img = decode_latents_to_pil(pipe, latents)
    out_path = cfg.get("output_image", "pipeline_out.png")
    final_img.save(out_path)
    print(f"Saved final image to {out_path}")


if __name__ == "__main__":
    main()
