#!/usr/bin/env python3
"""Stable Diffusion helper utilities for tournament pipeline code."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline
from PIL import Image

if TYPE_CHECKING:
    from typing import Tuple


def auto_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def encode_prompt(pipe: StableDiffusionPipeline, prompt: str, negative_prompt: str, device: str):
    try:
        return pipe._encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=negative_prompt,
        )
    except TypeError:
        return pipe._encode_prompt(prompt, device, 1, True, negative_prompt)


def decode_latents_to_pil(pipe: StableDiffusionPipeline, latents: torch.Tensor) -> Image.Image:
    if hasattr(pipe, "decode_latents"):
        image_np = pipe.decode_latents(latents.detach())
        return pipe.numpy_to_pil(image_np)[0]

    scale = getattr(pipe.vae.config, "scaling_factor", 0.18215)
    with torch.no_grad():
        decoded = pipe.vae.decode(latents / scale, return_dict=False)[0]
    decoded = (decoded / 2 + 0.5).clamp(0, 1)
    image_np = decoded.detach().permute(0, 2, 3, 1).float().cpu().numpy()
    return pipe.numpy_to_pil(image_np)[0]


def pil_to_latents(pipe: StableDiffusionPipeline, image: Image.Image, device: str, dtype: torch.dtype) -> torch.Tensor:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)
    with torch.no_grad():
        latent_dist = pipe.vae.encode(tensor).latent_dist
        z = latent_dist.mode()
    scale = getattr(pipe.vae.config, "scaling_factor", 0.18215)
    return z * scale


def mask_to_latent_space(mask_image: Image.Image, latent_shape: torch.Size, device: str, dtype: torch.dtype) -> torch.Tensor:
    m = np.asarray(mask_image.convert("L"), dtype=np.float32) / 255.0
    m = torch.from_numpy(m).unsqueeze(0).unsqueeze(0).to(device=device, dtype=dtype)
    m = F.interpolate(m, size=(latent_shape[-2], latent_shape[-1]), mode="bilinear", align_corners=False)
    m = torch.clamp(m, 0.0, 1.0)
    return m