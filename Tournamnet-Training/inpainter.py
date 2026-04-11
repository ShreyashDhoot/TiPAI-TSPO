#!/usr/bin/env python3
"""Utilities for tournament-driven inpainting."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image


def _auto_device() -> str:
	return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class InpaintConfig:
	guidance_scale: float = 7.5
	strength: float = 0.8
	num_inference_steps: int = 30
	noise_jitter: float = 0.0


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPAINTER_CANDIDATES = (
	os.path.join(PROJECT_DIR, "model-weights", "inpainter"),
	os.path.join(PROJECT_DIR, "model-weights", "inpainter.safetensors"),
	os.path.join(PROJECT_DIR, "model-weights", "inpainter.ckpt"),
	os.path.join(PROJECT_DIR, "model-weights", "inpaint"),
	os.path.join(PROJECT_DIR, "model-weights", "inpaint.safetensors"),
	os.path.join(PROJECT_DIR, "model-weights", "inpaint.ckpt"),
)


def _resolve_default_inpainter_path() -> str:
	for path in DEFAULT_INPAINTER_CANDIDATES:
		if os.path.exists(path):
			return path
	raise FileNotFoundError(
		"No local inpainter weights found under model-weights/. "
		"Expected one of: inpainter/, inpainter.safetensors, inpainter.ckpt, inpaint/, inpaint.safetensors, inpaint.ckpt"
	)


def heatmap_to_binary_mask(
	heatmap: np.ndarray,
	threshold: float = 0.5,
	dilation: int = 8,
) -> Image.Image:
	"""Convert an auditor heatmap (H, W) in [0,1] to a dilated binary PIL mask."""
	heatmap = np.asarray(heatmap, dtype=np.float32)
	heatmap = np.clip(heatmap, 0.0, 1.0)

	mask = (heatmap >= float(threshold)).astype(np.uint8) * 255

	dilation = int(max(0, dilation))
	if dilation > 0:
		k = 2 * dilation + 1
		kernel = np.ones((k, k), np.uint8)
		mask = cv2.dilate(mask, kernel, iterations=1)

	return Image.fromarray(mask, mode="L")


class TournamentInpainter:
	"""Wrapper around StableDiffusionInpaintPipeline for tournament candidates."""

	def __init__(
		self,
		model_path_or_id: Optional[str] = None,
		device: Optional[str] = None,
		dtype: Optional[torch.dtype] = None,
	) -> None:
		self.device = device or _auto_device()
		if dtype is None:
			dtype = torch.float16 if self.device == "cuda" else torch.float32
		self.dtype = dtype
		self.model_path_or_id = model_path_or_id or _resolve_default_inpainter_path()

		self.pipe = self._load_pipeline(self.model_path_or_id)
		self.pipe = self.pipe.to(self.device)
		self.pipe.safety_checker = None
		self.pipe.set_progress_bar_config(disable=True)

	def _load_pipeline(self, model_path_or_id: str) -> StableDiffusionInpaintPipeline:
		local_exists = os.path.exists(model_path_or_id)

		if local_exists and os.path.isfile(model_path_or_id):
			return StableDiffusionInpaintPipeline.from_single_file(
				model_path_or_id,
				torch_dtype=self.dtype,
			)

		return StableDiffusionInpaintPipeline.from_pretrained(
			model_path_or_id,
			torch_dtype=self.dtype,
		)

	def inpaint(
		self,
		image: Image.Image,
		mask_image: Image.Image,
		prompt: str,
		config: InpaintConfig,
		seed: Optional[int] = None,
		negative_prompt: str = "",
	) -> Image.Image:
		generator = None
		if seed is not None:
			generator = torch.Generator(device=self.device).manual_seed(int(seed))

		init_image = image.convert("RGB")
		if config.noise_jitter > 0:
			init_image = self._apply_noise_jitter(init_image, config.noise_jitter, generator)

		output = self.pipe(
			prompt=prompt,
			negative_prompt=negative_prompt,
			image=init_image,
			mask_image=mask_image.convert("L"),
			guidance_scale=float(config.guidance_scale),
			strength=float(config.strength),
			num_inference_steps=int(config.num_inference_steps),
			generator=generator,
		)
		return output.images[0]

	@staticmethod
	def _apply_noise_jitter(
		image: Image.Image,
		jitter: float,
		generator: Optional[torch.Generator],
	) -> Image.Image:
		jitter = float(max(0.0, min(1.0, jitter)))
		img = np.asarray(image).astype(np.float32)

		if generator is None:
			noise = np.random.randn(*img.shape).astype(np.float32)
		else:
			noise = torch.randn(img.shape, generator=generator, device="cpu").numpy().astype(np.float32)

		sigma = 255.0 * jitter
		img = np.clip(img + noise * sigma, 0, 255).astype(np.uint8)
		return Image.fromarray(img, mode="RGB")