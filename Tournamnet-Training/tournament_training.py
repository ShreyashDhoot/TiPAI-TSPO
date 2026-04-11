#!/usr/bin/env python3
"""Tournament policy model and continual training utilities."""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class PolicyBounds:
	cfg_min: float = 1.0
	cfg_max: float = 20.0
	dilation_min: float = 1.0
	dilation_max: float = 48.0
	jitter_min: float = 0.0
	jitter_max: float = 0.8
	threshold_min: float = 0.1
	threshold_max: float = 0.9
	strength_min: float = 0.2
	strength_max: float = 0.99
	steps_min: float = 10.0
	steps_max: float = 60.0


class TournamentPolicyMLP(torch.nn.Module):
	"""Maps intervention features to normalized parameter predictions in [0, 1]."""

	def __init__(self, in_features: int = 16, out_features: int = 6) -> None:
		super().__init__()
		self.net = torch.nn.Sequential(
			torch.nn.Linear(in_features, 128),
			torch.nn.SiLU(),
			torch.nn.Linear(128, 128),
			torch.nn.SiLU(),
			torch.nn.Linear(128, out_features),
			torch.nn.Sigmoid(),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.net(x)


class OnlineTournamentTrainer:
	"""Handles warmup, online training, and persistence for tournament policy."""

	def __init__(
		self,
		device: str,
		model_path: str,
		state_path: str,
		loss_log_path: str,
		warmup_prompts: int = 350,
		learning_rate: float = 1e-3,
		replay_capacity: int = 20000,
		batch_size: int = 64,
		feature_dim: int = 16,
		output_dim: int = 6,
	) -> None:
		self.device = device
		self.model_path = model_path
		self.state_path = state_path
		self.loss_log_path = loss_log_path
		self.warmup_prompts = int(max(1, warmup_prompts))
		self.replay_capacity = int(max(128, replay_capacity))
		self.batch_size = int(max(1, batch_size))
		self.bounds = PolicyBounds()

		self.model = TournamentPolicyMLP(in_features=feature_dim, out_features=output_dim).to(self.device)
		self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)

		self.prompts_seen = 0
		self.train_steps = 0
		self.replay: List[Dict] = []

		self._load_state()
		self._load_model_if_exists()

	def _load_model_if_exists(self) -> None:
		if os.path.exists(self.model_path):
			state = torch.load(self.model_path, map_location=self.device)
			self.model.load_state_dict(state)

	def _load_state(self) -> None:
		if not os.path.exists(self.state_path):
			return
		with open(self.state_path, "r", encoding="utf-8") as f:
			state = json.load(f)
		self.prompts_seen = int(state.get("prompts_seen", 0))
		self.train_steps = int(state.get("train_steps", 0))

	def _save_state(self) -> None:
		os.makedirs(os.path.dirname(self.state_path) or ".", exist_ok=True)
		with open(self.state_path, "w", encoding="utf-8") as f:
			json.dump(
				{
					"prompts_seen": self.prompts_seen,
					"train_steps": self.train_steps,
					"warmup_prompts": self.warmup_prompts,
					"replay_size": len(self.replay),
				},
				f,
				indent=2,
			)

	def should_use_policy(self) -> bool:
		return self.prompts_seen >= self.warmup_prompts

	def complete_prompt(self) -> None:
		self.prompts_seen += 1
		self._save_state()

	def normalized_to_params(self, pred: np.ndarray, seed: int) -> Dict:
		b = self.bounds
		pred = np.clip(pred.astype(np.float32), 0.0, 1.0)
		return {
			"cfg": float(b.cfg_min + pred[0] * (b.cfg_max - b.cfg_min)),
			"dilation": int(round(b.dilation_min + pred[1] * (b.dilation_max - b.dilation_min))),
			"noise_jitter": float(b.jitter_min + pred[2] * (b.jitter_max - b.jitter_min)),
			"mask_threshold": float(b.threshold_min + pred[3] * (b.threshold_max - b.threshold_min)),
			"strength": float(b.strength_min + pred[4] * (b.strength_max - b.strength_min)),
			"steps": int(round(b.steps_min + pred[5] * (b.steps_max - b.steps_min))),
			"seed": int(seed),
		}

	def params_to_normalized(self, params: Dict) -> np.ndarray:
		b = self.bounds

		def _norm(v: float, lo: float, hi: float) -> float:
			return float(np.clip((float(v) - lo) / (hi - lo + 1e-8), 0.0, 1.0))

		return np.array(
			[
				_norm(params["cfg"], b.cfg_min, b.cfg_max),
				_norm(params["dilation"], b.dilation_min, b.dilation_max),
				_norm(params["noise_jitter"], b.jitter_min, b.jitter_max),
				_norm(params["mask_threshold"], b.threshold_min, b.threshold_max),
				_norm(params["strength"], b.strength_min, b.strength_max),
				_norm(params["steps"], b.steps_min, b.steps_max),
			],
			dtype=np.float32,
		)

	def predict_base_params(self, features: np.ndarray, seed: int) -> Dict:
		self.model.eval()
		x = torch.from_numpy(features.astype(np.float32)).unsqueeze(0).to(self.device)
		with torch.no_grad():
			pred = self.model(x).squeeze(0).cpu().numpy()
		return self.normalized_to_params(pred, seed=seed)

	def add_intervention_example(self, features: np.ndarray, candidates: List[Dict]) -> None:
		if not candidates:
			return

		scores = np.array([float(c["score"]) for c in candidates], dtype=np.float32)
		winner_idx = int(np.argmax(scores))
		winner_params = candidates[winner_idx]["params"]

		cand_param_norm = np.stack([self.params_to_normalized(c["params"]) for c in candidates]).astype(np.float32)
		sample = {
			"x": features.astype(np.float32),
			"target": self.params_to_normalized(winner_params),
			"cand_params": cand_param_norm,
			"cand_scores": scores,
		}

		self.replay.append(sample)
		if len(self.replay) > self.replay_capacity:
			self.replay = self.replay[-self.replay_capacity :]

	def train_online(self, steps: int = 1, ranking_weight: float = 0.25) -> Dict:
		if len(self.replay) == 0 or steps <= 0:
			return {
				"train_steps": self.train_steps,
				"replay_size": len(self.replay),
				"loss_total": None,
				"loss_regression": None,
				"loss_ranking": None,
			}

		self.model.train()
		total_losses: List[float] = []
		reg_losses: List[float] = []
		rank_losses: List[float] = []

		for _ in range(steps):
			batch = random.sample(self.replay, k=min(self.batch_size, len(self.replay)))

			x = torch.tensor(np.stack([s["x"] for s in batch]), dtype=torch.float32, device=self.device)
			y = torch.tensor(np.stack([s["target"] for s in batch]), dtype=torch.float32, device=self.device)
			pred = self.model(x)

			reg_loss = F.smooth_l1_loss(pred, y)

			rank_loss_acc = []
			for b_idx, sample in enumerate(batch):
				cand_params = torch.tensor(sample["cand_params"], dtype=torch.float32, device=self.device)
				cand_scores = torch.tensor(sample["cand_scores"], dtype=torch.float32, device=self.device)
				pred_b = pred[b_idx].unsqueeze(0)

				# Candidates closer to predicted params should receive higher preference.
				pred_logits = -torch.sum((cand_params - pred_b) ** 2, dim=1)
				target_prob = torch.softmax(cand_scores, dim=0)
				rank_loss = F.kl_div(
					torch.log_softmax(pred_logits, dim=0),
					target_prob,
					reduction="batchmean",
				)
				rank_loss_acc.append(rank_loss)

			rank_loss_mean = torch.stack(rank_loss_acc).mean() if rank_loss_acc else torch.tensor(0.0, device=self.device)
			loss = reg_loss + float(ranking_weight) * rank_loss_mean

			self.optimizer.zero_grad(set_to_none=True)
			loss.backward()
			torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
			self.optimizer.step()

			self.train_steps += 1
			total_losses.append(float(loss.item()))
			reg_losses.append(float(reg_loss.item()))
			rank_losses.append(float(rank_loss_mean.item()))

		stats = {
			"train_steps": self.train_steps,
			"replay_size": len(self.replay),
			"loss_total": float(np.mean(total_losses)),
			"loss_regression": float(np.mean(reg_losses)),
			"loss_ranking": float(np.mean(rank_losses)),
		}

		self._save_checkpoint_and_logs(stats)
		return stats

	def _save_checkpoint_and_logs(self, stats: Dict) -> None:
		os.makedirs(os.path.dirname(self.model_path) or ".", exist_ok=True)
		torch.save(self.model.state_dict(), self.model_path)
		self._save_state()

		os.makedirs(os.path.dirname(self.loss_log_path) or ".", exist_ok=True)
		with open(self.loss_log_path, "a", encoding="utf-8") as f:
			f.write(
				json.dumps(
					{
						"prompts_seen": self.prompts_seen,
						"train_steps": stats["train_steps"],
						"replay_size": stats["replay_size"],
						"loss_total": stats["loss_total"],
						"loss_regression": stats["loss_regression"],
						"loss_ranking": stats["loss_ranking"],
					}
				)
				+ "\n"
			)
