#!/usr/bin/env python3
"""Stable-diffusion in-loop tournament runner for Tournamnet-Training."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
from datasets import load_dataset

from auditor import audit_image
from inpainter import InpaintConfig, TournamentInpainter, heatmap_to_binary_mask
from sd_helpers import auto_device, decode_latents_to_pil, encode_prompt, mask_to_latent_space, pil_to_latents
from tournament_training import OnlineTournamentTrainer


@dataclass
class CandidateParams:
    cfg: float
    dilation: int
    noise_jitter: float
    mask_threshold: float
    strength: float
    steps: int
    seed: int


def _load_prompt_state(state_file: str) -> Dict:
    if not os.path.exists(state_file):
        return {"prompt_index": 0}
    with open(state_file, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_prompt_state(state_file: str, state: Dict) -> None:
    os.makedirs(os.path.dirname(state_file) or ".", exist_ok=True)
    with open(state_file, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def resolve_prompt(args: argparse.Namespace) -> tuple[str, Dict]:
    '''
    Resolves the prompt repo 
    -> takes in repo name and split form the cli and loads the HF dataset 
    returns 
    prompt_text (str): The text for particular prompt 
    next_state (dict): repo,split,column,next prompt index,last prompt 
    '''
    if args.prompt:
        return args.prompt, {"source": "cli", "prompt_index": None, "prompt_repo": None}

    if not args.prompt_repo:
        raise ValueError("Provide either --prompt or --prompt-repo.")

    dataset = load_dataset(args.prompt_repo, split=args.prompt_split)
    if args.prompt_column not in dataset.column_names:
        raise ValueError(
            f"Prompt column '{args.prompt_column}' not found in dataset columns: {dataset.column_names}"
        )

    state_file = args.prompt_state_file or os.path.join(args.output_dir, "prompt_cursor.json")
    state = _load_prompt_state(state_file)
    prompt_index = int(args.prompt_index) if args.prompt_index is not None else int(state.get("prompt_index", 0))
    if len(dataset) == 0:
        raise ValueError(f"Dataset '{args.prompt_repo}' split '{args.prompt_split}' is empty.")

    row_index = prompt_index % len(dataset)
    prompt_text = str(dataset[row_index][args.prompt_column])
    next_state = {
        "prompt_repo": args.prompt_repo,
        "prompt_split": args.prompt_split,
        "prompt_column": args.prompt_column,
        "prompt_index": prompt_index + 1,
        "row_index": row_index,
        "last_prompt": prompt_text,
    }
    _save_prompt_state(state_file, next_state)
    return prompt_text, next_state


def sample_candidate_params(rng: np.random.Generator) -> CandidateParams:
    return CandidateParams(
        cfg=float(np.clip(rng.normal(7.5, 1.5), 1.0, 20.0)),
        dilation=int(np.clip(np.round(rng.normal(8.0, 3.0)), 1, 48)),
        noise_jitter=float(np.clip(rng.normal(0.12, 0.08), 0.0, 0.8)),
        mask_threshold=float(np.clip(rng.normal(0.55, 0.12), 0.1, 0.9)),
        strength=float(np.clip(rng.normal(0.78, 0.10), 0.2, 0.99)),
        steps=int(np.clip(np.round(rng.normal(30.0, 6.0)), 10, 60)),
        seed=int(rng.integers(0, 2**31 - 1)),
    )

def tournament_score(audit: Dict, params: CandidateParams) -> float:
    adv_prob = float(audit.get("adversarial_probability", 1.0))
    seam = float(audit.get("seam_quality_assessment", 0.0))
    faith = float(audit.get("text_faithfulness_score", 0.0))
    safe_prob = float(audit.get("category_probabilities", {}).get("Safe", 0.0))

    # Mild regularization so policy does not drift to extreme edits.
    reg = 0.02 * abs(params.cfg - 7.5) / 7.5 + 0.02 * params.noise_jitter
    score = 0.55 * (1.0 - adv_prob) + 0.20 * seam + 0.20 * faith + 0.05 * safe_prob - reg
    if audit.get("is_adversarial") is False:
        score += 0.05
    return float(score)


def build_policy_features(audit: Dict, timestep_norm: float, latents: torch.Tensor) -> np.ndarray:
    probs = audit.get("category_probabilities", {})
    class_vec = np.array([
        float(probs.get("Safe", 0.0)),
        float(probs.get("Violence", 0.0)),
        float(probs.get("Sexual", 0.0)),
        float(probs.get("Illegal Activity", 0.0)),
        float(probs.get("Disturbing", 0.0)),
    ], dtype=np.float32)

    latent_mean = float(latents.mean().item())
    latent_std = float(latents.std().item())
    latent_norm = float(latents.norm().item() / max(1, latents.numel()))
    entropy = float(-np.sum(class_vec * np.log(np.clip(class_vec, 1e-8, 1.0))))

    features = np.array([
        float(timestep_norm),
        float(audit.get("adversarial_probability", 1.0)),
        float(audit.get("relative_adversary_score", 0.0)),
        float(audit.get("seam_quality_assessment", 0.0)),
        float(audit.get("text_faithfulness_score", 0.0)),
        float(audit.get("is_adversarial", True)),
        latent_mean,
        latent_std,
        latent_norm,
        entropy,
        class_vec[0],
        class_vec[1],
        class_vec[2],
        class_vec[3],
        class_vec[4],
        1.0,
    ], dtype=np.float32)
    return features


def recommend_candidates(
    trainer: OnlineTournamentTrainer,
    use_policy: bool,
    features: np.ndarray,
    rng: np.random.Generator,
    num_candidates: int,
    exploration_std: float,
) -> List[CandidateParams]:
    if not use_policy:
        return [sample_candidate_params(rng) for _ in range(num_candidates)]

    base_pred = trainer.predict_base_params(features=features, seed=int(rng.integers(0, 2**31 - 1)))
    base = CandidateParams(**base_pred)

    candidates: List[CandidateParams] = []
    for _ in range(num_candidates):
        candidates.append(
            CandidateParams(
                cfg=float(np.clip(base.cfg + rng.normal(0.0, 2.0 * exploration_std), 1.0, 20.0)),
                dilation=int(np.clip(round(base.dilation + rng.normal(0.0, 5.0 * exploration_std)), 1, 48)),
                noise_jitter=float(np.clip(base.noise_jitter + rng.normal(0.0, 0.1 * exploration_std), 0.0, 0.8)),
                mask_threshold=float(np.clip(base.mask_threshold + rng.normal(0.0, 0.1 * exploration_std), 0.1, 0.9)),
                strength=float(np.clip(base.strength + rng.normal(0.0, 0.1 * exploration_std), 0.2, 0.99)),
                steps=int(np.clip(round(base.steps + rng.normal(0.0, 8.0 * exploration_std)), 10, 60)),
                seed=int(rng.integers(0, 2**31 - 1)),
            )
        )
    return candidates


def run_pipeline(args: argparse.Namespace) -> Dict:
    '''
    Runs the whole pipeline 
    prompt -> diffusion denoising -> auditor -> inpaint candidates -> tournament -> winner -> continual learning -> end generation 
    '''
    os.makedirs(args.output_dir, exist_ok=True)
    candidates_dir = os.path.join(args.output_dir, "candidates")
    os.makedirs(candidates_dir, exist_ok=True)

    device = auto_device()
    dtype = torch.float16 if device == "cuda" else torch.float32

    #loading the stable diffusion pipeline 
    sd_pipe = StableDiffusionPipeline.from_pretrained(args.sd_model, torch_dtype=dtype).to(device)
    sd_pipe.safety_checker = None
    sd_pipe.set_progress_bar_config(disable=True)

    inpainter = TournamentInpainter(device=device, dtype=dtype)
    rng = np.random.default_rng(args.seed)
    torch_generator = torch.Generator(device=device).manual_seed(args.seed)

    #calls tournament policy MLP weights 
    policy_path = args.policy_weights or os.path.join(args.output_dir, "policy_mlp.pt")
    #calls tournament trainer for continual learning 
    trainer = OnlineTournamentTrainer(
        device=device,
        model_path=policy_path,
        state_path=os.path.join(args.output_dir, "policy_state.json"),
        loss_log_path=os.path.join(args.output_dir, "policy_loss_log.jsonl"),
        warmup_prompts=args.warmup_prompts,
        learning_rate=args.policy_lr,
        batch_size=args.policy_batch_size,
    )
    use_policy = trainer.should_use_policy()

    #encodes the text into latentys 
    text_embeddings = encode_prompt(sd_pipe, args.prompt, args.negative_prompt, device)
    latent_h = args.height // 8
    latent_w = args.width // 8
    latents = torch.randn((1, 4, latent_h, latent_w), device=device, generator=torch_generator, dtype=dtype)
    latents = latents * sd_pipe.scheduler.init_noise_sigma

    sd_pipe.scheduler.set_timesteps(args.num_inference_steps, device=device)
    timesteps = sd_pipe.scheduler.timesteps

    interventions: List[Dict] = []
    tmp_img_path = os.path.join(args.output_dir, "_tmp_audit.png")

    for step_idx, t in enumerate(timesteps):
        latent_input = torch.cat([latents] * 2)
        latent_input = sd_pipe.scheduler.scale_model_input(latent_input, t)

        with torch.no_grad():
            noise_pred = sd_pipe.unet(latent_input, t, encoder_hidden_states=text_embeddings).sample
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_text - noise_pred_uncond)
        latents = sd_pipe.scheduler.step(noise_pred, t, latents).prev_sample

        current_image = decode_latents_to_pil(sd_pipe, latents)
        current_image.save(tmp_img_path)

        timestep_norm = float(step_idx / max(1, args.num_inference_steps - 1))
        audit = audit_image(
            model_path=args.auditor_model,
            image_path=tmp_img_path,
            prompt=args.prompt,
            vocab_path=args.auditor_vocab,
            return_heatmaps=True,
            timestep_value=timestep_norm,
        )

        if float(audit.get("adversarial_probability", 1.0)) < args.intervention_threshold:
            continue

        if "adversarial_heatmap" not in audit:
            continue

        features = build_policy_features(audit, timestep_norm, latents)
        params_list = recommend_candidates(
            trainer=trainer,
            use_policy=use_policy,
            features=features,
            rng=rng,
            num_candidates=args.num_candidates,
            exploration_std=args.exploration_std,
        )

        best_entry: Optional[Dict] = None
        all_candidate_entries: List[Dict] = []
        for c_idx, params in enumerate(params_list):
            mask = heatmap_to_binary_mask(
                audit["adversarial_heatmap"],
                threshold=params.mask_threshold,
                dilation=params.dilation,
            )
            config = InpaintConfig(
                guidance_scale=params.cfg,
                strength=params.strength,
                num_inference_steps=params.steps,
                noise_jitter=params.noise_jitter,
            )

            candidate_img = inpainter.inpaint(
                image=current_image,
                mask_image=mask,
                prompt=args.prompt,
                config=config,
                seed=params.seed,
                negative_prompt=args.negative_prompt,
            )

            candidate_path = os.path.join(candidates_dir, f"step_{step_idx:03d}_cand_{c_idx:02d}.png")
            candidate_img.save(candidate_path)

            cand_audit = audit_image(
                model_path=args.auditor_model,
                image_path=candidate_path,
                prompt=args.prompt,
                vocab_path=args.auditor_vocab,
                return_heatmaps=False,
                timestep_value=timestep_norm,
            )
            cand_score = tournament_score(cand_audit, params)

            entry = {
                "candidate_index": c_idx,
                "params": asdict(params),
                "audit": cand_audit,
                "score": cand_score,
                "image_path": candidate_path,
                "mask_path": None,
            }
            all_candidate_entries.append(entry)
            if best_entry is None or entry["score"] > best_entry["score"]:
                best_entry = entry

        if best_entry is None:
            continue

        winner_img = Image.open(best_entry["image_path"]).convert("RGB")
        winner_params = CandidateParams(**best_entry["params"])
        winner_mask = heatmap_to_binary_mask(
            audit["adversarial_heatmap"],
            threshold=winner_params.mask_threshold,
            dilation=winner_params.dilation,
        )

        z_fixed = pil_to_latents(sd_pipe, winner_img, device=device, dtype=dtype)
        noise = torch.randn(z_fixed.shape, device=device, generator=torch_generator, dtype=dtype)
        z_noisy = sd_pipe.scheduler.add_noise(z_fixed, noise, t)
        m = mask_to_latent_space(winner_mask, latents.shape, device=device, dtype=dtype)
        latents = (1.0 - m) * latents + m * z_noisy

        trainer.add_intervention_example(features=features, candidates=all_candidate_entries)
        train_stats = trainer.train_online(
            steps=args.online_train_steps,
            ranking_weight=args.ranking_weight,
        )

        interventions.append(
            {
                "step_idx": step_idx,
                "timestep_value": int(t.item()) if hasattr(t, "item") else int(t),
                "timestep_norm": timestep_norm,
                "audit_before": audit,
                "policy_features": features.tolist(),
                "winner": best_entry,
                "winner_index": int(best_entry["candidate_index"]),
                "candidates": all_candidate_entries,
                "policy_mode": "mlp" if use_policy else "random_gaussian",
                "train_stats": train_stats,
            }
        )

    trainer.complete_prompt()

    final_image = decode_latents_to_pil(sd_pipe, latents)
    final_path = os.path.join(args.output_dir, "final_image.png")
    final_image.save(final_path)

    summary = {
        "prompt": args.prompt,
        "prompt_source": args.prompt_source,
        "prompt_repo": args.prompt_repo,
        "prompt_split": args.prompt_split,
        "prompt_column": args.prompt_column,
        "prompt_index": args.prompt_state.get("prompt_index"),
        "num_inference_steps": args.num_inference_steps,
        "num_interventions": len(interventions),
        "num_candidates": args.num_candidates,
        "final_image": final_path,
        "policy_mode": "mlp" if use_policy else "random_gaussian",
        "prompts_seen": trainer.prompts_seen,
        "warmup_prompts": args.warmup_prompts,
        "policy_path": policy_path,
        "inpainter_model_path": inpainter.model_path_or_id,
        "loss_log_path": os.path.join(args.output_dir, "policy_loss_log.jsonl"),
    }
    summary_path = os.path.join(args.output_dir, "run_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    events_path = os.path.join(args.output_dir, "policy_training_events.jsonl")
    with open(events_path, "w", encoding="utf-8") as f:
        for item in interventions:
            f.write(json.dumps(item) + "\n")

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stable-diffusion in-loop tournament inpainting.")
    parser.add_argument("--prompt", type=str, default="", help="Text prompt; leave empty when using --prompt-repo")
    parser.add_argument("--prompt-repo", type=str, default="", help="Hugging Face dataset/repo containing prompts")
    parser.add_argument("--prompt-split", type=str, default="train", help="Dataset split to read prompts from")
    parser.add_argument("--prompt-column", type=str, default="prompt", help="Column name containing prompt text")
    parser.add_argument("--prompt-index", type=int, default=None, help="Optional starting prompt index")
    parser.add_argument("--prompt-state-file", type=str, default="", help="File used to persist the next prompt index")
    parser.add_argument("--negative-prompt", type=str, default="", help="Negative prompt")
    parser.add_argument("--sd-model", type=str, default="runwayml/stable-diffusion-v1-5", help="Stable diffusion base model")
    parser.add_argument("--auditor-model", type=str, default="model-weights/complete_auditor_best.pth", help="Path to auditor weights")
    parser.add_argument("--auditor-vocab", type=str, default="model-weights/vocab.json", help="Path to auditor vocab")
    parser.add_argument("--policy-weights", type=str, default="", help="Policy weights path (loads if exists, saves continually)")
    parser.add_argument("--num-inference-steps", type=int, default=50, help="Diffusion timesteps")
    parser.add_argument("--guidance-scale", type=float, default=7.5, help="CFG for base diffusion")
    parser.add_argument("--num-candidates", type=int, default=8, help="Candidates per intervention")
    parser.add_argument("--intervention-threshold", type=float, default=0.5, help="Audit threshold to trigger tournament")
    parser.add_argument("--exploration-std", type=float, default=1.0, help="Exploration around MLP recommendations")
    parser.add_argument("--warmup-prompts", type=int, default=350, help="Use random Gaussian policy for first N prompts")
    parser.add_argument("--policy-lr", type=float, default=1e-3, help="Learning rate for continual policy training")
    parser.add_argument("--policy-batch-size", type=int, default=64, help="Batch size for continual policy updates")
    parser.add_argument("--online-train-steps", type=int, default=1, help="Optimizer steps per intervention")
    parser.add_argument("--ranking-weight", type=float, default=0.25, help="Weight for ranking term in policy loss")
    parser.add_argument("--height", type=int, default=512, help="Output height")
    parser.add_argument("--width", type=int, default=512, help="Output width")
    parser.add_argument("--seed", type=int, default=42, help="Global seed")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")
    return parser.parse_args()


def main() -> None:
    #parses terminal arguments
    args = parse_args()
    if args.prompt_repo:
        args.prompt, args.prompt_state = resolve_prompt(args)
        args.prompt_source = "huggingface_repo"
    else:
        args.prompt_state = {"prompt_index": None}
        args.prompt_source = "cli"
    summary = run_pipeline(args)

    print("=" * 72)
    print("Pipeline Completed")
    print("=" * 72)
    print(f"Final image: {summary['final_image']}")
    print(f"Interventions: {summary['num_interventions']}")
    print(f"Candidates per intervention: {summary['num_candidates']}")
    print("=" * 72)


if __name__ == "__main__":
    main()