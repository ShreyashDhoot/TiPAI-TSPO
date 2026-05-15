# Alignment Methodology Evolution (Behavior-Focused)

Last updated: 2026-05-15

This write-up tracks how the alignment method evolved in terms of model behavior and training dynamics, not logging or infra changes. It is organized chronologically and then summarized as the current methodology.

## 1) Early baseline: KTO + LoRA on SD inpainting
- Started from the SD1.5 inpainting base with LoRA adapters to keep training lightweight while steering the UNet toward refusal behavior.
- Used a KTO-style objective where the trainable model is compared against a frozen reference to decide whether it should move closer to a refusal policy for unsafe content and stay faithful on safe content.
- Anchored the model to the original image manifold through reconstruction and identity penalties to prevent drift.

Behavioral intent:
- Refuse unsafe regions while preserving non-masked background.
- Avoid broad degradation of image quality on safe samples.

## 2) KTO stabilization and safety-focused shaping
- Shifted to tri-class labels (safe, nudity, violence) and began using class-specific weighting to control the relative “torque” of each unsafe category.
- Added KL-centered / baseline-centered KTO scoring so that “safe” samples define the neutral point, reducing reward drift and improving stability.
- Tightened the unsafe reward to avoid pathological exploitation (alignment hacking), including masking decisions and reward caps.
- Increased beta scaling and fixed gradient accumulation counting so the KTO signal had stronger and more consistent impact at each step.

Behavioral intent:
- Reduce oversuppression and blurry artifacts on safe images.
- Make refusal behavior stronger and more reliable on unsafe prompts.
- Stabilize training so refusal policy does not oscillate across runs.

## 3) Evaluation hardening
- Improved evaluation routines to better reflect inpainting behavior (prompt fallback, mask handling, stable inference settings).
- Added consistent per-step visual evaluation to catch spatial leakage (suppression bleeding outside the mask) and misalignment drift earlier.

Behavioral intent:
- Catch false positives (safe images being suppressed).
- Detect spatial failure modes (refusal leaking into arms/background).

## 4) Sampling and batching control
- Introduced stratified batch construction with fixed safe/nudity/violence counts per batch, ensuring every step includes balanced signals.
- This prevents long streaks of only-safe or only-unsafe batches that used to destabilize the alignment gradients.

Behavioral intent:
- Maintain steady pressure on refusal behavior without collapsing safe fidelity.
- Improve convergence consistency between runs.

## 5) From KTO to BCO (method switch)
- Replaced the KTO objective with a Binary Classifier Optimization (BCO) style objective.
- BCO uses a reward based on masked MSE difference between train and reference, then applies a BCE-style loss with per-class weights.
- Introduced a reward-shift EMA to stabilize the classification boundary between safe and unsafe rewards.

Behavioral intent:
- Make “safe vs unsafe” separation more stable and interpretable.
- Reduce overfitting to the reward baseline and make the refusal trigger more robust.

## 6) Spatial guardrails to avoid suppression leakage
- Reconstruction loss is applied mainly to safe samples to preserve background and overall fidelity.
- Identity guardrail is applied to safe samples and lightly to unsafe samples to prevent the model from over-editing unmasked regions.
- Mask weighting logic was refined to avoid the model exploiting the loss by hiding errors inside the mask.

Behavioral intent:
- Keep safe content visually intact.
- Prevent refusal from smearing into hands, faces, or background.

## 7) Multi-step alignment (closing train/infer gap)
- Added multi-step denoising unrolls periodically during training so the loss reflects multi-step inference behavior, not just a single-step proxy.
- Training now sometimes backprops through a lower-noise target to emulate actual generation steps.

Behavioral intent:
- Make refusal behavior appear at inference time, not only in single-step training diagnostics.
- Reduce mismatch between training loss and observed inference behavior.

## 8) Prompt dropping for spatial grounding
- Randomly drop text conditioning for a subset of samples, forcing the model to use spatial cues to decide where to suppress.

Behavioral intent:
- Improve spatial grounding of refusal to masked regions.
- Reduce prompt-only triggers that cause global suppression.

## 9) Iterative alignment via LoRA merge + retrain
- Added a merge script that can fold LoRA deltas into the base UNet and re-save a merged pipeline.
- This enables staged alignment: each round starts from a merged “aligned base” and trains a fresh LoRA on top.

Behavioral intent:
- Allow long-term alignment improvements without accumulating fragile LoRA stacks.
- Keep inference fast while preserving alignment gains.

## 10) Refusal SFT stage added (pipeline expansion)
- Added a separate refusal SFT training stage that teaches the model what a refusal looks like in pixel space before policy optimization.
- This provides a clearer target for the subsequent BCO alignment stage.

Behavioral intent:
- Improve the visual quality of refusals (less noise, more coherent inpainted refusal content).
- Reduce policy-stage burden by teaching the visual refusal manifold earlier.

---

# Current Methodology (as of 2026-05-15)

1) Stage 1: Refusal SFT (pixel-space supervision)
- Train LoRA on paired unsafe originals and human-approved refusal inpaints.
- Teaches the model the visual form of refusal.

2) Stage 2: BCO alignment on latent inpainting
- Frozen reference UNet provides a behavioral anchor.
- BCO reward compares masked MSE between train and reference.
- Per-class weights shape nudity vs violence torque.
- Reward shift EMA stabilizes the safe/unsafe boundary.

3) Guardrails + fidelity anchors
- Reconstruction anchors safe backgrounds.
- Identity loss protects unmasked regions (full on safe, light on unsafe).
- Full timestep training and multi-step unrolling reduce train/infer mismatch.
- Prompt dropping enforces spatial grounding of refusal.

4) Iterative alignment cycle
- Merge LoRA into base for each new round and re-train with fresh adapters.

---

# Behavioral Outcomes Targeted
- Strong refusal in masked unsafe regions.
- Minimal degradation on safe samples.
- Stable refusal without oscillation across runs.
- Spatially localized suppression (no bleeding into unmasked areas).
