# Diffusion Inpainting Safety Alignment via SFT → BCO

**A two-stage alignment pipeline for teaching a text-guided inpainting model when and how to refuse unsafe content.**

> *"First teach the model what a refusal looks like in pixel space. Then teach it when to apply that policy."*

---

## Table of Contents

- [Diffusion Inpainting Safety Alignment via SFT → BCO](#diffusion-inpainting-safety-alignment-via-sft--bco)
  - [Table of Contents](#table-of-contents)
  - [1. Overview and Motivation](#1-overview-and-motivation)
  - [2. The SFT → Alignment Analogy from LLMs](#2-the-sft--alignment-analogy-from-llms)
  - [3. Prior Work: SFT then Alignment in Diffusion Models](#3-prior-work-sft-then-alignment-in-diffusion-models)
  - [4. Full Pipeline Architecture](#4-full-pipeline-architecture)
  - [5. Stage 1: Supervised Fine-Tuning (Refusal SFT)](#5-stage-1-supervised-fine-tuning-refusal-sft)
    - [5.1 Dataset and Mask Derivation](#51-dataset-and-mask-derivation)
    - [5.2 9-Channel UNet Input](#52-9-channel-unet-input)
    - [5.3 Loss Function: Min-SNR Weighted MSE](#53-loss-function-min-snr-weighted-mse)
    - [5.4 LoRA Configuration and Module Selection](#54-lora-configuration-and-module-selection)
    - [5.5 Noise Offset](#55-noise-offset)
    - [5.6 Merging SFT LoRA for BCO](#56-merging-sft-lora-for-bco)
  - [6. Stage 2: BCO Alignment Training](#6-stage-2-bco-alignment-training)
    - [6.1 Data Format: Pre-computed Latents](#61-data-format-pre-computed-latents)
    - [6.2 Stratified Tri-Class Batching](#62-stratified-tri-class-batching)
    - [6.3 The Reference Model](#63-the-reference-model)
    - [6.4 DDIM Inversion and Its Relevance](#64-ddim-inversion-and-its-relevance)
    - [6.5 The Full BCO Loss](#65-the-full-bco-loss)
    - [6.6 Reward Formulation](#66-reward-formulation)
    - [6.7 Hinge Cap on Unsafe Reward](#67-hinge-cap-on-unsafe-reward)
    - [6.8 Reward Shift EMA (δ\_ema)](#68-reward-shift-ema-δ_ema)
    - [6.9 BCO BCE Loss](#69-bco-bce-loss)
    - [6.10 Per-Class Weighting](#610-per-class-weighting)
    - [6.11 Reconstruction Loss](#611-reconstruction-loss)
    - [6.12 Identity Guardrail Loss](#612-identity-guardrail-loss)
    - [6.13 Combined Loss Equation](#613-combined-loss-equation)
  - [7. Multi-Step Unrolling](#7-multi-step-unrolling)
  - [8. Prompt Dropping](#8-prompt-dropping)
  - [9. Debugging Metrics: What Each Signal Means](#9-debugging-metrics-what-each-signal-means)
    - [`step`](#step)
    - [`loss`](#loss)
    - [`h_S` (h\_safe)](#h_s-h_safe)
    - [`ΔN` (h\_nudity - h\_safe)](#δn-h_nudity---h_safe)
    - [`ΔV` (h\_violence - h\_safe)](#δv-h_violence---h_safe)
    - [`δ_ema`](#δ_ema)
    - [`id_gap`](#id_gap)
    - [`unrolls`](#unrolls)
    - [`drops`](#drops)
    - [`lr`](#lr)
    - [`beta`](#beta)
  - [10. Configuration Reference](#10-configuration-reference)
  - [11. Repository Structure](#11-repository-structure)
  - [12. Training Recipes](#12-training-recipes)
    - [Stage 1: Refusal SFT](#stage-1-refusal-sft)
    - [LoRA Merge](#lora-merge)
    - [Stage 2: BCO Alignment](#stage-2-bco-alignment)
    - [Inference Check (SFT)](#inference-check-sft)
  - [Appendix: Notation Reference](#appendix-notation-reference)

---

## 1. Overview and Motivation

Standard text-guided inpainting models — including `runwayml/stable-diffusion-inpainting` — will faithfully inpaint any content the user requests, including explicit nudity and graphic violence. The model has learned to be maximally helpful to the prompt; it has no concept of refusal.

This repository implements a **two-stage alignment pipeline** to teach such a model to refuse harmful inpainting requests:

- **Stage 1 (SFT):** Supervised fine-tuning on human-approved refusal examples. Given an unsafe input image and its mask, the model is trained to predict a "safe" output (clothing added, face blurred, violent content replaced) via standard diffusion loss. This stage answers: *"What should a refusal look like in pixel space?"*

- **Stage 2 (BCO):** Binary Classifier Optimization. Using a frozen copy of the Stage 1 model as a reference, the policy is further trained to discriminate safe from unsafe completions. The model is rewarded for outperforming the reference on unsafe inputs and penalized for underperforming on safe inputs. This stage answers: *"When should a refusal be applied?"*

The key novelty is applying the **KTO/BCO family of alignment objectives** — developed for autoregressive LLMs — to a **continuous-output diffusion model operating in masked latent space**, with modifications for mask-weighted rewards, multi-step unrolling to bridge the train/inference distribution gap, and a tri-class (safe/nudity/violence) treatment.

---

## 2. The SFT → Alignment Analogy from LLMs

The two-stage recipe is directly inspired by the standard LLM alignment pipeline:

```
LLMs:        Pretraining → Instruction Tuning (SFT) → RLHF / KTO / DPO
This work:   Base Inpainter → Refusal SFT → BCO Alignment
```

In large language models, the insight is fundamental: **you cannot align a model that does not know how to follow instructions.** RLHF/KTO alignment shifts the distribution of already-capable behaviors — it does not inject new capabilities from scratch. If a base model has never seen "here is a harmful request, here is a refused response" pairs, the alignment signal has no behavioral anchor to push against. The model will not know what format, style, or content a refusal should take.

The same logic applies in diffusion models with even greater force. A vanilla SD inpainting model has absolutely no notion of *refusing* to inpaint something. It has learned to minimize denoising loss on human image data — none of which contains "I won't draw that" examples. BCO alignment operates by comparing the policy's denoising trajectory against a reference model's trajectory. If the reference model also has no idea what a refusal looks like, the reward signal is diffuse and useless.

**SFT first provides the "instruction-following" foundation:**

1. The model learns the pixel-space structure of safe refusals: how clothing looks when added over the masked region, how a blurred face composes with the surrounding background, how abstract fill patterns differ from harmful content.
2. The SFT LoRA is then *merged* into the base weights, producing a new base that already has latent representations of refusal outputs baked in.
3. BCO then fine-tunes *on top of this merged checkpoint*, using the merged model as the frozen reference. The alignment signal is now meaningful because both the policy and the reference understand what a refusal should look like. BCO then pushes the policy to apply refusals *more selectively and reliably* than the reference.

Without Stage 1, running BCO directly on the base model would be like trying to RLHF a language model that has only been pre-trained on web text — it does not know what an "answer" is, let alone what a "refusal" is, so the reward gradient is nearly meaningless.

---

## 3. Prior Work: SFT then Alignment in Diffusion Models

The idea of sequencing SFT and preference-based alignment for diffusion models is nascent but growing. This work sits at the intersection of several lines:

**KTO for Diffusion (Diffusion-KTO, 2024):** Ethayarajh et al.'s KTO objective was designed for unpaired preference data ("I know this output is good/bad, but I don't have a paired comparison"). Direct-KTO applied to diffusion computes a per-sample reward as the KL-divergence between the policy's denoising distribution and the reference's, then applies the KTO sigmoid loss. However, the original formulation did not explicitly prescribe SFT first — it was applied as a single stage, which empirically required larger datasets and more unstable training.

**DDPO / DPOK (2023):** RL-based alignment via denoising policy gradients. These methods do full-trajectory rollouts during training and use reward model feedback. They require a differentiable reward signal and suffer from high variance. They do not use a two-stage recipe.

**InstructPix2Pix / Imagic:** These are SFT-like methods for editing — they teach the model a behavior but do not include a preference/alignment phase that distinguishes *when* to apply it.

**AlignProp (2023):** Backpropagates through the diffusion chain using a reward model. Conceptually similar to our unrolling mechanism but focused on aesthetic quality rather than safety classification.

**What is novel here:** To the authors' knowledge, this is the first explicit **SFT → BCO** two-stage pipeline applied to *inpainting safety*, where:
- Stage 1 SFT specifically teaches the model the *visual vocabulary* of refusals (not general capability).
- Stage 2 BCO uses the SFT-merged model as the reference, creating a principled anchor.
- The reward is computed in **latent space against the clean z₀**, not in pixel space or via an auxiliary classifier.
- The system handles **tri-class** (safe/nudity/violence) labeling with class-specific BCE torque.
- **Multi-step DDIM unrolling** bridges the single-step training vs. multi-step inference mismatch.

---

## 4. Full Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         DATA PREPARATION                        │
│                                                                 │
│  Unsafe images + human-approved safe outputs                    │
│  → Pixel-space SFT Dataset (original_image, inpainted_image,    │
│    original_prompt)                                             │
│                                                                 │
│  Pre-compute latents via VAE encoder                            │
│  → BCO Dataset (z0, masked_latent, mask_latent, input_ids,      │
│    label=[safe, nudity, violence])   stored as .parquet         │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                   STAGE 1: REFUSAL SFT                         │
│                                                                 │
│  Base: runwayml/stable-diffusion-inpainting                     │
│  Architecture: UNet (frozen VAE + CLIP) + LoRA rank=64          │
│  LoRA targets: attention + conv layers (incl. conv_in, conv1,   │
│                conv2, proj_in, proj_out)                        │
│                                                                 │
│  Per step:                                                      │
│    1. Encode inpainted image → target_latent                    │
│    2. Encode original image → orig_latent                       │
│    3. Derive mask (pixel diff or explicit)                      │
│    4. Sample noise ε + timestep t                               │
│    5. Forward diffuse: zt = √ᾱ_t · target + √(1-ᾱ_t) · ε      │
│    6. Build 9-ch input: [zt | mask | masked_orig]               │
│    7. UNet predict ε̂ (or v)                                    │
│    8. L_SFT = Min-SNR-weighted MSE(ε̂, ε)                       │
│    9. Backward → update LoRA                                    │
│                                                                 │
│  Target: loss 0.15 → 0.06-0.08 over ~1000 steps                │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                   LoRA MERGE                                    │
│                                                                 │
│  merge_lora_for_bco():                                          │
│  base_unet + LoRA weights → merged_unet                        │
│  Save full pipeline → sft_merged_for_bco/                       │
│  This becomes the BCO base model AND the frozen reference       │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                   STAGE 2: BCO ALIGNMENT                        │
│                                                                 │
│  base: sft_merged_for_bco (merged SFT checkpoint)              │
│  policy: base + LoRA rank=128 (trainable)                       │
│  reference: deepcopy of base, fully frozen                      │
│                                                                 │
│  Stratified DataLoader: 8 safe + 4 nudity + 4 violence/batch    │
│                                                                 │
│  Per step (Pass 1 — always):                                    │
│    1. Sample t ~ U[0, 1000), ε ~ N(0,I)                        │
│    2. zt = q_sample(z0, t, ε)                                   │
│    3. [optional] replace 10-20% prompts with null embedding      │
│    4. policy forward: ε̂_θ = UNet_θ([zt|mask|masked_z0], t, c)  │
│    5. reference forward (no_grad): ε̂_ref                        │
│    6. Predict clean: ẑ0_θ, ẑ0_ref via DDPM x0 formula          │
│    7. Compute masked MSE(ẑ0_θ, z0) and MSE(ẑ0_ref, z0)         │
│    8. reward = MSE_ref - MSE_policy                             │
│    9. BCO BCE loss + reconstruction + identity losses           │
│   10. scale × 0.5, backward (frees activation graph)           │
│                                                                 │
│  Pass 2 — every unroll_every steps:                             │
│    1. Multi-step DDIM denoise: zt → z_mid (no_grad)             │
│    2. Re-noise z_mid to t_final = t // unroll_steps             │
│    3. Policy forward at t_final → BCO loss × 0.5                │
│    4. Backward (accumulates into same params)                   │
│                                                                 │
│  After grad_accum_steps micro-steps:                            │
│    - Unscale + clip gradients                                   │
│    - Optimizer step + LR scheduler step                         │
│    - Update δ_ema (reward shift EMA)                            │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Stage 1: Supervised Fine-Tuning (Refusal SFT)

### 5.1 Dataset and Mask Derivation

The SFT dataset consists of triplets: `(original_image, inpainted_image, original_prompt)`. All rows are assumed unsafe — the inpainted image is the human-approved safe version.

**Automatic mask derivation:** If no explicit mask column is provided, the mask is computed from pixel differences between the original and inpainted images:

```python
diff = np.abs(original - inpainted).max(axis=-1)   # max diff across RGB channels
mask = (diff > mask_threshold).astype(uint8) * 255
```

The max across channels (rather than grayscale mean) ensures that hue-only changes — such as adding clothing of a different color — are captured, not just brightness changes. The `mask_threshold` (default 10/255) controls sensitivity; higher values reduce noisy mask edges.

The mask identifies **where the model must learn to change the image**. During training, the mask doubles as the supervised target region and as the conditioning input to the 9-channel UNet.

### 5.2 9-Channel UNet Input

Stable Diffusion Inpainting uses a 9-channel UNet input (vs. the standard 4 channels):

```
UNet input = [ zt (4ch) | mask (1ch) | masked_original (4ch) ]
```

- **zt (4ch):** The noisy latent of the *target* (inpainted) image at timestep t.
- **mask (1ch):** Binary mask downsampled to latent resolution (image_size // 8). Where mask=1, the model must predict the refusal content.
- **masked_original (4ch):** The original latent multiplied by `(1 - mask)`. This gives the model the unchanged background as a conditioning signal, so it knows what to preserve outside the mask.

The mask is downsampled via nearest-neighbor interpolation to preserve hard binary boundaries in latent space. Bilinear interpolation would create soft edges that blur the boundary between "must refuse" and "must preserve" regions.

### 5.3 Loss Function: Min-SNR Weighted MSE

The SFT loss is standard denoising diffusion MSE with **Min-SNR-γ reweighting** (Hang et al., 2023):

```
L_SFT = E[w(t) · ||ε̂ - ε||²]
```

where `w(t) = clamp(SNR(t), max=γ) / SNR(t)` and `SNR(t) = ᾱ_t / (1 - ᾱ_t)`.

**Why Min-SNR?** The raw DDPM training loss implicitly upweights high-noise timesteps (large t, low SNR) because those timesteps produce high loss values. But high-noise timesteps are essentially asking the model to denoise from near-pure Gaussian noise — contributing little semantic content. The model tends to waste capacity learning to denoise noise. Min-SNR clamps the effective loss weight at low-SNR timesteps, redirecting gradient budget toward medium and low-noise timesteps where semantic structure matters. With γ=5 (default), timesteps with SNR < 5 are upweighted by `γ/SNR` and those with SNR ≥ 5 contribute `w(t)=1`. This materially speeds convergence for structural tasks like refusal, which require coherent semantic output.

**Prediction type:** For SD 1.5, the scheduler uses `prediction_type="epsilon"`, so the target is the raw noise ε. For v-parameterized models, the target would be the velocity `v = √ᾱ·ε - √(1-ᾱ)·x₀`.

**Noise offset:** A small noise offset (`noise_offset=0.05`) is added to the sampled noise:

```python
noise += noise_offset * torch.randn(B, C, 1, 1)
```

This per-channel scalar offset shifts the mean of the noise slightly, which empirically improves generation of dark and saturated regions — common in clothing inpainting where dark fabrics must fill the masked region. Without it, diffusion models tend to produce washed-out content in very dark regions.

### 5.4 LoRA Configuration and Module Selection

LoRA (Low-Rank Adaptation) is applied with `rank=64, alpha=64`, giving an effective scale of `alpha/rank = 1.0` — no implicit scaling, full representational capacity.

**Why include convolutional layers in LoRA targets?**

Standard LoRA for diffusion models typically targets only the attention layers (`to_q, to_k, to_v, to_out.0`). For semantic tasks like style transfer or prompt following, this is sufficient because attention is where the model integrates text and spatial context.

**Refusal is fundamentally different.** It requires *structural* changes to the masked region — drawing fabric texture, generating hair in a new style, creating abstract fill. These spatial transformations are primarily mediated by the ResNet convolutional layers (`conv1`, `conv2`) and the input projection `conv_in`. The attention layers decide *what content* to place; the conv layers determine *how that content is rendered spatially*.

```python
LORA_TARGET_MODULES = [
    "to_q", "to_k", "to_v", "to_out.0",   # cross/self-attention
    "ff.net.0.proj", "ff.net.2",           # transformer feed-forward
    "proj_in", "proj_out",                 # spatial projections
    "conv1", "conv2",                      # ResNet spatial layers
    "conv_in",                             # 9-channel input projection
]
```

`conv_in` is particularly important: it is the layer that projects the 9-channel input (noisy latent + mask + masked original) into the UNet's internal feature space. By including it in LoRA, the model can learn to weight these three inputs differently for refusal outputs — for instance, attending more strongly to the mask channel to decide which region to refuse.

Dropout of 0.05 on LoRA adapters prevents memorization, which is important with relatively small refusal datasets (~2000 samples).

### 5.5 Noise Offset

The `noise_offset` hyperparameter addresses a known failure mode of diffusion models: poor reproduction of very dark or very bright image regions. This manifests in clothing inpainting as washed-out or grey fills where dark fabric should appear.

Mathematically, the noise offset adds a per-channel constant to the per-pixel Gaussian noise:

```
ε_effective = ε_normal + δ · ε_channel    where δ=0.05, ε_channel ~ N(0,I) at channel level
```

This shifts the noise mean away from zero, which during training teaches the model to denoise toward darker or brighter values when appropriate, rather than always regressing toward grey.

### 5.6 Merging SFT LoRA for BCO

After SFT, the LoRA adapters must be **merged into the base weights** before BCO begins:

```python
peft_unet = PeftModel.from_pretrained(base_unet, lora_checkpoint_dir)
merged_unet = peft_unet.merge_and_unload()
```

This is a hard requirement, not an implementation convenience. BCO requires a **frozen reference model** that represents the policy's starting point. If LoRA adapters were left separate:
- The reference model and the new policy's LoRA adapters would share the same base, making it impossible to truly freeze the reference.
- The reward (MSE_ref - MSE_policy) would be undefined at initialization because both models would be identical.
- After merging, the reference is a complete, standalone model, and the new policy starts as an identical copy with fresh LoRA adapters on top. This creates a clean separation.

---

## 6. Stage 2: BCO Alignment Training

### 6.1 Data Format: Pre-computed Latents

Unlike Stage 1 which works in pixel space, Stage 2 operates entirely on **pre-computed VAE latents** stored in Parquet files. Each sample contains:

| Field | Shape | Description |
|---|---|---|
| `z0` | `[4, H, W]` | Clean latent of the original (unsafe) image, VAE-encoded |
| `masked_latent` | `[4, H, W]` | `z0 * (1 - mask)` — background preserved, masked region zeroed |
| `mask_latent` | `[1, H, W]` | Binary mask in latent space |
| `input_ids` | `[77]` | CLIP token IDs for the original (unsafe) prompt |
| `label` | `[3]` | One-hot: `[safe, nudity, violence]` |

Pre-computing latents eliminates VAE encoding from the training loop, which was a significant bottleneck. Since the VAE is frozen throughout both stages, encoding each image once and caching the result is mathematically equivalent to encoding on the fly — but 3-4× faster per step.

The dataset class uses row-group caching and binary search (`bisect`) over cumulative row counts to support efficient random access across large Parquet shards without loading entire files into memory.

### 6.2 Stratified Tri-Class Batching

A key design decision is the `StratifiedBatchSampler`, which enforces a fixed composition per batch:

```
Batch (16 samples) = 8 safe + 4 nudity + 4 violence
```

**Why not random sampling?** Without stratification, class imbalance would dominate. Real-world inpainting safety datasets are skewed toward safe examples (the internet has many more benign inpainting prompts than explicit ones). A randomly sampled batch of 16 might contain 14 safe and 2 unsafe samples, leading to:
- The BCO loss being dominated by the safe-side term.
- The nudity and violence suppression signals barely contributing gradient.
- The model learning to "be safe in a general sense" rather than specifically refusing graphic content.

With the 8/4/4 split, every batch guarantees that violence examples (the rarest and hardest class) contribute `4/16 = 25%` of the gradient. The BCE loss is further weighted by class coefficients (nudity ×6, violence ×4 in default config), so the effective torque from unsafe samples significantly exceeds their batch fraction.

The sampler uses cyclic sampling within each class pool to handle class size differences — if the violence pool is small, samples are repeated (with shuffling) until all classes have contributed equally across the epoch.

### 6.3 The Reference Model

The reference model is a **frozen deepcopy of the SFT-merged base model**:

```python
ref_unet = copy.deepcopy(base_unet)
ref_unet.eval()
for p in ref_unet.parameters():
    p.requires_grad_(False)
```

The reference model serves two critical roles:

**Role 1 — Reward anchor.** The BCO reward for a sample is `reward = MSE(ẑ0_ref, z0) - MSE(ẑ0_θ, z0)`. This is a *relative* metric: the policy is rewarded when it gets *closer* to the true z0 than the reference on a safe sample, and punished when it drifts *further* than the reference on an unsafe sample. Without the reference, there would be no stable baseline — the policy could game the reward by making both MSEs very large.

**Role 2 — KL regularization (implicit).** BCO/KTO is derived from a KL-constrained optimization: maximize expected reward subject to `KL(π || π_ref) ≤ ε`. The reference model IS π_ref. The `delta` (reward shift) term in the BCO BCE loss implements this KL penalty implicitly without computing the KL directly. The beta parameter controls how tightly the policy is regularized toward the reference — high beta = stay close to reference, low beta = allow large deviations.

The reference model runs in `torch.no_grad()` context on every step. Its outputs `pred_ref` and `z0_pred_ref` are computed but not differentiated through.

### 6.4 DDIM Inversion and Its Relevance

**What is DDIM Inversion?** DDIM (Denoising Diffusion Implicit Models, Song et al. 2020) deterministically inverts a real image into its corresponding noise latent by reversing the deterministic DDIM sampling process. Given a real image x₀, DDIM inversion finds the noise latent x_T such that running DDIM forward sampling from x_T exactly reconstructs x₀. This is the inverse of the standard denoising trajectory.

**How it is used here (in the multi-step unrolling):** The `_multistep_denoise` function in `train_one_epoch.py` implements a forward DDIM trajectory with η=0 (fully deterministic):

```python
# After predicting z0_hat from (zt, t):
zt_next = a_next * z0_hat + sigma_next * pred
```

This is the DDIM update rule with no stochastic noise injection. The deterministic path means:
1. Given the same initial noise, the trajectory is fully reproducible.
2. The intermediate latents `z_mid` follow the same deterministic path the model would take during inference.
3. Re-noising `z_mid` to a lower timestep `t_final = t // unroll_steps` and computing the BCO loss there simulates the model's actual denoising behavior during inference — not just a single arbitrary step.

**Why DDIM over DDPM for unrolling?** Standard DDPM steps inject fresh random noise at each step, making the trajectory stochastic. This means running 2-3 DDPM steps produces a different intermediate latent every time, even for the same starting point. The BCO loss computed on this stochastic intermediate has high variance. DDIM's deterministic path is essential for stable unrolled gradients.

**Connection to DDIM Inversion:** The unrolling is effectively the *forward* direction of DDIM. DDIM inversion in the traditional sense (real image → noise) is not explicitly performed during BCO training — instead, the model unrolls from a noisy latent toward z0, which is the standard (generative) direction. However, the DDIM mathematics are the same: the same `(zt, t, α_bar)` → `z_{t-k}` formula is used. If one were to cache the noise latents corresponding to training images and start unrolling from those, it would be equivalent to starting from the DDIM-inverted noise of each training image, which would provide more realistic training conditions.

### 6.5 The Full BCO Loss

The total loss is:

```
L = 4 · L_BCO  +  L_recon  +  L_identity
```

Each component controls a different aspect of the policy's behavior, described in detail below.

### 6.6 Reward Formulation

The reward for a sample is the relative improvement of the policy over the reference, measured in **masked latent MSE against the ground-truth clean latent z0**:

```
reward_i = MSE(ẑ0_ref_i, z0_i) - MSE(ẑ0_θ_i, z0_i)
```

where the MSE is computed **only over the masked region**, with optional extra weighting on masked pixels:

```python
w = 1.0 + mask_weight * mask   # default mask_weight=0.5 → masked pixels weighted 1.5×
mse_weighted = mse * w
reward = (mse_weighted * mask).sum() / mask.sum()
```

**Why masked MSE?** The BCO signal should reflect how well the model handles the *edited region*, not the entire image. The unmasked background is trivially preserved (both policy and reference are constrained to keep it); including it would dilute the reward signal with low-information background pixels.

**Why against z0 (clean latent) rather than against ε (the noise target)?** Noise-space MSE is a necessary objective for diffusion training but is a poor reward for alignment. The noise ε is randomized anew each step, so `||ε̂_θ - ε||` is a noisy proxy for actual image quality. The x0-prediction `ẑ0 = (zt - σ·ε̂) / √ᾱ` is a more direct estimate of the final denoised output and is comparable across timesteps, making it a more stable and interpretable reward signal.

**Initialization:** At the start of BCO training, policy = reference (identical merged checkpoint). Therefore `reward_i ≈ 0` for all samples. As training progresses:
- On safe samples: the policy should improve (reward > 0) — it is reinforced to preserve the original content better.
- On unsafe samples: the policy should diverge from z0 (reward < 0, or close to 0) — it is reinforced to produce different (refused) content.

### 6.7 Hinge Cap on Unsafe Reward

Without a cap, the reward on unsafe samples could grow without bound in the negative direction (the policy becomes arbitrarily bad at reconstructing the unsafe content). This leads to training instability and mode collapse — the model produces grey blobs for all unsafe inputs.

```python
ref_reward_mean = reward[~is_safe].mean().detach()
unsafe_reward_cap = ref_reward_mean.abs() * 1.5
reward = torch.where(is_safe, reward, reward.clamp(max=unsafe_reward_cap))
```

The cap is set at 1.5× the absolute mean reward of the reference model on unsafe samples. This allows the policy to get somewhat worse than the reference at reconstructing unsafe content (which is desired — it should refuse) but not catastrophically worse. The `1.5×` factor is a generous margin that prevents collapse while still allowing meaningful divergence.

### 6.8 Reward Shift EMA (δ_ema)

In KTO theory, the reward must be centered around a baseline to avoid mode collapse. The `delta` term serves this role:

```python
delta_raw = (mean_safe_reward + mean_unsafe_reward) / 2.0
```

A running EMA of delta is maintained across steps:

```python
ema.value = momentum * ema.value + (1 - momentum) * delta_raw
```

with `momentum=0.999` — a very slow-moving average.

**Why EMA?** The raw per-step delta is noisy (it depends on which specific samples are in the batch). Using the instantaneous delta would inject noise into the BCO loss, making training unstable. The EMA smooths this to a stable baseline representing the *expected* average reward over the training distribution.

**Why clip delta to [-0.03, 0.03]?** This prevents the centering term from dominating the loss signal. If delta grows large (e.g., 0.2), the shifted reward `r - delta` would be near zero for all samples, and the BCO gradients would vanish. The clamp ensures the centering is a gentle correction, not a nullifying shift.

**In the BCO BCE loss:**
```python
r_shifted = reward - delta
bce_per_sample = softplus(-label_sgn * beta * r_shifted)
```

For a safe sample with `label_sgn = +1`: loss = `log(1 + exp(-beta * (reward - delta)))`. This is minimized when `reward >> delta`, i.e., when the policy significantly outperforms the reference on safe content.

For an unsafe sample with `label_sgn = -1`: loss = `log(1 + exp(+beta * (reward - delta)))`. This is minimized when `reward << delta`, i.e., when the policy significantly underperforms the reference on unsafe content (it is diverging from the unsafe z0, which is the desired behavior).

### 6.9 BCO BCE Loss

The core alignment loss is a binary cross-entropy formulation over the reward signal:

```
L_BCO = E_y[w_y · softplus(-y · β · (r - δ))]
```

where `y ∈ {+1, -1}` is the signed label (safe → +1, unsafe → -1), `β` is the logit scale, `r` is the reward, and `δ` is the EMA-stabilized reward shift.

This is mathematically equivalent to a binary logistic regression over the reward gap. The model is being trained to be a binary classifier: "does this completion look more like a safe inpainting (reward > δ) or a refused inpainting (reward < δ)?" The diffusion weights are updated to push the reward in the correct direction for each class.

**Beta (β) controls sensitivity.** With β=50 (default config), the BCE loss saturates quickly once `|r - δ|` exceeds ~0.02. This means the model stops receiving gradient once it has clearly classified a sample correctly, focusing gradient budget on the borderline cases. Low β (e.g., 7 in the `bco_loss.py` default) spreads gradient more broadly, which is more stable early in training but less precise at convergence.

### 6.10 Per-Class Weighting

```python
bco_coeffs = {"safe": 1.0, "nudity": 6.0, "violence": 4.0}
```

These weights scale the BCO BCE loss per sample class. Despite the stratified batching (8 safe, 4 nudity, 4 violence), the class sizes in the actual dataset and the difficulty of each class vary. Nudity (×6) is weighted more heavily than violence (×4) because:
1. Nudity samples are more common and diverse — more variance to learn from, so a stronger signal is warranted.
2. Violence is visually more heterogeneous and harder to refuse consistently; an overweighted violence signal early in training can destabilize the safe class.

The safe class (×1) acts as an anchor. Upweighting it would cause the model to over-restrict its safe behavior, becoming reluctant to make even benign edits. Keeping it at 1.0 preserves the model's utility for legitimate inpainting.

**The interaction with stratified batching:** The effective gradient contribution of each class = `class_batch_fraction × class_weight`. For nudity: `(4/16) × 6 = 1.5×`. For violence: `(4/16) × 4 = 1.0×`. For safe: `(8/16) × 1 = 0.5×`. This means unsafe classes collectively receive 2.5× more gradient torque than safe, ensuring the refusal policy is actively shaped rather than just maintained.

### 6.11 Reconstruction Loss

The reconstruction loss anchors the **background** of safe images:

```python
unmask = (1 - mask).expand_as(pred_train)
sq_err_recon = ((pred_train * unmask) - (noise * unmask)) ** 2
L_recon = E[recon_weight_safe · recon_per_sample]
```

This loss is applied **only to safe samples** (`recon_weight` is set to 0 for unsafe samples in the training loop). It measures the noise-prediction error in the unmasked region and penalizes deviation from standard diffusion behavior there.

**Why is it needed?** Without this loss, the BCO alignment gradient has no explicit constraint on the background region. In practice, early BCO training causes diffuse suppression that bleeds outside the mask — the model slightly degrades background quality in its attempt to refuse content everywhere. The reconstruction loss corrects this by saying "outside the mask, you must still be a good denoising model."

**Why only safe images?** On unsafe images, the desired behavior is to refuse the masked region. A reconstruction loss on unsafe backgrounds would be fine in principle, but the `identity_weight_unsafe` (small identity penalty on unmasked region) serves the same purpose. Separating the two losses allows independent control of each.

### 6.12 Identity Guardrail Loss

The identity loss prevents the policy from drifting too far from the reference in absolute terms, using a **hinge formulation**:

```python
sq_err_id = (z0_pred_train - z0_pred_ref.detach()) ** 2
hinged = relu(sq_err_id.mean(dim=[1,2,3]) - drift_threshold) ** 2
L_identity = E[identity_weight · hinged]
```

with `drift_threshold = 0.02`.

**Why a hinge instead of plain MSE?** A plain MSE would constantly penalize any deviation from the reference, essentially preventing the policy from learning anything beyond the reference. The hinge only activates when the policy drifts *beyond* 0.02 MSE from the reference. Below that threshold, the loss is zero and the policy is free to move. Above it, the quadratic penalty kicks in strongly (note: it is `relu(...)^2`, not `relu(...)` — this quadratic tail prevents extreme drift while allowing moderate exploration).

**Why is identity_weight set differently for safe vs. unsafe?**

| Sample class | `identity_weight` | Reason |
|---|---|---|
| Safe | 10.0 (strong) | Preserve the reference's hand/face rendering quality on legitimate edits |
| Unsafe | 7.0 (moderate) | Allow divergence in the masked region but anchor the unmasked background |

For unsafe samples, a zero identity weight would allow the model to arbitrarily distort even unmasked regions (arms, background) in its attempt to suppress content. A small but nonzero weight on unsafe samples prevents this "bleeding" — the suppression signal is directed to the masked area while the background is gently held in place.

### 6.13 Combined Loss Equation

```
L_total = 4 · L_BCO + L_recon + L_identity
```

The `4×` multiplier on BCO ensures the alignment signal dominates. With typical values:
- `L_BCO ≈ 0.4–0.7` (softplus near saturation)
- `L_recon ≈ 0.01–0.05` (MSE on background noise)
- `L_identity ≈ 0–0.01` (hinge, near zero when tracking well)

Without the `4×`, the reconstruction loss would dominate by raw magnitude and the model would optimize for background fidelity at the expense of the alignment signal.

---

## 7. Multi-Step Unrolling

**The train/inference distribution mismatch** is a fundamental problem in diffusion alignment. During BCO training, the model sees a *single* arbitrary timestep t per sample. During inference, it runs 20–50 sequential denoising steps, each building on the previous prediction. A model that correctly predicts ε̂ at t=800 might produce completely different outputs at t=400 than it would if it had run the first step through t=800 → t=400 sequentially.

**Unrolling closes this gap.** Every `unroll_every` steps (default 40), the training performs two BCO loss calculations:

**Pass 1 (standard, every step):**
- Timestep t sampled from [0, 1000)
- Single-step BCO loss at t
- `loss × 0.5` (weight halved when unrolling)
- Backward → frees activation graph

**Pass 2 (unrolled, every unroll_every steps):**
- From zt, run `(unroll_steps - 1)` DDIM steps under `no_grad` to reach `z_mid`
- Re-noise `z_mid` to `t_final = t // unroll_steps` (a lower timestep)
- Compute BCO loss at `t_final` on the new trajectory
- `loss × 0.5` backward → gradients accumulate

**Memory management:** The two passes are deliberately separated — Pass 1's backward is called *before* Pass 2's forward. This means only one UNet activation graph is held in VRAM at any time, avoiding the OOM errors that would result from holding both simultaneously.

**Why `0.5` weight for each pass?** The combined gradient on unroll steps is approximately `0.5 + 0.5 = 1.0 × standard step`. This keeps the effective learning rate consistent regardless of whether unrolling occurs.

With `unroll_steps=7` (config default), the model takes 6 DDIM steps from a high-noise latent, then evaluates the BCO loss at the resulting lower-noise point. At t_start=700, this means evaluating loss at t≈100, which is in the "near-clean" regime where semantic content is already largely determined — exactly where inference decisions about content vs. refusal are made.

---

## 8. Prompt Dropping

With probability `p_drop` (default 10–20%), each sample's text conditioning is replaced with the **null embedding** (an empty string tokenized to `""`):

```python
drop_mask = torch.rand(B) < p_drop
enc_hidden = torch.where(drop_mask[:, None, None], null_hidden.expand(B, -1, -1), enc_hidden)
```

**Why?** The original motivation was to fix a specific failure mode: **arm smudging**. Early BCO training without prompt dropping showed the model using the text prompt as its primary signal for where to suppress content. On prompts like "nude woman standing," the model would suppress the entire upper body, including arms and hands that were outside the mask boundary.

Prompt dropping forces the model to rely on spatial/visual context (the mask and masked latent channels) rather than the text prompt to determine *where* to suppress. With 20% null conditioning:
- 80% of steps: the model learns the alignment policy conditioned on text.
- 20% of steps: the model must make decisions from visual context alone, preventing over-reliance on text.

This is analogous to **classifier-free guidance (CFG) training**, where null conditioning is used during training to allow unconditional generation at inference. Here it serves a different purpose — spatial grounding of the safety policy rather than guidance scale control.

---

## 9. Debugging Metrics: What Each Signal Means

The training loop prints and logs the following metrics every `log_every` steps:

```
step=500 loss=0.8234 h_S=0.823 ΔN=-0.412 ΔV=-0.389 δ_ema=0.0012 id_gap=0.0018 unrolls=12 drops=9(11.3%) lr=8.00e-07 beta=50.0
```

### `step`
Global optimizer step count. One step = `grad_accum_steps` micro-steps. Expected to increase monotonically. Checkpoints are saved at multiples of `save_every` (default 500).

### `loss`
Average total loss `4·L_BCO + L_recon + L_identity` over the last `log_every` steps. **Expected behavior:** decreases from ~2.0–3.0 at initialization to ~0.6–1.2 at convergence. Very low loss (<0.3) indicates the policy has saturated the BCO signal — reduce beta or check for collapse. Very high loss (>5) suggests a learning rate or beta issue.

### `h_S` (h_safe)
The BCO satisfaction probability for safe samples: `σ(+β·(r - δ))`. Ranges [0, 1]. **Expected behavior:**
- At initialization: ~0.5 (policy and reference identical, reward ≈ 0)
- At convergence: 0.75–0.90 (policy consistently outperforms reference on safe content)
- If h_S > 0.95: safe reward may be saturating; the model is over-rewarded for safe content, risking that it suppresses unsafe content less aggressively.
- If h_S < 0.55 after 1000 steps: the policy is failing to improve on safe content; check recon_weight.

### `ΔN` (h_nudity - h_safe)
The satisfaction gap between nudity and safe classes. **Expected behavior:** strongly negative (−0.3 to −0.6) at convergence. This means the policy is less "satisfied" (in the BCO sense) on nudity samples than on safe ones — it is correctly applying the refusal policy to nudity but not to safe content. If ΔN approaches 0, the model is treating nudity and safe samples identically — the nudity suppression signal has failed.

### `ΔV` (h_violence - h_safe)
Same as ΔN but for violence class. Expected to be negative, typically less negative than ΔN (violence is harder to refuse consistently). If |ΔV| < |ΔN| throughout training, this is expected given the smaller violence class weight.

### `δ_ema`
The exponential moving average of the raw reward shift `delta`. **Expected behavior:**
- Near zero for most of training (healthy centering).
- Should stay within [−0.03, 0.03] (enforced by clamping).
- Steady positive drift indicates the policy is consistently outperforming the reference (on average) — fine.
- Rapid oscillation suggests unstable batching or too-low EMA momentum.

### `id_gap`
Mean of `||ẑ0_θ - ẑ0_ref||²` across the batch (identity gap before hinging). **Expected behavior:**
- At initialization: 0 (policy = reference).
- Healthy training: 0.005–0.025.
- If `id_gap > 0.05`: the policy has drifted significantly from the reference; check if identity_weight is too low or learning rate too high.
- The hinge threshold of 0.02 means that once `id_gap > 0.02`, the identity loss is actively constraining drift.

### `unrolls`
Cumulative count of multi-step unrolling passes executed. Increments by 1 every `unroll_every` steps. **Expected:** roughly `global_step / unroll_every`. Confirms that unrolling is occurring at the configured frequency.

### `drops`
Count (and percentage) of prompt-dropped samples in the last logging window. **Expected:** approximately `p_drop × log_every × batch_size` samples, e.g., at p_drop=0.20 and log_every=50 with batch=16: ~160 drops over 50 steps. Significant deviation suggests a bug in the prompt dropping logic.

### `lr`
Current learning rate from the scheduler. For the cosine schedule with warmup, this should:
- Ramp from 0 → `lr_base` over `warmup_steps`.
- Decay cosinely from `lr_base` toward 0 over `max_steps`.
- **Monitor:** if lr is not decaying, the scheduler may not be stepping correctly.

### `beta`
The BCO logit scale. Currently read from `cfg["training"]["beta"]` each step (no dynamic schedule). With β=50, the loss is sensitive to reward differences of ~1/β = 0.02 — appropriate given that rewards are masked MSE differences typically in [−0.05, 0.05]. If rewards are much larger, reduce beta; if the loss is always near saturation, reduce beta.

---

## 10. Configuration Reference

Key parameters in `configs/inpaint.yaml`:

```yaml
model:
  base_model: sft_merged_for_bco        # Stage 1 SFT merged checkpoint
  merged_base: true
  hf_dataset_repo: org/dataset-name
  use_lora: true
  lora:
    r: 128                             # BCO LoRA rank (higher than SFT's 64)
    lora_alpha: 128

training:
  lr: 8.0e-7                           # Conservative LR for alignment
  warmup_steps: 1000
  max_steps: 5000
  batch_size: 16
  grad_accum_steps: 2                  # Effective batch = 32
  beta: 50                             # BCO logit scale

  t_min: 0                             # Full noise schedule
  t_max: 1000

  unroll_every: 40                     # Unroll every 40 steps
  unroll_steps: 7                      # 7-step DDIM trajectory

  prompt_drop_prob: 0.20               # 20% null conditioning

  safe_count_per_batch: 8
  nudity_count_per_batch: 4
  violence_count_per_batch: 4

  kto_coeffs:
    safe: 1.0
    nudity: 6
    violence: 4

  recon_weight: 1                      # Background anchor (safe only)
  identity_weight: 10                  # Full drift penalty (safe)
  identity_weight_unsafe: 7            # Background anchor (unsafe)
```

---

## 11. Repository Structure

```
inpainter-training-BCO/
├── configs/
│   └── inpaint.yaml              # Main training configuration
├── data/
│   ├── dataset.py                # LatentInpaintDataset (Parquet loader)
│   ├── sampler.py                # StratifiedBatchSampler (8/4/4 tri-class)
│   └── collate.py                # DataLoader collation
├── engine/
│   ├── train_one_epoch.py        # Main BCO training loop (Pass 1 + Pass 2)
│   ├── train_one_epoch-old.py    # Previous single-pass version (archived)
│   ├── evaluate.py               # Visual evaluation (decode + inpaint)
│   └── checkpoint.py             # Save/resume checkpoints
├── losses/
│   ├── bco_loss.py               # BCO loss (reward, hinge, BCE, recon, identity)
│   └── kto_loss-old.py           # Previous KTO implementation (archived)
├── models/
│   ├── unet_wrapper.py           # 9-channel UNet forward pass
│   └── diffusion_utils.py        # q_sample (forward diffusion)
├── scripts/
│   ├── train.py                  # Entry point (load config, init, call train_loop)
│   ├── eval_all_checkpoints.py   # Batch evaluation across checkpoint history
│   ├── push_to_hf.py             # Upload checkpoints to HuggingFace Hub
│   ├── track_gpu.py              # GPU memory and utilization monitor
│   └── wandb_sync_and_clean.py   # Offline W&B sync utility
├── utils/
│   ├── logging.py                # W&B init and logging helpers
│   ├── plotting.py               # Training curve plots
│   └── seed.py                   # Reproducible seeding
├── refusal_training.py           # Stage 1 SFT (self-contained script)
├── merge_lora.py                 # LoRA merge utilities
└── requirements.txt
```

---

## 12. Training Recipes

### Stage 1: Refusal SFT

```bash
# From HuggingFace dataset
accelerate launch refusal_training.py \
  --hf_dataset   your-org/your-refusal-dataset \
  --hf_token     hf_xxxxxxxxxxxxxxxxxxxx \
  --output_dir   ./sft_refusal_ckpt \
  --num_epochs   8 \
  --batch_size   8 \
  --gradient_accumulation_steps 2 \
  --lora_rank    64 \
  --lora_alpha   64 \
  --learning_rate 1e-4 \
  --snr_gamma    5.0 \
  --noise_offset 0.05

# From local parquet / csv
accelerate launch refusal_training.py \
  --dataset_path ./data.parquet \
  --output_dir   ./sft_refusal_ckpt \
  --num_epochs   8 --batch_size 8 --gradient_accumulation_steps 2 \
  --lora_rank 64 --lora_alpha 64 --learning_rate 1e-4
```

**Expected loss trajectory:** 0.15 → 0.06–0.08 by epoch 5. Extend to 10–12 epochs if still declining.

### LoRA Merge

```python
from refusal_training import merge_lora_for_bco
merge_lora_for_bco(
    base_model_id       = "runwayml/stable-diffusion-inpainting",
    lora_checkpoint_dir = "./sft_refusal_ckpt/final/unet_lora",
    output_dir          = "./sft_merged_for_bco",
)
```

### Stage 2: BCO Alignment

```bash
# Edit configs/inpaint.yaml to point base_model at ./sft_merged_for_bco
python scripts/train.py
```

**Expected metric trajectory:**
- Steps 0–500: `h_S` rises from 0.5 → 0.65, `ΔN` drops from 0 → −0.2
- Steps 500–2000: `h_S` reaches 0.80–0.85, `ΔN` → −0.35–0.45
- Steps 2000–5000: convergence, `id_gap` stabilizes below 0.02

### Inference Check (SFT)

```python
from refusal_training import run_inference_check
run_inference_check(
    checkpoint_dir  = "./sft_merged_for_bco",
    test_image_path = "unsafe_image.jpg",
    test_mask_path  = "mask.png",
    prompt          = "a person wearing a shirt",
    output_path     = "refused_output.png",
)
```

---

## Appendix: Notation Reference

| Symbol | Meaning |
|---|---|
| z0 | Clean latent (VAE-encoded image) |
| zt | Noisy latent at timestep t: `√ᾱ_t · z0 + √(1-ᾱ_t) · ε` |
| ᾱ_t | Cumulative noise schedule coefficient at t |
| ε | Ground truth Gaussian noise |
| ε̂_θ | Policy (trainable) UNet noise prediction |
| ε̂_ref | Reference (frozen) UNet noise prediction |
| ẑ0_θ | Policy's z0 estimate: `(zt - √(1-ᾱ_t)·ε̂_θ) / √ᾱ_t` |
| ẑ0_ref | Reference's z0 estimate |
| r | BCO reward: `MSE(ẑ0_ref, z0) - MSE(ẑ0_θ, z0)` |
| δ | Reward shift (EMA of mean reward) |
| β | BCO logit scale |
| h | BCO satisfaction probability: `σ(±β·(r-δ))` |
| w_y | Per-class BCE weight |
| L_BCO | Binary cross-entropy alignment loss |
| L_recon | Noise-space MSE on background (safe only) |
| L_identity | Hinge MSE between ẑ0_θ and ẑ0_ref |

---
