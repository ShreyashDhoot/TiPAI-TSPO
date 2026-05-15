"""
merge_lora_into_base.py
───────────────────────
Load Stable Diffusion 1.5 Inpainter + a LoRA / full-weight safetensors
checkpoint, merge the weights mathematically into the base UNet, and save
a clean merged pipeline that can be reloaded with a standard
StableDiffusionInpaintPipeline.from_pretrained() call — no PEFT/LoRA
runtime overhead, ready for the next alignment round.

Supports three checkpoint flavours automatically detected at runtime:
  1. PEFT LoRA  (.safetensors with keys like  unet.lora_A / lora_B)
  2. Raw LoRA   (.safetensors with keys like  lora_unet_* or unet_*
                 following the Kohya / A1111 naming convention)
  3. Full UNet  (.safetensors / .pt  — every key is a UNet weight)

Usage:
    python merge_lora_into_base.py \
        --base_model  runwayml/stable-diffusion-inpainting \
        --checkpoint  /path/to/your_lora.safetensors \
        --output_dir  /path/to/merged_pipeline \
        [--lora_alpha 1.0]          # scale applied to LoRA delta (default 1.0)
        [--lora_rank  64]           # only needed for raw LoRA (auto-detected)
        [--dtype       float16]     # float16 | float32 | bfloat16
        [--device      cpu]         # cpu is safest for merging; use cuda to speed up
"""

import argparse
import os
import re
from pathlib import Path
from typing import Dict, Optional

import torch
from safetensors.torch import load_file as load_safetensors
from diffusers import StableDiffusionInpaintPipeline, UNet2DConditionModel


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

DTYPE_MAP = {
    "float32":  torch.float32,
    "float16":  torch.float16,
    "bfloat16": torch.bfloat16,
}


def load_checkpoint(path: str) -> Dict[str, torch.Tensor]:
    """Load .safetensors or .pt/.pth checkpoint into a state-dict."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    if p.suffix == ".safetensors":
        print(f"[load] Reading safetensors: {p}")
        return load_safetensors(str(p), device="cpu")
    elif p.suffix in (".pt", ".pth", ".bin"):
        print(f"[load] Reading torch checkpoint: {p}")
        ckpt = torch.load(str(p), map_location="cpu")
        # unwrap common wrappers
        for key in ("state_dict", "model", "unet"):
            if isinstance(ckpt, dict) and key in ckpt:
                ckpt = ckpt[key]
        return ckpt
    else:
        raise ValueError(f"Unsupported checkpoint extension: {p.suffix}")


def detect_flavour(state_dict: Dict[str, torch.Tensor]) -> str:
    """
    Detect whether the checkpoint is:
      'peft_lora'  — PEFT-style keys (base_model.model.* + lora_A/lora_B)
      'raw_lora'   — Kohya / A1111 style (lora_unet_* with .lora_up/.lora_down)
      'full_unet'  — plain UNet state dict

    Scans ALL keys (not just first 20) to handle checkpoints where metadata
    or non-LoRA keys appear at the front.
    """
    keys = list(state_dict.keys())

    # Count evidence for each flavour across the full key set
    peft_hits   = sum(1 for k in keys if "lora_A" in k or "lora_B" in k)
    raw_hits    = sum(1 for k in keys if "lora_up" in k or "lora_down" in k)
    unet_hits   = sum(1 for k in keys if k.startswith("conv_in")
                                      or k.startswith("down_blocks")
                                      or k.startswith("up_blocks")
                                      or k.startswith("mid_block"))

    print(f"[detect] Key evidence — peft_lora: {peft_hits}, "
          f"raw_lora: {raw_hits}, full_unet: {unet_hits}  (total keys: {len(keys)})")

    if peft_hits > 0:
        return "peft_lora"
    elif raw_hits > 0:
        return "raw_lora"
    else:
        return "full_unet"


# ─────────────────────────────────────────────────────────────────────────────
# Flavour 1: PEFT LoRA merge
# ─────────────────────────────────────────────────────────────────────────────

def merge_peft_lora(
    unet: UNet2DConditionModel,
    state_dict: Dict[str, torch.Tensor],
    alpha: float,
    dtype: torch.dtype,
) -> UNet2DConditionModel:
    """
    Merge PEFT-style LoRA weights into UNet in-place.

    PEFT stores weights as:
        base_model.model.<unet_path>.lora_A.weight  — [r, in]
        base_model.model.<unet_path>.lora_B.weight  — [out, r]

    Merged delta: W += alpha * (B @ A)
    """
    print("[merge] Detected PEFT LoRA format")

    # ── Diagnostic: show which of the 6 target module types were found ────
    TARGET_SUFFIXES = ["to_q", "to_k", "to_v", "to_out.0",
                       "ff.net.0.proj", "ff.net.2"]
    found_targets: Dict[str, int] = {t: 0 for t in TARGET_SUFFIXES}

    # Group A/B pairs by their module path
    lora_pairs: Dict[str, Dict] = {}
    for k, v in state_dict.items():
        # strip "base_model.model." prefix (present in PEFT-saved files)
        # also handle files saved without this prefix (some PEFT versions omit it)
        k_clean = re.sub(r"^base_model\.model\.", "", k)
        if ".lora_A." in k_clean:
            mod_path = k_clean.replace(".lora_A.weight", "")
            lora_pairs.setdefault(mod_path, {})["A"] = v
            for t in TARGET_SUFFIXES:
                if mod_path.endswith(t):
                    found_targets[t] += 1
        elif ".lora_B." in k_clean:
            mod_path = k_clean.replace(".lora_B.weight", "")
            lora_pairs.setdefault(mod_path, {})["B"] = v

    print("[merge] Target module coverage in checkpoint:")
    for t, count in found_targets.items():
        status = "✓" if count > 0 else "✗ MISSING"
        print(f"         {status}  {t:20s}  ({count} layers)")
    print(f"         Total LoRA module pairs found: {len(lora_pairs)}")

    merged_count = 0
    skipped      = []

    unet_sd = dict(unet.named_parameters())

    for mod_path, ab in lora_pairs.items():
        if "A" not in ab or "B" not in ab:
            skipped.append(mod_path)
            continue

        weight_key = mod_path + ".weight"
        if weight_key not in unet_sd:
            skipped.append(weight_key)
            continue

        param = unet_sd[weight_key]
        A = ab["A"].to(dtype=dtype, device=param.device)
        B = ab["B"].to(dtype=dtype, device=param.device)

        # delta = alpha * B @ A  (rank decomposition)
        # Handle Conv2d: A is [r, Cin, kH, kW] — reshape to [r, Cin*kH*kW]
        if A.dim() == 4:
            r = A.shape[0]
            A_2d = A.reshape(r, -1)
            B_2d = B.reshape(B.shape[0], r)
            delta = (B_2d @ A_2d).reshape(param.shape)
        else:
            delta = B @ A

        with torch.no_grad():
            param.data.add_(alpha * delta)

        merged_count += 1

    print(f"[merge] PEFT LoRA: merged {merged_count} modules, skipped {len(skipped)}")
    if skipped:
        print(f"         skipped keys (first 5): {skipped[:5]}")

    return unet


# ─────────────────────────────────────────────────────────────────────────────
# Flavour 2: Raw / Kohya LoRA merge
# ─────────────────────────────────────────────────────────────────────────────

# Kohya key format:
#   lora_unet_down_blocks_0_attentions_0_to_q.lora_down.weight  → [r, in]
#   lora_unet_down_blocks_0_attentions_0_to_q.lora_up.weight    → [out, r]
#   lora_unet_down_blocks_0_attentions_0_to_q.alpha              → scalar

def _kohya_key_to_unet_path(kohya_key: str) -> Optional[str]:
    """
    Convert a Kohya lora key to a diffusers UNet parameter path.
    e.g. lora_unet_down_blocks_0_attentions_0_to_q
      →  down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q
    This handles the most common patterns; extend as needed.
    """
    # strip prefix and suffix
    key = re.sub(r"^lora_unet_", "", kohya_key)
    key = re.sub(r"\.(lora_down|lora_up|alpha).*$", "", key)

    # underscores → dots, but we must be careful with numeric indices
    # Strategy: replace _<digit> with .<digit> only when following a word char
    key = re.sub(r"_(\d+)", r".\1", key)
    key = key.replace("_", ".")

    return key + ".weight"


def merge_raw_lora(
    unet: UNet2DConditionModel,
    state_dict: Dict[str, torch.Tensor],
    alpha: float,
    dtype: torch.dtype,
) -> UNet2DConditionModel:
    """
    Merge Kohya / A1111-style raw LoRA weights into UNet in-place.
    """
    print("[merge] Detected raw (Kohya/A1111) LoRA format")

    # Group by module path
    modules: Dict[str, Dict] = {}
    for k, v in state_dict.items():
        base = re.sub(r"\.(lora_down|lora_up|alpha)\.weight$", "", k)
        base = re.sub(r"\.alpha$", "", base)
        modules.setdefault(base, {})
        if "lora_down" in k:
            modules[base]["down"] = v
        elif "lora_up" in k:
            modules[base]["up"] = v
        elif k.endswith(".alpha"):
            modules[base]["alpha"] = v.item() if hasattr(v, "item") else float(v)

    unet_sd = dict(unet.named_parameters())
    merged_count = 0
    skipped      = []

    for mod_key, parts in modules.items():
        if "down" not in parts or "up" not in parts:
            skipped.append(mod_key)
            continue

        # Per-layer alpha scaling: scale = (layer_alpha / rank) * user_alpha
        down = parts["down"]
        rank = down.shape[0]
        layer_alpha = parts.get("alpha", rank)   # default to rank if absent
        scale = alpha * (layer_alpha / rank)

        unet_key = _kohya_key_to_unet_path(mod_key)
        if unet_key not in unet_sd:
            # try without .weight
            unet_key_nw = unet_key.replace(".weight", "")
            if unet_key_nw not in unet_sd:
                skipped.append(mod_key)
                continue
            unet_key = unet_key_nw

        param = unet_sd[unet_key]
        A = down.to(dtype=dtype, device=param.device)   # [r, in, ...]
        B = parts["up"].to(dtype=dtype, device=param.device)  # [out, r, ...]

        if A.dim() == 4:
            r = A.shape[0]
            delta = (B.reshape(B.shape[0], r) @ A.reshape(r, -1)).reshape(param.shape)
        else:
            delta = B @ A

        with torch.no_grad():
            param.data.add_(scale * delta)

        merged_count += 1

    print(f"[merge] Raw LoRA: merged {merged_count} modules, skipped {len(skipped)}")
    if skipped:
        print(f"         skipped keys (first 5): {skipped[:5]}")

    return unet


# ─────────────────────────────────────────────────────────────────────────────
# Flavour 3: Full UNet state dict
# ─────────────────────────────────────────────────────────────────────────────

def merge_full_unet(
    unet: UNet2DConditionModel,
    state_dict: Dict[str, torch.Tensor],
    dtype: torch.dtype,
) -> UNet2DConditionModel:
    """
    Direct load of a full UNet state dict — no delta math, just load_state_dict.
    Handles strict=False so partial checkpoints (e.g. LoRA-only layers saved
    as full weights) also work.
    """
    print("[merge] Detected full UNet state dict — loading directly")

    # Cast to target dtype
    cast_sd = {k: v.to(dtype=dtype) for k, v in state_dict.items()}

    missing, unexpected = unet.load_state_dict(cast_sd, strict=False)
    print(f"[merge] Full UNet: missing={len(missing)}, unexpected={len(unexpected)}")
    if missing:
        print(f"         missing (first 5): {missing[:5]}")
    if unexpected:
        print(f"         unexpected (first 5): {unexpected[:5]}")

    return unet


# ─────────────────────────────────────────────────────────────────────────────
# Main merge + save
# ─────────────────────────────────────────────────────────────────────────────

def merge_and_save(
    base_model:  str,
    checkpoint:  str,
    output_dir:  str,
    lora_alpha:  float = 1.0,
    lora_rank:   int   = 64,
    dtype_str:   str   = "float16",
    device:      str   = "cpu",
):
    dtype = DTYPE_MAP[dtype_str]

    # ── 1. Load base pipeline ──────────────────────────────────────────────
    print(f"\n[load] Base model: {base_model}")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        base_model,
        torch_dtype=dtype,
        safety_checker=None,         # disable for alignment training context
        requires_safety_checker=False,
    )
    pipe = pipe.to(device)
    unet: UNet2DConditionModel = pipe.unet
    unet.eval()

    print(f"[load] UNet loaded — {sum(p.numel() for p in unet.parameters()):,} params")

    # ── 2. Load checkpoint ────────────────────────────────────────────────
    state_dict = load_checkpoint(checkpoint)
    print(f"[load] Checkpoint keys: {len(state_dict)}")

    # ── 3. Detect and merge ───────────────────────────────────────────────
    flavour = detect_flavour(state_dict)

    if flavour == "peft_lora":
        unet = merge_peft_lora(unet, state_dict, alpha=lora_alpha, dtype=dtype)
    elif flavour == "raw_lora":
        unet = merge_raw_lora(unet, state_dict, alpha=lora_alpha, dtype=dtype)
    else:
        unet = merge_full_unet(unet, state_dict, dtype=dtype)

    # ── 4. Put merged UNet back into pipeline ─────────────────────────────
    pipe.unet = unet

    # ── 5. Save full merged pipeline ─────────────────────────────────────
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n[save] Writing merged pipeline to: {out}")
    pipe.save_pretrained(str(out))
    print("[save] Done.")

    # ── 6. Verify round-trip load ─────────────────────────────────────────
    print("\n[verify] Round-trip load check...")
    try:
        verify_pipe = StableDiffusionInpaintPipeline.from_pretrained(
            str(out),
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )
        param_count = sum(p.numel() for p in verify_pipe.unet.parameters())
        print(f"[verify] ✓ Reloaded successfully — UNet params: {param_count:,}")
        del verify_pipe
    except Exception as e:
        print(f"[verify] ✗ Round-trip failed: {e}")

    return str(out)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA / full weights into SD1.5 Inpainter")

    parser.add_argument("--base_model",  type=str,   default="runwayml/stable-diffusion-inpainting",
                        help="HuggingFace repo or local path to base SD1.5 inpainter")
    parser.add_argument("--checkpoint",  type=str,   required=True,
                        help="Path to your .safetensors (or .pt) LoRA / full-UNet checkpoint")
    parser.add_argument("--output_dir",  type=str,   required=True,
                        help="Directory to save the merged pipeline")
    parser.add_argument("--lora_alpha",  type=float, default=1.0,
                        help="LoRA merge scale (1.0 = full strength). "
                             "Use <1.0 to partially merge for gradual alignment.")
    parser.add_argument("--lora_rank",   type=int,   default=128,
                        help="LoRA rank — used only if alpha metadata missing from checkpoint")
    parser.add_argument("--dtype",       type=str,   default="float16",
                        choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--device",      type=str,   default="cuda:2",
                        help="'cpu' is safest for merging; 'cuda' is faster")

    args = parser.parse_args()

    merge_and_save(
        base_model  = args.base_model,
        checkpoint  = args.checkpoint,
        output_dir  = args.output_dir,
        lora_alpha  = args.lora_alpha,
        lora_rank   = args.lora_rank,
        dtype_str   = args.dtype,
        device      = args.device,
    )