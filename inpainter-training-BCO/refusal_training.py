"""
SFT Training for Inpainting Refusal Learning
=============================================
Stage 1 of the 3-stage alignment pipeline.
Teach the inpainter WHAT a refusal looks like in pixel space
before BCO/KTO teaches it WHEN to apply that policy.

Dataset columns expected:
  - original_prompt : str  -- text prompt used during inpainting
  - original_image  : PIL | path | bytes | HF Image dict -- the original (unsafe) image
  - inpainted_image : PIL | path | bytes | HF Image dict -- the human-approved "refused" output

All rows are assumed unsafe -- no label filtering is performed.

Recommended launch (single A6000, 48 GB):

  # From Hugging Face Hub (private dataset):
  accelerate launch sft_inpainting_refusal.py \\
    --hf_dataset   your-org/your-dataset \\
    --hf_token     hf_xxxxxxxxxxxxxxxxxxxx \\
    --output_dir   ./sft_refusal_ckpt \\
    --num_epochs   8 \\
    --batch_size   8 \\
    --gradient_accumulation_steps 2 \\
    --lora_rank    64 \\
    --lora_alpha   64 \\
    --learning_rate 1e-4

  # From a local file (parquet or csv):
  accelerate launch sft_inpainting_refusal.py \\
    --dataset_path ./data.parquet \\
    --output_dir   ./sft_refusal_ckpt \\
    --num_epochs   8 \\
    --batch_size   8 \\
    --gradient_accumulation_steps 2 \\
    --lora_rank    64 \\
    --lora_alpha   64 \\
    --learning_rate 1e-4

Effective batch = 8 x 2 = 16. With 2000 samples:
  steps/epoch  = ceil(2000 / 16) = 125
  total steps  = 125 x 8         = 1000

Loss should fall from ~0.15 to ~0.06-0.08 by epoch 5.
If still declining at epoch 8, extend to 10-12.
"""

import io
import os
import math
import time
import logging
import argparse
from pathlib import Path
from typing import Optional
from datetime import timedelta

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from diffusers import (
    StableDiffusionInpaintPipeline,
    AutoencoderKL,
    UNet2DConditionModel,
    DDPMScheduler,
)
from diffusers.optimization import get_scheduler
from transformers import CLIPTokenizer, CLIPTextModel

from peft import LoraConfig, get_peft_model, PeftModel

logger = get_logger(__name__, log_level="INFO")


# ============================================================================
# Logging helpers
# ============================================================================

def log_separator(accelerator, char="─", width=70):
    if accelerator.is_main_process:
        logger.info(char * width)


def log_section(accelerator, title: str):
    if accelerator.is_main_process:
        logger.info("")
        logger.info("+" + "-" * 68 + "+")
        logger.info(f"|  {title:<66}|")
        logger.info("+" + "-" * 68 + "+")


def format_duration(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


def log_dataset_stats(accelerator, df: pd.DataFrame, source: str):
    if not accelerator.is_main_process:
        return
    log_section(accelerator, "Dataset Summary")
    logger.info(f"  Source        : {source}")
    logger.info(f"  Total rows    : {len(df):,}")
    logger.info(f"  Columns       : {list(df.columns)}")
    sample_prompts = df["original_prompt"].dropna().head(3).tolist()
    for i, p in enumerate(sample_prompts):
        logger.info(f"  prompt[{i}]     : {str(p)[:80]}")


def log_training_plan(
    accelerator,
    args,
    dataset_size: int,
    steps_per_epoch: int,
    total_steps: int,
    trainable_params: int,
    total_params: int,
):
    if not accelerator.is_main_process:
        return
    log_section(accelerator, "Training Plan")
    effective_batch = (
        args.batch_size * args.gradient_accumulation_steps * accelerator.num_processes
    )
    dataset_source = args.hf_dataset if args.hf_dataset else args.dataset_path
    logger.info(f"  Model            : {args.model_id}")
    logger.info(f"  Dataset          : {dataset_source}")
    logger.info(f"  Dataset size     : {dataset_size:,} samples")
    logger.info(f"  Epochs           : {args.num_epochs}")
    logger.info(f"  Batch (device)   : {args.batch_size}")
    logger.info(f"  Grad accum steps : {args.gradient_accumulation_steps}")
    logger.info(f"  Num processes    : {accelerator.num_processes}")
    logger.info(f"  Effective batch  : {effective_batch}")
    logger.info(f"  Steps / epoch    : {steps_per_epoch}")
    logger.info(f"  Total steps      : {total_steps}")
    logger.info(f"  Learning rate    : {args.learning_rate}")
    logger.info(f"  LR scheduler     : {args.lr_scheduler}")
    logger.info(f"  Warmup steps     : {args.lr_warmup_steps}")
    logger.info(f"  Mixed precision  : {args.mixed_precision}")
    logger.info(f"  LoRA rank/alpha  : {args.lora_rank} / {args.lora_alpha}")
    logger.info(
        f"  Trainable params : {trainable_params:,} / {total_params:,}  "
        f"({100 * trainable_params / total_params:.2f}%)"
    )
    logger.info(f"  Image size       : {args.image_size}")
    logger.info(f"  SNR gamma        : {args.snr_gamma}")
    logger.info(f"  Noise offset     : {args.noise_offset}")
    log_separator(accelerator)


def log_epoch_end(
    accelerator,
    epoch: int,
    num_epochs: int,
    avg_loss: float,
    best_loss: float,
    epoch_duration: float,
    global_step: int,
):
    if not accelerator.is_main_process:
        return
    improved = "  <-- new best" if avg_loss <= best_loss else ""
    logger.info(
        f"  Epoch {epoch+1:>2}/{num_epochs}"
        f"  |  avg_loss={avg_loss:.5f}"
        f"  |  best={best_loss:.5f}{improved}"
        f"  |  step={global_step}"
        f"  |  time={format_duration(epoch_duration)}"
    )


def log_step(accelerator, global_step: int, loss: float, lr: float, log_every: int = 25):
    if not accelerator.is_main_process:
        return
    if global_step % log_every == 0:
        logger.info(f"    step={global_step:>5}  loss={loss:.5f}  lr={lr:.3e}")


# ============================================================================
# Dataset
# ============================================================================

class InpaintingRefusalDataset(Dataset):
    """
    All rows are treated as refusal training examples.
    No label filtering -- caller passes a clean dataframe.

    Mask derivation:
      - If mask_col is provided and non-null, use that column.
      - Otherwise auto-derive by pixel difference between original and inpainted.
        Regions that changed are the masked (edited) region.

    Image column dtype support:
      - PIL.Image.Image
      - File path (str / Path)
      - Raw bytes / bytearray
      - HuggingFace Image feature dict: {'bytes': b'...', 'path': str | None}
        This is the format produced by datasets.load_dataset() + .to_pandas()
        when the column dtype is Image().

    Args:
        dataframe       : pd.DataFrame with columns [original_prompt, original_image,
                          inpainted_image]
        tokenizer       : CLIP tokenizer
        image_size      : target resolution (512 for SD1.5)
        mask_threshold  : pixel diff threshold (0-255) for auto mask derivation.
                          Lower = more of the image treated as masked.
                          Start at 10; increase if masks look too noisy/large.
        mask_col        : optional column name containing explicit mask paths/PIL
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer: CLIPTokenizer,
        image_size: int = 512,
        mask_threshold: int = 10,
        mask_col: Optional[str] = None,
    ):
        self.df             = dataframe.reset_index(drop=True)
        self.tokenizer      = tokenizer
        self.image_size     = image_size
        self.mask_threshold = mask_threshold
        self.mask_col       = mask_col

        self._log_mask_stats()

    def _log_mask_stats(self):
        """Derive masks for a small sample and log average coverage."""
        sample_size = min(20, len(self.df))
        coverages = []
        for i in range(sample_size):
            row = self.df.iloc[i]
            try:
                orig    = self._resize(self._load_image(row["original_image"]))
                inpaint = self._resize(self._load_image(row["inpainted_image"]))
                mask    = self._derive_mask(orig, inpaint)
                coverages.append(np.array(mask).mean() / 255.0)
            except Exception:
                pass
        if coverages:
            logger.info(
                f"  Mask coverage sample (n={sample_size}, threshold={self.mask_threshold}): "
                f"mean={np.mean(coverages)*100:.1f}%  "
                f"min={np.min(coverages)*100:.1f}%  "
                f"max={np.max(coverages)*100:.1f}%"
            )

    def __len__(self):
        return len(self.df)

    def _load_image(self, val) -> Image.Image:
        """
        Load an image from any of the supported formats:

          - PIL.Image          : returned as-is (converted to RGB)
          - bytes / bytearray  : decoded via BytesIO
          - dict               : HuggingFace Image feature {'bytes': ..., 'path': ...}
                                 produced by datasets.to_pandas() when column dtype=Image().
                                 'bytes' is tried first; 'path' is the fallback.
          - str / Path         : treated as a file path
        """
        # Already a PIL image
        if isinstance(val, Image.Image):
            return val.convert("RGB")

        # Raw bytes (e.g. from a parquet binary column)
        if isinstance(val, (bytes, bytearray)):
            return Image.open(io.BytesIO(val)).convert("RGB")

        # HuggingFace Image feature dict: {'bytes': b'...', 'path': str | None}
        # This is the native format when dtype=Image() columns are converted to pandas
        if isinstance(val, dict):
            if val.get("bytes"):
                return Image.open(io.BytesIO(val["bytes"])).convert("RGB")
            if val.get("path"):
                return Image.open(val["path"]).convert("RGB")
            raise ValueError(
                f"Image dict has neither 'bytes' nor 'path'. keys={list(val.keys())}"
            )

        # File path (str or Path)
        return Image.open(str(val)).convert("RGB")

    def _resize(self, img: Image.Image) -> Image.Image:
        return img.resize((self.image_size, self.image_size), Image.LANCZOS)

    def _derive_mask(self, orig: Image.Image, inpainted: Image.Image) -> Image.Image:
        """
        White pixels = region that was edited = the masked / refused region.
        Max diff across RGB channels so hue-only changes are also captured.
        """
        o = np.array(orig).astype(np.int32)
        i = np.array(inpainted).astype(np.int32)
        diff = np.abs(o - i).max(axis=-1)           # (H, W)
        mask = (diff > self.mask_threshold).astype(np.uint8) * 255
        return Image.fromarray(mask, mode="L")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        orig_img    = self._resize(self._load_image(row["original_image"]))
        inpaint_img = self._resize(self._load_image(row["inpainted_image"]))

        if self.mask_col and self.mask_col in row.index and row[self.mask_col] is not None:
            mask_img = self._load_image(row[self.mask_col]).convert("L")
            mask_img = mask_img.resize((self.image_size, self.image_size), Image.NEAREST)
        else:
            mask_img = self._derive_mask(orig_img, inpaint_img)

        # Images: (3, H, W) in [-1, 1]
        to_tensor = lambda img: (
            torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 127.5 - 1.0
        )
        # Mask: (1, H, W) in [0, 1]  where 1.0 = masked region
        mask_to_tensor = lambda m: (
            torch.from_numpy(np.array(m)).unsqueeze(0).float() / 255.0
        )

        orig_tensor    = to_tensor(orig_img)
        inpaint_tensor = to_tensor(inpaint_img)    # <-- training TARGET
        mask_tensor    = mask_to_tensor(mask_img)

        prompt = str(row["original_prompt"]) if pd.notna(row["original_prompt"]) else ""
        tokens = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        return {
            "original_pixel_values":  orig_tensor,
            "inpainted_pixel_values": inpaint_tensor,
            "mask":                   mask_tensor,
            "input_ids":              tokens.input_ids.squeeze(0),
            "attention_mask":         tokens.attention_mask.squeeze(0),
            "prompt":                 prompt,
        }


# ============================================================================
# LoRA
# ============================================================================

# Why include conv layers?
#
# Attention-only LoRA is fine for: style, prompt-following, minor attribute edits.
# Refusal requires STRUCTURAL changes to the masked region -- drawing clothing,
# blurring a face, filling with abstract content. Those are primarily handled by
# the ResNet conv layers (conv1, conv2) and the 9-channel input projection
# (conv_in). On a 48 GB A6000 with rank=64, including these costs nothing
# and is the right call for this task.
LORA_TARGET_MODULES = [
    # Cross- and self-attention
    "to_q", "to_k", "to_v", "to_out.0",
    # Feed-forward in transformer blocks
    "ff.net.0.proj", "ff.net.2",
    # Spatial projection layers
    "proj_in", "proj_out",
    # ResNet conv layers -- handle structural / spatial output changes
    "conv1", "conv2",
    # Input conv (receives the full 9-channel inpainting tensor)
    "conv_in",
]


def apply_lora_to_unet(
    unet: UNet2DConditionModel,
    rank: int = 64,
    alpha: float = 64.0,
) -> UNet2DConditionModel:
    """
    rank=64, alpha=64  =>  effective scale = alpha/rank = 1.0
    Standard "no implicit scaling" setting. For a genuinely new concept
    (refusal) you want the full representational capacity of rank 64.
    """
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=0.05,   # small dropout prevents memorisation
        bias="none",
    )
    unet = get_peft_model(unet, lora_config)

    trainable = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in unet.parameters())

    logger.info(f"  LoRA applied  rank={rank}  alpha={alpha}")
    logger.info(f"  Target modules : {LORA_TARGET_MODULES}")
    logger.info(
        f"  Trainable      : {trainable:,} / {total:,}  "
        f"({100*trainable/total:.2f}%)"
    )
    return unet


# ============================================================================
# Diffusion helpers
# ============================================================================

def encode_images(vae, pixel_values: torch.Tensor, dtype) -> torch.Tensor:
    with torch.no_grad():
        latents = vae.encode(pixel_values.to(dtype=dtype)).latent_dist.sample()
    return latents * vae.config.scaling_factor


def downsample_mask(mask: torch.Tensor, latent_size: int) -> torch.Tensor:
    return F.interpolate(mask, size=(latent_size, latent_size), mode="nearest")


def compute_snr(noise_scheduler, timesteps: torch.Tensor) -> torch.Tensor:
    alphas     = noise_scheduler.alphas_cumprod.to(timesteps.device)
    sqrt_a     = alphas[timesteps] ** 0.5
    sqrt_one_m = (1 - alphas[timesteps]) ** 0.5
    return (sqrt_a / sqrt_one_m) ** 2


# ============================================================================
# Args
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="SFT training for inpainting refusal -- Stage 1 of 3",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    p.add_argument("--model_id",   default="runwayml/stable-diffusion-inpainting",
                   help="HuggingFace model ID or local path for the base inpainting model.")
    p.add_argument("--lora_rank",  type=int,   default=64,
                   help="LoRA rank. 64 recommended for A6000 48GB SFT.")
    p.add_argument("--lora_alpha", type=float, default=64.0,
                   help="LoRA alpha. Keep equal to rank => scale=1.0.")

    # ── Data source (HF Hub OR local file -- one is required) ────────────────
    data_group = p.add_mutually_exclusive_group(required=True)
    data_group.add_argument(
        "--hf_dataset",
        default=None,
        metavar="ORG/DATASET",
        help="Hugging Face dataset name, e.g. 'your-org/your-dataset'. "
             "Pass --hf_token as well for private repos.",
    )
    data_group.add_argument(
        "--dataset_path",
        default=None,
        metavar="PATH",
        help="Path to a local .parquet or .csv file. "
             "Required columns: original_prompt, original_image, inpainted_image.",
    )

    p.add_argument(
        "--hf_token",
        default=None,
        metavar="hf_xxx",
        help="Hugging Face access token for private datasets. "
             "Only used when --hf_dataset is set. "
             "Alternatively set the HF_TOKEN environment variable.",
    )
    p.add_argument(
        "--hf_split",
        default="train",
        help="Which dataset split to load when using --hf_dataset. Default: train.",
    )

    p.add_argument("--image_size",     type=int, default=512)
    p.add_argument("--mask_threshold", type=int, default=10,
                   help="Pixel diff threshold for auto mask derivation (0-255). "
                        "Raise if auto-masks are too noisy.")
    p.add_argument("--mask_col",       default=None,
                   help="Optional: column name with explicit mask paths.")

    # Training
    p.add_argument("--output_dir",    default="./sft_refusal_checkpoint")
    p.add_argument("--num_epochs",    type=int,   default=8,
                   help="8 epochs over 2000 samples = ~1000 gradient steps "
                        "(effective batch 16). Extend to 10-12 if loss still declining.")
    p.add_argument("--batch_size",    type=int,   default=8,
                   help="Per-device batch. 8 fits A6000 48GB at 512px fp16.")
    p.add_argument("--gradient_accumulation_steps", type=int, default=2,
                   help="Effective batch = batch_size x this. Default gives 16.")
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--lr_scheduler",  default="cosine",
                   choices=["cosine", "linear", "constant", "constant_with_warmup"])
    p.add_argument("--lr_warmup_steps",     type=int,   default=50)
    p.add_argument("--weight_decay",        type=float, default=1e-2)
    p.add_argument("--max_grad_norm",       type=float, default=1.0)
    p.add_argument("--mixed_precision",     default="fp16",
                   choices=["no", "fp16", "bf16"])
    p.add_argument("--gradient_checkpointing", action="store_true", default=True,
                   help="Saves ~8 GB VRAM at slight speed cost. Recommended.")
    p.add_argument("--seed",                type=int, default=42)
    p.add_argument("--save_every_n_epochs", type=int, default=2)
    p.add_argument("--num_workers",         type=int, default=4)
    p.add_argument("--log_every_n_steps",   type=int, default=25,
                   help="Log loss/lr to console every N gradient steps.")

    # Diffusion
    p.add_argument("--snr_gamma",    type=float, default=5.0,
                   help="Min-SNR-gamma loss weighting (Hang et al. 2023). "
                        "Reduces over-weighting of high-noise timesteps. "
                        "Set 0 to disable.")
    p.add_argument("--noise_offset", type=float, default=0.05,
                   help="Small noise offset (0.05) helps with dark/saturated "
                        "regions common in clothing inpainting.")

    return p.parse_args()


# ============================================================================
# Dataset loading
# ============================================================================

def load_dataframe(args) -> tuple[pd.DataFrame, str]:
    """
    Load the dataset as a pandas DataFrame from either:
      - Hugging Face Hub  (--hf_dataset + optional --hf_token)
      - Local file        (--dataset_path, parquet or csv)

    Image columns with dtype=Image() are kept as HF dicts {'bytes': ..., 'path': ...}.
    The InpaintingRefusalDataset._load_image() method handles this format natively,
    decoding bytes on demand per sample rather than all at once.

    Returns (df, source_description_string).
    """
    if args.hf_dataset:
        # Resolve token: CLI arg > HF_TOKEN env var > None (public dataset)
        token = args.hf_token or os.environ.get("HF_TOKEN", None)

        logger.info(f"  Loading dataset from Hugging Face Hub: {args.hf_dataset}")
        logger.info(f"  Split  : {args.hf_split}")
        logger.info(f"  Token  : {'provided' if token else 'not provided (public dataset)'}")

        from datasets import load_dataset as hf_load_dataset
        hf_ds = hf_load_dataset(
            args.hf_dataset,
            token=token,
            split=args.hf_split,
        )
        # NOTE: intentionally NOT casting Image() columns to PIL before to_pandas().
        # Columns with dtype=Image() become dicts {'bytes': b'...', 'path': str|None}
        # in the DataFrame. _load_image() handles this format directly, decoding
        # bytes on demand per sample to keep peak RAM low during multi-worker loading.
        df = hf_ds.to_pandas()
        source = f"HuggingFace Hub -- {args.hf_dataset} (split={args.hf_split})"

    else:
        ext = Path(args.dataset_path).suffix.lower()
        logger.info(f"  Loading dataset from local file: {args.dataset_path}")
        if ext == ".parquet":
            df = pd.read_parquet(args.dataset_path)
        elif ext in (".csv", ".tsv"):
            df = pd.read_csv(args.dataset_path)
        else:
            raise ValueError(
                f"Unsupported local file format: {ext}. Use .parquet or .csv"
            )
        source = f"Local file -- {args.dataset_path}"

    # Validate required columns
    required = {"original_prompt", "original_image", "inpainted_image"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )

    return df, source


# ============================================================================
# Training
# ============================================================================

def train(args):

    # Accelerator
    log_dir     = Path(args.output_dir) / "logs"
    project_cfg = ProjectConfiguration(
        project_dir=args.output_dir,
        logging_dir=str(log_dir),
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        project_config=project_cfg,
    )

    logging.basicConfig(
        format="%(asctime)s  %(levelname)-8s  %(name)s -- %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )

    set_seed(args.seed)

    weight_dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }.get(args.mixed_precision, torch.float32)

    log_section(accelerator, "SFT Inpainting Refusal  --  Stage 1 of 3")
    logger.info(f"  Device          : {accelerator.device}")
    logger.info(f"  Mixed precision : {args.mixed_precision} ({weight_dtype})")
    logger.info(f"  Num processes   : {accelerator.num_processes}")

    # Load models
    log_section(accelerator, "Loading Models")
    logger.info(f"  Tokenizer + text encoder from {args.model_id} ...")
    tokenizer    = CLIPTokenizer.from_pretrained(args.model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.model_id, subfolder="text_encoder")

    logger.info("  VAE ...")
    vae = AutoencoderKL.from_pretrained(args.model_id, subfolder="vae")

    logger.info("  UNet ...")
    unet = UNet2DConditionModel.from_pretrained(args.model_id, subfolder="unet")

    logger.info("  Noise scheduler ...")
    noise_scheduler = DDPMScheduler.from_pretrained(args.model_id, subfolder="scheduler")
    logger.info(
        f"  Scheduler: {noise_scheduler.config.num_train_timesteps} timesteps, "
        f"prediction_type={noise_scheduler.config.prediction_type}"
    )

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    logger.info("  VAE + text encoder frozen.")

    # Apply LoRA
    log_section(accelerator, "Applying LoRA to UNet")
    unet = apply_lora_to_unet(unet, rank=args.lora_rank, alpha=args.lora_alpha)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        logger.info("  Gradient checkpointing: enabled")

    # Load dataset
    log_section(accelerator, "Loading Dataset")
    df, source = load_dataframe(args)
    log_dataset_stats(accelerator, df, source)

    dataset = InpaintingRefusalDataset(
        dataframe=df,
        tokenizer=tokenizer,
        image_size=args.image_size,
        mask_threshold=args.mask_threshold,
        mask_col=args.mask_col,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
    )

    # Optimizer & LR scheduler
    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    steps_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    total_steps     = args.num_epochs * steps_per_epoch

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=total_steps * accelerator.num_processes,
    )

    # Accelerate prepare
    unet, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, dataloader, lr_scheduler
    )

    total_params    = sum(p.numel() for p in unet.parameters())
    trainable_count = sum(p.numel() for p in unet.parameters() if p.requires_grad)

    log_training_plan(
        accelerator, args,
        dataset_size=len(dataset),
        steps_per_epoch=steps_per_epoch,
        total_steps=total_steps,
        trainable_params=trainable_count,
        total_params=total_params,
    )

    accelerator.init_trackers("sft_refusal", config=vars(args))

    latent_size = args.image_size // 8   # 64 for 512px images

    # =========================================================================
    # Training loop
    # =========================================================================
    log_section(accelerator, "Training")

    global_step  = 0
    best_loss    = float("inf")
    train_start  = time.time()

    for epoch in range(args.num_epochs):
        unet.train()
        epoch_loss_sum   = 0.0
        epoch_step_count = 0
        epoch_start      = time.time()

        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch+1:>2}/{args.num_epochs}",
            disable=not accelerator.is_local_main_process,
            dynamic_ncols=True,
        )

        for batch in progress_bar:
            with accelerator.accumulate(unet):

                # 1. Encode images to latent space
                target_latents = encode_images(
                    vae, batch["inpainted_pixel_values"], weight_dtype
                )
                orig_latents = encode_images(
                    vae, batch["original_pixel_values"], weight_dtype
                )

                # 2. Prepare mask at latent resolution + masked conditioning
                mask_latents        = downsample_mask(batch["mask"], latent_size).to(weight_dtype)
                masked_orig_latents = orig_latents * (1 - mask_latents)

                # 3. Sample noise and timesteps
                noise = torch.randn_like(target_latents)
                if args.noise_offset > 0:
                    noise += args.noise_offset * torch.randn(
                        target_latents.shape[0], target_latents.shape[1], 1, 1,
                        device=noise.device,
                    )

                bsz       = target_latents.shape[0]
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (bsz,), device=target_latents.device,
                ).long()

                noisy_latents = noise_scheduler.add_noise(target_latents, noise, timesteps)

                # 4. Build 9-channel UNet input
                #    [noisy_target (4ch) | mask (1ch) | masked_orig (4ch)]
                unet_input = torch.cat(
                    [noisy_latents, mask_latents, masked_orig_latents], dim=1
                )

                # 5. Text conditioning (frozen)
                with torch.no_grad():
                    encoder_hidden_states = text_encoder(
                        batch["input_ids"].to(accelerator.device),
                        attention_mask=batch["attention_mask"].to(accelerator.device),
                    ).last_hidden_state

                # 6. UNet forward pass
                model_pred = unet(
                    unet_input.to(weight_dtype),
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states.to(weight_dtype),
                ).sample

                # 7. Prediction target
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(target_latents, noise, timesteps)
                else:
                    raise ValueError(
                        f"Unknown prediction_type: {noise_scheduler.config.prediction_type}"
                    )

                # 8. Loss with optional min-SNR weighting
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                loss = loss.mean(dim=[1, 2, 3])   # (B,) per-sample

                if args.snr_gamma and args.snr_gamma > 0:
                    snr     = compute_snr(noise_scheduler, timesteps)
                    weights = torch.clamp(snr, max=args.snr_gamma) / snr
                    loss    = (loss * weights).mean()
                else:
                    loss = loss.mean()

                # 9. Backward + optimiser step
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Step-level logging
            if accelerator.sync_gradients:
                global_step      += 1
                step_loss         = loss.detach().item()
                current_lr        = lr_scheduler.get_last_lr()[0]
                epoch_loss_sum   += step_loss
                epoch_step_count += 1

                progress_bar.set_postfix(
                    loss=f"{step_loss:.4f}",
                    lr=f"{current_lr:.2e}",
                    step=global_step,
                )
                accelerator.log(
                    {
                        "train/loss":        step_loss,
                        "train/lr":          current_lr,
                        "train/epoch":       epoch + 1,
                        "train/global_step": global_step,
                    },
                    step=global_step,
                )
                log_step(
                    accelerator, global_step, step_loss, current_lr,
                    log_every=args.log_every_n_steps,
                )

        # Epoch-level logging
        avg_loss   = epoch_loss_sum / max(epoch_step_count, 1)
        best_loss  = min(best_loss, avg_loss)
        epoch_time = time.time() - epoch_start
        elapsed    = time.time() - train_start
        eta        = (elapsed / (epoch + 1)) * (args.num_epochs - epoch - 1)

        log_epoch_end(
            accelerator, epoch, args.num_epochs,
            avg_loss, best_loss, epoch_time, global_step,
        )
        if accelerator.is_main_process:
            logger.info(
                f"  Elapsed: {format_duration(elapsed)}"
                f"  |  ETA: {format_duration(eta)}"
            )

        accelerator.log({"train/epoch_avg_loss": avg_loss}, step=global_step)

        # Checkpoint
        if (epoch + 1) % args.save_every_n_epochs == 0:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                ckpt_dir = Path(args.output_dir) / f"checkpoint-epoch-{epoch+1}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                unwrapped = accelerator.unwrap_model(unet)
                unwrapped.save_pretrained(str(ckpt_dir / "unet_lora"))
                logger.info(f"  Checkpoint saved --> {ckpt_dir / 'unet_lora'}")

    # Final save
    log_section(accelerator, "Saving Final Checkpoint")
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_dir = Path(args.output_dir) / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        unwrapped = accelerator.unwrap_model(unet)
        unwrapped.save_pretrained(str(final_dir / "unet_lora"))

        total_time = time.time() - train_start
        logger.info(f"  LoRA weights saved  --> {final_dir / 'unet_lora'}")
        logger.info(f"  Best avg loss       :   {best_loss:.5f}")
        logger.info(f"  Total training time :   {format_duration(total_time)}")
        logger.info("")
        logger.info("  Next: run merge_lora_for_bco() to produce the BCO base model.")

    accelerator.end_training()


# ============================================================================
# Post-SFT utilities
# ============================================================================

def merge_lora_for_bco(
    base_model_id: str,
    lora_checkpoint_dir: str,
    output_dir: str,
    torch_dtype=torch.float16,
):
    """
    Merge SFT LoRA into the base UNet and save a full pipeline checkpoint.
    The merged checkpoint is your Stage 2 (BCO/KTO) base model.

    Usage:
        from sft_inpainting_refusal import merge_lora_for_bco
        merge_lora_for_bco(
            base_model_id       = "runwayml/stable-diffusion-inpainting",
            lora_checkpoint_dir = "./sft_refusal_checkpoint/final/unet_lora",
            output_dir          = "./sft_merged_for_bco",
        )
    """
    print(f"Loading base UNet from {base_model_id} ...")
    base_unet = UNet2DConditionModel.from_pretrained(base_model_id, subfolder="unet")

    print(f"Loading LoRA from {lora_checkpoint_dir} ...")
    peft_unet = PeftModel.from_pretrained(base_unet, lora_checkpoint_dir)

    print("Merging LoRA weights into base UNet ...")
    merged_unet = peft_unet.merge_and_unload()

    print("Saving merged pipeline ...")
    pipeline = StableDiffusionInpaintPipeline.from_pretrained(
        base_model_id,
        unet=merged_unet,
        torch_dtype=torch_dtype,
    )
    pipeline.save_pretrained(output_dir)
    print(f"Merged SFT pipeline saved --> {output_dir}")
    print("This is now your BCO Stage 2 base model.")


def run_inference_check(
    checkpoint_dir: str,
    test_image_path: str,
    test_mask_path:  str,
    prompt:          str   = "a person wearing a shirt",
    output_path:     str   = "sft_check_output.png",
    num_steps:       int   = 50,
    guidance_scale:  float = 7.5,
):
    """
    Visual sanity check after training. Pass an unsafe image + its mask;
    the SFT model should now produce clothing/blurring rather than nudity.
    """
    pipeline = StableDiffusionInpaintPipeline.from_pretrained(
        checkpoint_dir, torch_dtype=torch.float16,
    ).to("cuda")

    image = Image.open(test_image_path).convert("RGB").resize((512, 512))
    mask  = Image.open(test_mask_path).convert("L").resize((512, 512))

    result = pipeline(
        prompt=prompt,
        image=image,
        mask_image=mask,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
    ).images[0]

    result.save(output_path)
    print(f"Inference check saved --> {output_path}")


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    args = parse_args()
    train(args)