#!/usr/bin/env python3
"""
Unified KTO data pipeline:
1) Build mask dataset with auditor heatmaps (+ optional face exclusion for nudity samples)
2) Convert mask dataset into latent parquet shards
3) Upload both outputs to Hugging Face datasets

This script intentionally reuses infer_train_new.py utilities for auditor loading/inference.
"""

import argparse
import io
import os
import tempfile
import math
import time
from dataclasses import dataclass
from itertools import islice
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image
from datasets import Dataset, Features, Image as HFImage, Value, load_dataset
from diffusers import StableDiffusionInpaintPipeline
from huggingface_hub import HfApi, login
from scipy.ndimage import gaussian_filter
from torchvision import transforms as T
from tqdm.auto import tqdm
from transformers import pipeline as hf_pipeline

# Reuse the auditor inference module with first-run artifact download.
import auditor_inference as auditor_module


def ensure_pil_image(value, mode: str = "RGB") -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert(mode)

    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return Image.open(io.BytesIO(value["bytes"])).convert(mode)
        if value.get("path"):
            return Image.open(value["path"]).convert(mode)

    raise TypeError(f"Unsupported image type: {type(value)}")


def pil_to_png_bytes(img: Image.Image) -> bytes:
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def dilate_mask(mask_pil: Image.Image, radius: int = 8) -> Image.Image:
    arr = np.array(mask_pil) > 127
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(arr.astype(np.uint8), kernel, iterations=max(0, int(radius)))
    return Image.fromarray((dilated * 255).astype(np.uint8))


def random_dilate(mask_pil: Image.Image, low: int = 4, high: int = 15) -> Image.Image:
    radius = np.random.randint(low, high)
    return dilate_mask(mask_pil, radius=radius)


@dataclass
class MaskShardUploader:
    api: HfApi
    repo_id: str
    shard_size: int
    features: Features
    path_prefix: str = "data"

    def __post_init__(self):
        self.rows: List[Dict] = []
        self.shard_idx = self._existing_shard_count()

    def _existing_shard_count(self) -> int:
        try:
            files = self.api.list_repo_files(self.repo_id, repo_type="dataset")
            return len([f for f in files if f.startswith(f"{self.path_prefix}/") and f.endswith(".parquet")])
        except Exception:
            return 0

    def add(self, row: Dict) -> None:
        self.rows.append(row)
        if len(self.rows) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self.rows:
            return

        ds = Dataset.from_list(self.rows, features=self.features,num_proc=1)
        buffer = io.BytesIO()
        ds.to_parquet(buffer)
        buffer.seek(0)

            path_in_repo = f"{self.path_prefix}/train-{self.shard_idx:05d}-of-NNNNN.parquet"
            t2 = time.time()
            self.api.upload_file(
                path_or_fileobj=tmp_path,
                path_in_repo=path_in_repo,
                repo_id=self.repo_id,
                repo_type="dataset",
                commit_message=f"Upload shard {self.shard_idx}",
            )
            print(f"[MASK] Upload finished in {time.time() - t2:.1f}s")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        elapsed = time.time() - flush_started
        print(f"[MASK] Uploaded shard {self.shard_idx} -> {path_in_repo} ({elapsed:.1f}s total)")

        self.shard_idx += 1
        self.rows = []


@dataclass
class LatentShardUploader:
    api: HfApi
    repo_id: str
    split: str
    shard_size: int

    def __post_init__(self):
        self.rows: List[Dict] = []
        self.shard_idx = self._existing_shard_count()

    def _existing_shard_count(self) -> int:
        try:
            files = self.api.list_repo_files(self.repo_id, repo_type="dataset")
            return len([f for f in files if f.startswith(f"{self.split}/") and f.endswith(".parquet")])
        except Exception:
            return 0

    def add(self, row: Dict) -> None:
        self.rows.append(row)
        if len(self.rows) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self.rows:
            return

        table = pa.Table.from_pylist(self.rows)
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            pq.write_table(table, tmp_path, compression="zstd")
            path_in_repo = f"{self.split}/shard-{self.shard_idx:05d}.parquet"
            self.api.upload_file(
                path_or_fileobj=tmp_path,
                path_in_repo=path_in_repo,
                repo_id=self.repo_id,
                repo_type="dataset",
                commit_message=f"Upload {self.split} latent shard {self.shard_idx}",
            )
            print(f"Uploaded latent shard: {path_in_repo}")
            self.shard_idx += 1
            self.rows = []
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


class AuditorRunner:
    def __init__(
        self,
        checkpoint: str,
        vocab: str,
        device: str = "auto",
        model_url: str = auditor_module.DEFAULT_MODEL_URL,
        vocab_url: str = auditor_module.DEFAULT_VOCAB_URL,
    ):
        if device == "auto":
            torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            torch_device = torch.device(device)

        auditor_module.DEVICE = torch_device
        self.auditor = auditor_module.AdversarialAuditor(
            model_path=os.path.abspath(checkpoint),
            vocab_path=os.path.abspath(vocab),
            model_url=model_url,
            vocab_url=vocab_url,
        )

    def infer_heatmap(self, image: Image.Image, prompt: str) -> np.ndarray:
        result = self.auditor.audit(image_input=image, prompt=prompt)
        heatmap = result.get("adversarial_heatmap")
        if heatmap is None:
            raise RuntimeError("Auditor inference did not return 'adversarial_heatmap'.")
        return np.asarray(heatmap)


class FaceParser:
    FACE_LABEL_KEYWORDS = {
        "face",
        "skin",
        "nose",
        "eye",
        "brow",
        "ear",
        "mouth",
        "lip",
        "neck",
        "forehead",
    }

    def __init__(self, model_id: str, device: str = "auto"):
        if device == "auto":
            use_gpu = torch.cuda.is_available()
        else:
            use_gpu = device == "cuda"

        pipe_device = 0 if use_gpu else -1
        self.pipe = hf_pipeline("image-segmentation", model=model_id, device=pipe_device)

    def _is_face_label(self, label: str) -> bool:
        label = label.lower().strip()
        return any(keyword in label for keyword in self.FACE_LABEL_KEYWORDS)

    def face_mask(self, image: Image.Image) -> np.ndarray:
        outputs = self.pipe(image)
        mask = np.zeros((image.height, image.width), dtype=np.uint8)

        for item in outputs:
            label = str(item.get("label", "")).lower()
            if not self._is_face_label(label):
                continue

            seg = item.get("mask")
            if seg is None:
                continue

            seg_arr = np.array(seg)
            if seg_arr.ndim == 3:
                seg_arr = seg_arr[..., 0]
            seg_bin = (seg_arr > 0).astype(np.uint8)
            if seg_bin.shape != mask.shape:
                seg_bin = cv2.resize(seg_bin, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
            mask = np.maximum(mask, seg_bin)

        return mask


def collect_existing_ids(repo_id: str) -> set:
    try:
        ds = load_dataset(repo_id, split="train", columns=["id"])
        return set(str(x) for x in ds["id"])
    except Exception:
        return set()


def build_mask_from_heatmap(
    heatmap: np.ndarray,
    percentile: float,
    dilate_kernel: int,
    dilate_iters: int,
) -> np.ndarray:
    threshold = np.percentile(heatmap, percentile)
    binary = (heatmap >= threshold).astype(np.uint8)
    kernel = np.ones((dilate_kernel, dilate_kernel), np.uint8)
    return cv2.dilate(binary, kernel, iterations=dilate_iters)


def resize_heatmap_to_image(
    heatmap: np.ndarray,
    target_hw: Tuple[int, int],
    interpolation: str = "bilinear",
) -> np.ndarray:
    interpolation_map = {
        "nearest": cv2.INTER_NEAREST,
        "bilinear": cv2.INTER_LINEAR,
        "bicubic": cv2.INTER_CUBIC,
    }
    interp = interpolation_map.get(interpolation, cv2.INTER_LINEAR)

    target_h, target_w = target_hw
    if heatmap.shape[:2] == (target_h, target_w):
        return heatmap.astype(np.float32)

    resized = cv2.resize(heatmap.astype(np.float32), (target_w, target_h), interpolation=interp)
    return resized


def apply_face_exclusion(
    auditor_mask: np.ndarray,
    face_mask: np.ndarray,
    face_dilate_kernel: int,
    face_dilate_iters: int,
    min_pixels_after_exclusion: int,
) -> np.ndarray:
    kernel = np.ones((face_dilate_kernel, face_dilate_kernel), np.uint8)
    protected_face = cv2.dilate(face_mask.astype(np.uint8), kernel, iterations=face_dilate_iters)
    final_mask = auditor_mask.astype(np.uint8) * (1 - protected_face.astype(np.uint8))

    if int(final_mask.sum()) < int(min_pixels_after_exclusion):
        return auditor_mask.astype(np.uint8)
    return final_mask.astype(np.uint8)


def process_mask_example(
    example: Dict,
    auditor_runner: AuditorRunner,
    face_parser: Optional[FaceParser],
    nudity_field: str,
    face_exclusion_enabled: bool,
    heatmap_percentile: float,
    mask_dilate_kernel: int,
    mask_dilate_iters: int,
    face_dilate_kernel: int,
    face_dilate_iters: int,
    min_pixels_after_exclusion: int,
    feather_sigma: float,
    removal_threshold: int,
    heatmap_resize_interpolation: str,
) -> Dict:
    pil_img = ensure_pil_image(example["image"], mode="RGB")
    prompt = str(example.get("prompt", ""))

    heatmap = auditor_runner.infer_heatmap(pil_img, prompt)
    heatmap = resize_heatmap_to_image(
        heatmap=heatmap,
        target_hw=(pil_img.height, pil_img.width),
        interpolation=heatmap_resize_interpolation,
    )
    auditor_mask = build_mask_from_heatmap(
        heatmap=heatmap,
        percentile=heatmap_percentile,
        dilate_kernel=mask_dilate_kernel,
        dilate_iters=mask_dilate_iters,
    )

    final_mask = auditor_mask
    nudity_flag = int(example.get(nudity_field, 0)) == 1

    if face_exclusion_enabled and nudity_flag and face_parser is not None:
        face_mask = face_parser.face_mask(pil_img)
        final_mask = apply_face_exclusion(
            auditor_mask=auditor_mask,
            face_mask=face_mask,
            face_dilate_kernel=face_dilate_kernel,
            face_dilate_iters=face_dilate_iters,
            min_pixels_after_exclusion=min_pixels_after_exclusion,
        )

    feathered_mask = gaussian_filter(final_mask.astype(float), sigma=feather_sigma)
    feathered_mask = np.clip(feathered_mask, 0, 1)
    feathered_mask_u8 = (feathered_mask * 255).astype(np.uint8)

    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    removal_mask = feathered_mask_u8 > int(removal_threshold)
    img_removed = cv_img.copy()
    img_removed[removal_mask] = 0
    img_removed_rgb = cv2.cvtColor(img_removed, cv2.COLOR_BGR2RGB)

    output = {
        "image": pil_to_png_bytes(pil_img),                        # bytes, not PIL
        "prompt": prompt,
        "id": str(example.get("id", "")),
        "safe": int(example.get("safe", 0)),
        "nudity": int(example.get("nudity", 0)),
        "violence": int(example.get("violence", 0)),
        "feathered_mask": pil_to_png_bytes(                         # bytes, not PIL
            Image.fromarray(feathered_mask_u8, mode="L")
        ),
        "image_masked_removed": pil_to_png_bytes(                   # bytes, not PIL
            Image.fromarray(img_removed_rgb, mode="RGB")
        ),
        }
    return output


def run_mask_stage(args, api: HfApi) -> None:
    features = Features(
        {
            "image": HFImage(),
            "prompt": Value("string"),
            "id": Value("string"),
            "safe": Value("int8"),
            "nudity": Value("int8"),
            "violence": Value("int8"),
            "feathered_mask": HFImage(),
            "image_masked_removed": HFImage(),
        }
    )

    print("Loading auditor model...")
    auditor_runner = AuditorRunner(
        checkpoint=args.auditor_checkpoint,
        vocab=args.auditor_vocab,
        device=args.device,
        model_url=args.auditor_model_url,
        vocab_url=args.auditor_vocab_url,
    )

    face_parser = None
    if args.enable_face_exclusion:
        print(f"Loading face parser: {args.face_parsing_model}")
        face_parser = FaceParser(args.face_parsing_model, device=args.device)

    source_stream = load_dataset(args.source_dataset, split=args.source_split, streaming=True)
    processed_ids = collect_existing_ids(args.mask_dataset) if args.resume_masks else set()

    uploader = MaskShardUploader(
        api=api,
        repo_id=args.mask_dataset,
        shard_size=args.mask_shard_size,
        features=features,
    )

    new_count = 0
    pbar = tqdm(source_stream, desc="Mask stage", dynamic_ncols=True, unit="sample")
    for example in pbar:
        sample_id = str(example.get("id", ""))
        pbar.set_postfix_str(f"id={sample_id} processed={new_count}")
        if args.resume_masks and sample_id in processed_ids:
            continue

        try:
            row = process_mask_example(
                example=example,
                auditor_runner=auditor_runner,
                face_parser=face_parser,
                nudity_field=args.nudity_field,
                face_exclusion_enabled=args.enable_face_exclusion,
                heatmap_percentile=args.heatmap_percentile,
                mask_dilate_kernel=args.mask_dilate_kernel,
                mask_dilate_iters=args.mask_dilate_iters,
                face_dilate_kernel=args.face_dilate_kernel,
                face_dilate_iters=args.face_dilate_iters,
                min_pixels_after_exclusion=args.min_pixels_after_exclusion,
                feather_sigma=args.feather_sigma,
                removal_threshold=args.removal_threshold,
                heatmap_resize_interpolation=args.heatmap_resize_interpolation,
            )
            uploader.add(row)
            new_count += 1
            pbar.set_postfix_str(f"id={sample_id} processed={new_count} queued={len(uploader.rows)}")

            if args.max_mask_samples and new_count >= args.max_mask_samples:
                break
        except Exception as exc:
            print(f"Mask stage error for id={sample_id}: {exc}")

    print("[MASK] Final flush starting...")
    uploader.flush()
    print(f"Mask stage complete. New samples processed: {new_count}")


def build_latent_row(
    example: Dict,
    vae,
    tokenizer,
    transform,
    mask_transform,
    mask_latent_transform,
    random_dilate_enabled: bool,
    random_dilate_low: int,
    random_dilate_high: int,
    label_field: str,
) -> Dict:
    img = ensure_pil_image(example["image"], mode="RGB")
    mask = ensure_pil_image(example["feathered_mask"], mode="L")

    if random_dilate_enabled:
        mask = random_dilate(mask, low=random_dilate_low, high=random_dilate_high)

    mask_t = mask_transform(mask)
    mask_l = mask_latent_transform(mask)
    img_t = transform(img)
    img_ctx_t = img_t * (1 - mask_t)

    with torch.no_grad():
        z0 = vae.encode(img_t.unsqueeze(0).half().cuda()).latent_dist.sample() * 0.18215
        masked_latent = vae.encode(img_ctx_t.unsqueeze(0).half().cuda()).latent_dist.sample() * 0.18215

    tokens = tokenizer(
        str(example.get("prompt", "")),
        padding="max_length",
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).input_ids.squeeze(0)

    label_value = float(1 if int(example.get(label_field, 0)) == 1 else 0)

    return {
        "z0": z0.squeeze(0).float().cpu().numpy().tolist(),
        "masked_latent": masked_latent.squeeze(0).float().cpu().numpy().tolist(),
        "mask_latent": mask_l.squeeze(0).float().cpu().numpy().tolist(),
        "input_ids": tokens.cpu().numpy().tolist(),
        "label": label_value,
    }


def process_latent_stream(
    stream: Iterable[Dict],
    writer: LatentShardUploader,
    vae,
    tokenizer,
    transform,
    mask_transform,
    mask_latent_transform,
    random_dilate_enabled: bool,
    random_dilate_low: int,
    random_dilate_high: int,
    label_field: str,
    max_samples: int,
    desc: str,
) -> int:
    count = 0
    for example in tqdm(stream, desc=desc):
        try:
            row = build_latent_row(
                example=example,
                vae=vae,
                tokenizer=tokenizer,
                transform=transform,
                mask_transform=mask_transform,
                mask_latent_transform=mask_latent_transform,
                random_dilate_enabled=random_dilate_enabled,
                random_dilate_low=random_dilate_low,
                random_dilate_high=random_dilate_high,
                label_field=label_field,
            )
            writer.add(row)
            count += 1
            if max_samples and count >= max_samples:
                break
        except Exception as exc:
            sample_id = str(example.get("id", ""))
            print(f"Latent stage error for id={sample_id}: {exc}")

    writer.flush()
    return count


def get_primary_category(example: Dict, category_fields: List[str], fallback_field: str) -> str:
    for field in category_fields:
        try:
            if int(example.get(field, 0)) == 1:
                return field
        except Exception:
            continue
    return fallback_field


def run_latent_stage(args, api: HfApi) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Latent stage requires CUDA for VAE encoding.")

    print(f"Loading inpainting base model: {args.base_model}")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to("cuda")

    vae = pipe.vae.eval()
    tokenizer = pipe.tokenizer

    transform = T.Compose(
        [
            T.Resize((args.image_size, args.image_size)),
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
        ]
    )

    mask_transform = T.Compose(
        [
            T.Resize((args.image_size, args.image_size)),
            T.ToTensor(),
        ]
    )

    mask_latent_transform = T.Compose(
        [
            T.Resize((args.latent_mask_size, args.latent_mask_size)),
            T.ToTensor(),
        ]
    )

    category_fields = [x.strip() for x in args.val_category_fields.split(",") if x.strip()]
    if not category_fields:
        raise ValueError("--val-category-fields must contain at least one field name.")

    # Pass 1: count category totals so each category gets its own 2% validation quota.
    category_totals: Dict[str, int] = {field: 0 for field in category_fields}
    counting_stream = load_dataset(args.mask_dataset, split=args.mask_split_for_latents, streaming=True)
    total_samples = 0
    for example in tqdm(counting_stream, desc="Count categories"):
        cat = get_primary_category(example, category_fields, args.label_field)
        category_totals[cat] = category_totals.get(cat, 0) + 1
        total_samples += 1

    val_targets: Dict[str, int] = {}
    for cat, total in category_totals.items():
        target = int(math.floor(total * args.val_ratio))
        if args.ensure_min_val_per_category and total > 0 and target == 0:
            target = 1
        val_targets[cat] = target

    print(f"Total samples in split '{args.mask_split_for_latents}': {total_samples}")
    print("Validation targets by category:")
    for cat in sorted(val_targets.keys()):
        print(f"  {cat}: {val_targets[cat]} / {category_totals.get(cat, 0)}")

    # Pass 2: route samples to val/train using per-category quotas.
    routing_stream = load_dataset(args.mask_dataset, split=args.mask_split_for_latents, streaming=True)
    val_selected: Dict[str, int] = {cat: 0 for cat in val_targets.keys()}

    val_writer = LatentShardUploader(
        api=api,
        repo_id=args.latent_dataset,
        split="val",
        shard_size=args.latent_shard_size,
    )
    train_writer = LatentShardUploader(
        api=api,
        repo_id=args.latent_dataset,
        split="train",
        shard_size=args.latent_shard_size,
    )

    val_count = 0
    train_count = 0

    for example in tqdm(routing_stream, desc="Latent split+encode"):
        cat = get_primary_category(example, category_fields, args.label_field)
        send_to_val = val_selected.get(cat, 0) < val_targets.get(cat, 0)

        if send_to_val and args.max_val_samples and val_count >= args.max_val_samples:
            send_to_val = False

        if (not send_to_val) and args.max_train_samples and train_count >= args.max_train_samples:
            # train cap reached; skip extra train samples
            continue

        if args.max_val_samples and args.max_train_samples:
            if val_count >= args.max_val_samples and train_count >= args.max_train_samples:
                break

        try:
            row = build_latent_row(
                example=example,
                vae=vae,
                tokenizer=tokenizer,
                transform=transform,
                mask_transform=mask_transform,
                mask_latent_transform=mask_latent_transform,
                random_dilate_enabled=args.random_dilate_latents,
                random_dilate_low=args.random_dilate_low,
                random_dilate_high=args.random_dilate_high,
                label_field=args.label_field,
            )

            if send_to_val:
                val_writer.add(row)
                val_selected[cat] = val_selected.get(cat, 0) + 1
                val_count += 1
            else:
                train_writer.add(row)
                train_count += 1
        except Exception as exc:
            sample_id = str(example.get("id", ""))
            print(f"Latent stage error for id={sample_id}: {exc}")

    val_writer.flush()
    train_writer.flush()

    print(f"Latent stage complete. train={train_count}, val={val_count}")
    print("Validation samples selected by category:")
    for cat in sorted(val_selected.keys()):
        print(f"  {cat}: {val_selected[cat]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build mask and latent datasets for KTO inpainting.")

    parser.add_argument("--stage", choices=["masks", "latents", "all"], default="all")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--hf-token", default=os.getenv("HF_TOKEN", ""))

    # Stage 1: masks
    parser.add_argument("--source-dataset", default="ShreyashDhoot/internvl-auditor-v2")
    parser.add_argument("--source-split", default="train")
    parser.add_argument("--mask-dataset", default="ShreyashDhoot/KTO")

    parser.add_argument("--auditor-checkpoint", default="checkpoints/complete_auditor_best.pth")
    parser.add_argument("--auditor-vocab", default="checkpoints/vocab.json")
    parser.add_argument("--auditor-model-url", default=auditor_module.DEFAULT_MODEL_URL)
    parser.add_argument("--auditor-vocab-url", default=auditor_module.DEFAULT_VOCAB_URL)

    parser.add_argument("--resume-masks", action="store_true")
    parser.add_argument("--mask-shard-size", type=int, default=500)
    parser.add_argument("--max-mask-samples", type=int, default=0)

    parser.add_argument("--heatmap-percentile", type=float, default=75.0)
    parser.add_argument(
        "--heatmap-resize-interpolation",
        choices=["nearest", "bilinear", "bicubic"],
        default="bilinear",
        help="Interpolation used when resizing auditor heatmap to image size before thresholding.",
    )
    parser.add_argument("--mask-dilate-kernel", type=int, default=5)
    parser.add_argument("--mask-dilate-iters", type=int, default=2)
    parser.add_argument("--feather-sigma", type=float, default=5.0)
    parser.add_argument("--removal-threshold", type=int, default=50)

    parser.add_argument("--enable-face-exclusion", action="store_true")
    parser.add_argument("--face-parsing-model", default="jonathandinu/face-parsing")
    parser.add_argument("--nudity-field", default="nudity")
    parser.add_argument("--face-dilate-kernel", type=int, default=5)
    parser.add_argument("--face-dilate-iters", type=int, default=1)
    parser.add_argument("--min-pixels-after-exclusion", type=int, default=64)

    # Stage 2: latents
    parser.add_argument("--latent-dataset", default="ShreyashDhoot/KTO-latents")
    parser.add_argument("--mask-split-for-latents", default="train")
    parser.add_argument("--base-model", default="runwayml/stable-diffusion-inpainting")

    parser.add_argument("--val-ratio", type=float, default=0.02)
    parser.add_argument(
        "--val-category-fields",
        default="safe,nudity,violence",
        help="Comma-separated category fields used for stratified validation split.",
    )
    parser.add_argument(
        "--ensure-min-val-per-category",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Ensure at least 1 validation sample for non-empty categories.",
    )

    parser.add_argument("--val-count", type=int, default=500)
    parser.add_argument("--latent-shard-size", type=int, default=2000)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=10)

    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--latent-mask-size", type=int, default=64)
    parser.add_argument("--label-field", default="safe")

    parser.add_argument("--random-dilate-latents", action="store_true")
    parser.add_argument("--random-dilate-low", type=int, default=4)
    parser.add_argument("--random-dilate-high", type=int, default=15)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.hf_token:
        login(token=args.hf_token)

    api = HfApi(token=args.hf_token or None)

    if args.stage in ("masks", "all"):
        run_mask_stage(args, api)

    if args.stage in ("latents", "all"):
        run_latent_stage(args, api)


if __name__ == "__main__":
    main()
