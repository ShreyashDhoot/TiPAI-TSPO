#!/usr/bin/env python3
"""
Unified KTO data pipeline:
1) Build mask dataset with auditor heatmaps (+ optional face exclusion for nudity samples)
2) Convert mask dataset into latent parquet shards
3) Upload both outputs to Hugging Face datasets

This script intentionally reuses infer_train_new.py utilities for auditor loading/inference.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import io
import os
import tempfile
import math
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None

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


if load_dotenv is not None:
    load_dotenv()


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

        ds = Dataset.from_list(self.rows, features=self.features)
        buffer = io.BytesIO()
        ds.to_parquet(buffer)
        buffer.seek(0)

        path_in_repo = f"{self.path_prefix}/train-{self.shard_idx:05d}-of-NNNNN.parquet"
        self.api.upload_file(
            path_or_fileobj=buffer,
            path_in_repo=path_in_repo,
            repo_id=self.repo_id,
            repo_type="dataset",
            commit_message=f"Upload shard {self.shard_idx}",
        )
        print(f"[MASK] Uploaded shard {self.shard_idx} -> {path_in_repo}")

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

    def infer_heatmap_batch(self, images: List[Image.Image], prompts: List[str]) -> List[np.ndarray]:
        if len(images) != len(prompts):
            raise ValueError("images and prompts must have the same length.")
        if not images:
            return []
        if len(images) == 1:
            return [self.infer_heatmap(images[0], prompts[0])]

        img_tensors = [self.auditor.transform(img.convert("RGB")) for img in images]
        token_tensors = [self.auditor.tokenizer.encode(prompt) for prompt in prompts]

        img_batch = torch.stack(img_tensors, dim=0).to(auditor_module.DEVICE)
        token_batch = torch.stack(token_tensors, dim=0).to(auditor_module.DEVICE)
        timestep_batch = torch.zeros((img_batch.shape[0], 1), device=auditor_module.DEVICE)

        with torch.inference_mode():
            outputs = self.auditor.model(img_batch, text_tokens=token_batch, timestep=timestep_batch)

        maps = outputs.get("adversarial_map")
        if maps is None:
            raise RuntimeError("Auditor inference did not return 'adversarial_map'.")

        maps = maps[:, 0].detach().float().cpu().numpy()
        return [maps[i] for i in range(maps.shape[0])]


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
        self.pipe = hf_pipeline("image-segmentation", model=model_id, device=pipe_device, trust_remote_code=True)

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
    preloaded_image: Optional[Image.Image] = None,
    precomputed_heatmap: Optional[np.ndarray] = None,
) -> Dict:
    pil_img = preloaded_image if preloaded_image is not None else ensure_pil_image(example["image"], mode="RGB")
    prompt = str(example.get("prompt", ""))

    heatmap = precomputed_heatmap if precomputed_heatmap is not None else auditor_runner.infer_heatmap(pil_img, prompt)
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
    face_exclusion_enabled = args.enable_face_exclusion
    if face_exclusion_enabled:
        print(f"Loading face parser: {args.face_parsing_model}")
        try:
            face_parser = FaceParser(args.face_parsing_model, device=args.device)
        except Exception as exc:
            print(
                "Warning: failed to initialize face parser; continuing with face exclusion disabled. "
                f"Model='{args.face_parsing_model}' error='{exc}'"
            )
            face_exclusion_enabled = False

    source_stream = load_dataset(args.source_dataset, split=args.source_split, streaming=True)
    processed_ids = collect_existing_ids(args.mask_dataset) if args.resume_masks else set()

    if args.mask_batch_size < 1:
        raise ValueError("--mask-batch-size must be >= 1.")

    if args.mask_cpu_workers < 0:
        raise ValueError("--mask-cpu-workers must be >= 0.")

    if args.mask_progress_log_every < 1:
        raise ValueError("--mask-progress-log-every must be >= 1.")

    batch_size = args.mask_batch_size
    if args.device == "cpu":
        batch_size = 1
    elif args.device == "auto" and not torch.cuda.is_available():
        batch_size = 1

    auto_workers = max(1, min(16, (os.cpu_count() or 1)))
    cpu_workers = args.mask_cpu_workers if args.mask_cpu_workers > 0 else auto_workers

    if torch.cuda.is_available() and args.device in ("auto", "cuda"):
        torch.backends.cudnn.benchmark = True

    print(f"[MASK] Runtime config: batch_size={batch_size}, cpu_workers={cpu_workers}")

    uploader = MaskShardUploader(
        api=api,
        repo_id=args.mask_dataset,
        shard_size=args.mask_shard_size,
        features=features,
    )

    def process_batch(filtered_examples: List[Dict], executor: Optional[ThreadPoolExecutor]) -> int:
        if not filtered_examples:
            return 0

        prepared: List[Tuple[Dict, Image.Image, str]] = []
        for sample in filtered_examples:
            pil_img = ensure_pil_image(sample["image"], mode="RGB")
            prompt = str(sample.get("prompt", ""))
            prepared.append((sample, pil_img, prompt))

        heatmaps = auditor_runner.infer_heatmap_batch(
            images=[x[1] for x in prepared],
            prompts=[x[2] for x in prepared],
        )

        def build_row(item: Tuple[Dict, Image.Image, str], heatmap: np.ndarray) -> Dict:
            sample, pil_img, _ = item
            return process_mask_example(
                example=sample,
                auditor_runner=auditor_runner,
                face_parser=face_parser,
                nudity_field=args.nudity_field,
                face_exclusion_enabled=face_exclusion_enabled,
                heatmap_percentile=args.heatmap_percentile,
                mask_dilate_kernel=args.mask_dilate_kernel,
                mask_dilate_iters=args.mask_dilate_iters,
                face_dilate_kernel=args.face_dilate_kernel,
                face_dilate_iters=args.face_dilate_iters,
                min_pixels_after_exclusion=args.min_pixels_after_exclusion,
                feather_sigma=args.feather_sigma,
                removal_threshold=args.removal_threshold,
                heatmap_resize_interpolation=args.heatmap_resize_interpolation,
                preloaded_image=pil_img,
                precomputed_heatmap=heatmap,
            )

        created = 0
        if executor is None:
            for item, heatmap in zip(prepared, heatmaps):
                row = build_row(item, heatmap)
                uploader.add(row)
                created += 1
            return created

        futures = [executor.submit(build_row, item, heatmap) for item, heatmap in zip(prepared, heatmaps)]
        for fut in futures:
            row = fut.result()
            uploader.add(row)
            created += 1
        return created

    new_count = 0
    last_logged = 0
    pbar = tqdm(
        total=(args.max_mask_samples if args.max_mask_samples > 0 else None),
        desc="Mask stage",
        dynamic_ncols=True,
        unit="sample",
        mininterval=0.5,
    )
    pending: List[Dict] = []
    source_iter = iter(source_stream)

    executor: Optional[ThreadPoolExecutor] = None
    try:
        if cpu_workers > 1:
            executor = ThreadPoolExecutor(max_workers=cpu_workers)

        for example in source_iter:
            sample_id = str(example.get("id", ""))
            if args.resume_masks and sample_id in processed_ids:
                continue

            pending.append(example)
            if len(pending) < batch_size:
                continue

            if args.max_mask_samples:
                remaining = args.max_mask_samples - new_count
                if remaining <= 0:
                    break
                pending = pending[:remaining]

            try:
                created = process_batch(pending, executor)
                new_count += created
                pbar.update(created)
                pbar.set_postfix_str(f"processed={new_count} queued={len(uploader.rows)}")
                if (new_count - last_logged) >= args.mask_progress_log_every:
                    tqdm.write(f"[MASK] processed={new_count} queued={len(uploader.rows)}")
                    last_logged = new_count
            except Exception as exc:
                print(f"Mask stage batch error near id={sample_id}: {exc}")
            finally:
                pending = []

            if args.max_mask_samples and new_count >= args.max_mask_samples:
                break

        if pending and (not args.max_mask_samples or new_count < args.max_mask_samples):
            if args.max_mask_samples:
                remaining = args.max_mask_samples - new_count
                pending = pending[:remaining]
            try:
                created = process_batch(pending, executor)
                new_count += created
                pbar.update(created)
                pbar.set_postfix_str(f"processed={new_count} queued={len(uploader.rows)}")
                if (new_count - last_logged) >= args.mask_progress_log_every:
                    tqdm.write(f"[MASK] processed={new_count} queued={len(uploader.rows)}")
                    last_logged = new_count
            except Exception as exc:
                sample_id = str(pending[-1].get("id", "")) if pending else ""
                print(f"Mask stage final batch error near id={sample_id}: {exc}")
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    pbar.close()

    if new_count != last_logged:
        print(f"[MASK] processed={new_count} queued={len(uploader.rows)}")

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
    label_field: str,
) -> Dict:
    img = ensure_pil_image(example["image"], mode="RGB")
    mask = ensure_pil_image(example["feathered_mask"], mode="L")

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


def build_latent_rows_batch(
    batch_items: List[Tuple[Dict, bool, str]],
    vae,
    tokenizer,
    transform,
    mask_transform,
    mask_latent_transform,
    label_field: str,
    device: torch.device,
    executor: Optional[ThreadPoolExecutor] = None,
) -> List[Tuple[Dict, bool, str]]:
    def prepare_item(example: Dict, send_to_val: bool, cat: str) -> Optional[Dict]:
        sample_id = str(example.get("id", ""))
        try:
            img = ensure_pil_image(example["image"], mode="RGB")
            mask = ensure_pil_image(example["feathered_mask"], mode="L")

            mask_t = mask_transform(mask)
            mask_l = mask_latent_transform(mask)
            img_t = transform(img)
            img_ctx_t = img_t * (1 - mask_t)
            prompt = str(example.get("prompt", ""))
            label_value = float(1 if int(example.get(label_field, 0)) == 1 else 0)
        except Exception as exc:
            print(f"Latent prep error for id={sample_id}: {exc}")
            return None

        return {
            "sample_id": sample_id,
            "cat": cat,
            "send_to_val": send_to_val,
            "img_t": img_t,
            "img_ctx_t": img_ctx_t,
            "mask_l": mask_l,
            "prompt": prompt,
            "label": label_value,
        }

    prepared: List[Dict] = []
    if executor is None:
        for example, send_to_val, cat in batch_items:
            item = prepare_item(example, send_to_val, cat)
            if item is not None:
                prepared.append(item)
    else:
        futures = [executor.submit(prepare_item, example, send_to_val, cat) for example, send_to_val, cat in batch_items]
        for fut in futures:
            item = fut.result()
            if item is not None:
                prepared.append(item)

    if not prepared:
        return []

    img_batch = torch.stack([x["img_t"] for x in prepared], dim=0).to(device=device, dtype=torch.float16)
    img_ctx_batch = torch.stack([x["img_ctx_t"] for x in prepared], dim=0).to(device=device, dtype=torch.float16)

    with torch.inference_mode():
        encode_input = torch.cat([img_batch, img_ctx_batch], dim=0)
        encoded = vae.encode(encode_input).latent_dist.sample() * 0.18215
    z0_batch, masked_batch = encoded.chunk(2, dim=0)

    token_ids = tokenizer(
        [x["prompt"] for x in prepared],
        padding="max_length",
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).input_ids

    rows: List[Tuple[Dict, bool, str]] = []
    for idx, item in enumerate(prepared):
        row = {
            "z0": z0_batch[idx].float().cpu().numpy().tolist(),
            "masked_latent": masked_batch[idx].float().cpu().numpy().tolist(),
            "mask_latent": item["mask_l"].squeeze(0).float().cpu().numpy().tolist(),
            "input_ids": token_ids[idx].cpu().numpy().tolist(),
            "label": item["label"],
        }
        rows.append((row, item["send_to_val"], item["cat"]))

    return rows


def process_latent_stream(
    stream: Iterable[Dict],
    writer: LatentShardUploader,
    vae,
    tokenizer,
    transform,
    mask_transform,
    mask_latent_transform,
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

    if args.latent_batch_size < 1:
        raise ValueError("--latent-batch-size must be >= 1.")

    if args.latent_progress_log_every < 1:
        raise ValueError("--latent-progress-log-every must be >= 1.")

    if args.latent_cpu_workers < 0:
        raise ValueError("--latent-cpu-workers must be >= 0.")

    device = torch.device("cuda")
    auto_workers = max(1, min(16, (os.cpu_count() or 1)))
    cpu_workers = args.latent_cpu_workers if args.latent_cpu_workers > 0 else auto_workers

    print(f"Loading inpainting base model: {args.base_model}")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to("cuda")

    vae = pipe.vae.eval()
    tokenizer = pipe.tokenizer

    print(f"[LATENT] Runtime config: batch_size={args.latent_batch_size}, cpu_workers={cpu_workers}")

    torch.backends.cudnn.benchmark = True

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

    val_targets: Dict[str, int] = {}
    category_totals: Dict[str, int] = {field: 0 for field in category_fields}

    if args.val_target_mode == "ratio":
        # Ratio mode needs a first pass to estimate per-category targets.
        counting_stream = load_dataset(args.mask_dataset, split=args.mask_split_for_latents, streaming=True)
        total_samples = 0
        for example in tqdm(counting_stream, desc="Count categories"):
            cat = get_primary_category(example, category_fields, args.label_field)
            category_totals[cat] = category_totals.get(cat, 0) + 1
            total_samples += 1

        for cat, total in category_totals.items():
            target = int(math.floor(total * args.val_ratio))
            if args.ensure_min_val_per_category and total > 0 and target == 0:
                target = 1
            val_targets[cat] = target

        print(f"Total samples in split '{args.mask_split_for_latents}': {total_samples}")
        print("Validation targets by category:")
        for cat in sorted(val_targets.keys()):
            print(f"  {cat}: {val_targets[cat]} / {category_totals.get(cat, 0)}")
    else:
        # Fixed mode avoids the counting pass entirely.
        fixed_target = max(0, int(args.val_count))
        val_targets = {cat: fixed_target for cat in category_fields}
        print("Validation target mode: fixed (counting pass skipped)")
        print("Validation targets by category:")
        for cat in sorted(val_targets.keys()):
            print(f"  {cat}: {val_targets[cat]}")

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
    last_logged = 0

    pbar_total = None
    if args.max_train_samples and args.max_val_samples:
        pbar_total = args.max_train_samples + args.max_val_samples
    pbar = tqdm(total=pbar_total, desc="Latent split+encode", dynamic_ncols=True, unit="sample")

    pending_items: List[Tuple[Dict, bool, str]] = []
    pending_val_by_cat: Dict[str, int] = {}
    executor: Optional[ThreadPoolExecutor] = None

    def flush_latent_batch() -> Tuple[int, int, Dict[str, int]]:
        if not pending_items:
            return 0, 0, {}

        rows = build_latent_rows_batch(
            batch_items=pending_items,
            vae=vae,
            tokenizer=tokenizer,
            transform=transform,
            mask_transform=mask_transform,
            mask_latent_transform=mask_latent_transform,
            label_field=args.label_field,
            device=device,
            executor=executor,
        )

        batch_train = 0
        batch_val = 0
        val_increments: Dict[str, int] = {}

        for row, send_to_val, cat in rows:
            if send_to_val:
                val_writer.add(row)
                batch_val += 1
                val_increments[cat] = val_increments.get(cat, 0) + 1
            else:
                train_writer.add(row)
                batch_train += 1

        return batch_train, batch_val, val_increments

    try:
        if cpu_workers > 1:
            executor = ThreadPoolExecutor(max_workers=cpu_workers)

        for example in routing_stream:
            cat = get_primary_category(example, category_fields, args.label_field)
            planned_val = val_selected.get(cat, 0) + pending_val_by_cat.get(cat, 0)
            send_to_val = planned_val < val_targets.get(cat, 0)

            if send_to_val and args.max_val_samples and (val_count + sum(pending_val_by_cat.values())) >= args.max_val_samples:
                send_to_val = False

            projected_train = train_count + (len(pending_items) - sum(pending_val_by_cat.values()))
            if (not send_to_val) and args.max_train_samples and projected_train >= args.max_train_samples:
                continue

            if args.max_val_samples and args.max_train_samples:
                if val_count >= args.max_val_samples and train_count >= args.max_train_samples:
                    break

            pending_items.append((example, send_to_val, cat))
            if send_to_val:
                pending_val_by_cat[cat] = pending_val_by_cat.get(cat, 0) + 1

            if len(pending_items) < args.latent_batch_size:
                continue

            try:
                batch_train, batch_val, val_increments = flush_latent_batch()
                train_count += batch_train
                val_count += batch_val
                for c, inc in val_increments.items():
                    val_selected[c] = val_selected.get(c, 0) + inc

                processed_now = batch_train + batch_val
                pbar.update(processed_now)
                pbar.set_postfix_str(f"train={train_count} val={val_count} queued={len(train_writer.rows)+len(val_writer.rows)}")
                if (train_count + val_count - last_logged) >= args.latent_progress_log_every:
                    tqdm.write(f"[LATENT] train={train_count} val={val_count}")
                    last_logged = train_count + val_count
            except Exception as exc:
                sample_id = str(pending_items[-1][0].get("id", "")) if pending_items else ""
                print(f"Latent stage batch error near id={sample_id}: {exc}")
            finally:
                pending_items = []
                pending_val_by_cat = {}

        if pending_items:
            try:
                batch_train, batch_val, val_increments = flush_latent_batch()
                train_count += batch_train
                val_count += batch_val
                for c, inc in val_increments.items():
                    val_selected[c] = val_selected.get(c, 0) + inc

                processed_now = batch_train + batch_val
                pbar.update(processed_now)
                pbar.set_postfix_str(f"train={train_count} val={val_count} queued={len(train_writer.rows)+len(val_writer.rows)}")
            except Exception as exc:
                sample_id = str(pending_items[-1][0].get("id", "")) if pending_items else ""
                print(f"Latent stage final batch error near id={sample_id}: {exc}")
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    pbar.close()

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
    parser.add_argument("--mask-shard-size", type=int, default=5000)
    parser.add_argument("--max-mask-samples", type=int, default=0)
    parser.add_argument(
        "--mask-progress-log-every",
        type=int,
        default=100,
        help="Emit a plain-text mask progress line every N processed samples.",
    )
    parser.add_argument(
        "--mask-batch-size",
        type=int,
        default=64,
        help="Batch size for auditor GPU inference during mask creation. Forced to 1 on CPU-only runs.",
    )
    parser.add_argument(
        "--mask-cpu-workers",
        type=int,
        default=64,
        help="CPU worker threads for per-sample mask postprocessing. 0 selects an automatic value.",
    )

    parser.add_argument("--heatmap-percentile", type=float, default=80.0)
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

    parser.add_argument(
        "--val-target-mode",
        choices=["fixed", "ratio"],
        default="fixed",
        help="How to choose per-category validation targets. 'fixed' skips counting and uses --val-count per category.",
    )
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

    parser.add_argument(
        "--val-count",
        type=int,
        default=10,
        help="Validation samples per category when --val-target-mode fixed.",
    )
    parser.add_argument("--latent-shard-size", type=int, default=20000)
    parser.add_argument(
        "--latent-batch-size",
        type=int,
        default=32,
        help="Batch size for latent-stage VAE encoding.",
    )
    parser.add_argument(
        "--latent-cpu-workers",
        type=int,
        default=16,
        help="CPU worker threads for latent-stage sample preparation. 0 selects an automatic value.",
    )
    parser.add_argument(
        "--latent-progress-log-every",
        type=int,
        default=1000,
        help="Emit a plain-text latent progress line every N processed samples.",
    )
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)

    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--latent-mask-size", type=int, default=64)
    parser.add_argument("--label-field", default="safe")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.hf_token:
        login(token=args.hf_token)

    api = HfApi(token=args.hf_token or None)

    for repo_id in (args.mask_dataset, args.latent_dataset):
        try:
            api.repo_info(repo_id, repo_type="dataset")
        except Exception as exc:
            raise RuntimeError(
                f"Hugging Face dataset repo '{repo_id}' is not accessible with the current token. "
                f"Make sure it exists, is spelled correctly, and that HF_TOKEN has write access."
            ) from exc

    if args.stage in ("masks", "all"):
        run_mask_stage(args, api)

    if args.stage in ("latents", "all"):
        run_latent_stage(args, api)


if __name__ == "__main__":
    main()
