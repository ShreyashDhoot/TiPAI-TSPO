import os
import sys
from dotenv import load_dotenv
load_dotenv()
# === PATH FIX: must come before ANY project imports ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# === NOW safe to import project modules ===
import torch
import glob
import yaml
from diffusers import StableDiffusionInpaintPipeline
from peft import PeftModel, PeftConfig
from data.dataset import LatentInpaintDataset
from engine.evaluate import visual_eval
from peft import PeftModel


# === CONFIG ===
def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    config_path = os.path.join(PROJECT_ROOT, "configs", "inpaint.yaml")
    config = load_config(config_path)

    base_model  = config["model"]["base_model"]
    eval_dir    = config["output"]["eval_dir"]
    cache_dir   = config["data"]["cache_dir"]
    val_subdir  = config["data"]["val_subdir"]
    repo_id     = config["model"]["hf_dataset_repo"]
    eval_cfg    = config.get("output", {})

    # === LOAD VALIDATION DATASET ===
    val_ds = LatentInpaintDataset(repo_id=repo_id, split=val_subdir, cache_dir=cache_dir)
    val_vis_samples = [val_ds[i] for i in range(len(val_ds))]
    print(f"Loaded {len(val_vis_samples)} validation samples.")

    # === FIND ALL CHECKPOINTS (absolute glob, works regardless of CWD) ===
    ckpt_pattern = os.path.join(PROJECT_ROOT, "checkpoint--*")
    checkpoint_dirs = sorted([
        d for d in glob.glob(ckpt_pattern) if os.path.isdir(d)
    ])

    if not checkpoint_dirs:
        print(f"No checkpoint-- directories found under {PROJECT_ROOT}. Exiting.")
        return

    print(f"Found {len(checkpoint_dirs)} checkpoint(s): {[os.path.basename(d) for d in checkpoint_dirs]}")

    # === EVALUATE EACH CHECKPOINT ===
    for ckpt_dir in checkpoint_dirs:
        adapter_path = os.path.join(ckpt_dir, "adapter_model.safetensors")
        if not os.path.exists(adapter_path):
            print(f"Skipping {ckpt_dir}: adapter_model.safetensors not found.")
            continue

        print(f"\nEvaluating checkpoint: {os.path.basename(ckpt_dir)}")

        # Load pipeline fresh for each checkpoint (avoids LoRA weight bleed)
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            safety_checker=None,
        ).to("cuda")

        # Load LoRA weights
        try:
            pipe.unet = PeftModel.from_pretrained(pipe.unet, ckpt_dir)
            pipe.unet = pipe.unet.merge_and_unload()   # optional: bake weights in for faster inference
            pipe.unet = pipe.unet.to("cuda", dtype=torch.float16)
        except Exception as e:
            print(f"  Failed to load LoRA weights from {ckpt_dir}: {e}")
            del pipe
            torch.cuda.empty_cache()
            continue

        # Output dir for this checkpoint
        ckpt_name = os.path.basename(ckpt_dir)
        out_dir = os.path.join(eval_dir, f"eval_{ckpt_name}")
        os.makedirs(out_dir, exist_ok=True)

        # Run eval
        visual_eval(
            pipe.unet,
            pipe,
            val_vis_samples,
            step=ckpt_name,
            out_dir=out_dir,
            eval_cfg=eval_cfg,
        )
        print(f"  Saved outputs to {out_dir}")

        # Free VRAM between checkpoints
        pipe.unload_lora_weights()
        del pipe
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()