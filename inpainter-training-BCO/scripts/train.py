import os
import yaml
import copy
import torch
import math
from torch.utils.data import DataLoader
from dotenv import load_dotenv
from diffusers import StableDiffusionInpaintPipeline, UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from transformers import CLIPTextModel
from peft import LoraConfig, get_peft_model
from diffusers.training_utils import EMAModel

# Load environment variables from .env file
load_dotenv()

from data.dataset import LatentInpaintDataset
from data.collate import latent_collate
from utils.seed import seed_everything
from utils.logging import init_wandb, log_metrics, finish_wandb
from engine.checkpoint import save_checkpoint
from engine.evaluate import visual_eval
from engine.train_one_epoch import train_loop
from data.sampler import StratifiedBatchSampler 

try:
    from utils.plotting import plot_training_metrics
except Exception:
    plot_training_metrics = None

def main():
    with open("configs/inpaint.yaml", "r") as f: #opens the yaml config file 
        cfg = yaml.safe_load(f) # loads cfg with configurations from yaml 

    seed_everything(42) # sets all the seeds to 42 for reproducability 
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training script, but no GPU is available.")

    device = torch.device("cuda:3")
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    gpu_name = torch.cuda.get_device_name(device)
    print(f"Using GPU: {gpu_name}")

    # Launch a tiny compute warmup to verify kernels execute on GPU immediately.
    with torch.no_grad():
        warmup_a = torch.randn((1024, 1024), device=device, dtype=torch.float16)
        warmup_b = torch.randn((1024, 1024), device=device, dtype=torch.float16)
        for _ in range(8):
            warmup_a = warmup_a @ warmup_b
        torch.cuda.synchronize(device)
        del warmup_a, warmup_b

    run = init_wandb(cfg["output"]["wandb_project"], cfg) #initializes Wt & Bias for logging the training 

    train_ds = LatentInpaintDataset(
        repo_id=cfg["model"]["hf_dataset_repo"], #passes hf repo name from yml config file 
        split=cfg["data"]["train_subdir"], # passes the split to be used from hf repo 
        cache_dir=cfg["data"]["cache_dir"], #passes file loc to dump cache 
    )
    val_ds = LatentInpaintDataset(
        repo_id=cfg["model"]["hf_dataset_repo"],
        split=cfg["data"]["val_subdir"],
        cache_dir=cfg["data"]["cache_dir"],
    )

    _labels = train_ds.get_all_labels()

    stratified_sampler = StratifiedBatchSampler(
        labels=_labels,
        batch_size=cfg["training"]["batch_size"], # Should be 16
        safe_count=8,
        nudity_count=4,
        violence_count=4,
        shuffle=True,
    )

    train_loader = DataLoader(
        train_ds, 
        batch_sampler=stratified_sampler,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=True,
        persistent_workers=cfg["training"]["num_workers"] > 0,
        prefetch_factor=4 if cfg["training"]["num_workers"] > 0 else None,
        collate_fn=latent_collate,
    )

    val_vis_samples = [val_ds[i] for i in range(min(30, len(val_ds)))]

    # ── Pipeline loading ──────────────────────────────────────────────────
    # Supports two modes controlled by config:
    #
    #   merged_base: false  (default)
    #     Loads the original HuggingFace model. safety_checker=None is passed
    #     because the base runwayml model ships with one we don't need.
    #
    #   merged_base: true
    #     Loads a local merged pipeline produced by merge_lora_into_base.py.
    #     These pipelines were saved without a safety_checker, so we must NOT
    #     pass safety_checker=None (it would error on the missing component).
    #     The merged UNet already has the previous round's alignment baked in
    #     as full float16 weights — no LoRA layers present yet.
    #
    is_merged_base = cfg["model"].get("merged_base", False)

    if is_merged_base:
        print(f"[train] Loading MERGED base from: {cfg['model']['base_model']}")
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            cfg["model"]["base_model"],
            torch_dtype=torch.float16,
            safety_checker=None,
        )
    else:
        print(f"[train] Loading BASE model from: {cfg['model']['base_model']}")
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            cfg["model"]["base_model"],
            torch_dtype=torch.float16,
            safety_checker=None,
        )

    # ── LoRA injection ────────────────────────────────────────────────────
    # Whether we loaded from the original base or a merged checkpoint, we
    # always inject a FRESH set of LoRA adapters here. The merged weights
    # become the new "base" that the adapters train on top of.
    # This is correct: get_peft_model adds zero-initialised A/B matrices,
    # so the model's output is identical to the merged base at step 0.
    unet = pipe.unet
    if cfg["model"].get("use_lora", False):
        lora_config = LoraConfig(
            r=cfg["model"]["lora"]["r"],
            lora_alpha=cfg["model"]["lora"]["lora_alpha"],
            target_modules=cfg["model"]["lora"]["target_modules"],
            lora_dropout=cfg["model"]["lora"]["lora_dropout"],
            bias=cfg["model"]["lora"]["bias"],
        )
        unet = get_peft_model(unet, lora_config)
        unet.print_trainable_parameters()

    pipe.unet = unet
    pipe = pipe.to(device)
    unet = pipe.unet
    
    '''
    replacing this with ema-lora i.e the reference model remains static and as the learner 
    moves away from the reference it starts deviating and making blue patches so trying to 
    update the reference so the learner moves in a better direction for optimization
    '''
    # make a deep copy for reference net 
    ref_unet = copy.deepcopy(unet).eval()
    for p in ref_unet.parameters():
        #turn of gradients for the reference model 
        p.requires_grad = False
    ref_unet.to(device)

    # set up common image encoder for both the models 
    vae = pipe.vae.eval()
    
    #setting up common text encoder 
    text_enc = pipe.text_encoder.eval()
    # setting up the schedular
    scheduler = DDPMScheduler.from_pretrained(cfg["model"]["base_model"], subfolder="scheduler")

    optimizer = torch.optim.AdamW(unet.parameters(), lr=cfg["training"]["lr"],eps=1e-6)
    total_steps = cfg["training"]["max_steps"]
    warmup_steps = cfg["training"]["warmup_steps"]

    def lr_lambda(step):
        warmup_steps = cfg["training"]["warmup_steps"] # e.g., 800
        # Add a constant phase of ~8000 steps (approx 3 epochs of your Unsafe set)
        constant_steps = 8000 
        total_steps = cfg["training"]["max_steps"]
    
        # 1. Linear Warmup Phase
        if step < warmup_steps:
            return max(1e-8, float(step + 1) / float(max(1, warmup_steps)))
    
        # 2. Constant Phase (The "Breakthrough" Zone)
        # This keeps the LR at 100% of your peak (e.g., 4e-06)
        if step < warmup_steps + constant_steps:
            return 1.0
    
        # 3. Cosine Decay Phase
        # Decay only starts after step 8800 in this example
        decay_steps = total_steps - warmup_steps - constant_steps
        if decay_steps <= 0:
            return 1.0
            
        progress = float(step - warmup_steps - constant_steps) / float(decay_steps)
        # Standard cosine decay down to 10% of peak
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    lr_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scaler = torch.amp.GradScaler("cuda")

    os.makedirs(cfg["output"]["checkpoint_dir"], exist_ok=True)
    os.makedirs(cfg["output"]["eval_dir"], exist_ok=True)
    metric_history = []

    def save_fn(global_step, unet, optimizer, scheduler, scaler, epoch):
        ckpt_path = os.path.join(cfg["output"]["checkpoint_dir"], f"step_{global_step}.pt")
        save_checkpoint(ckpt_path, unet, optimizer, scheduler, scaler, global_step, epoch)
        unet.save_pretrained(f"checkpoint--{global_step}")

    def wandb_log_fn(metrics, step):
        metric_row = {"step": int(step)}
        metric_row.update(metrics)
        metric_history.append(metric_row)
        log_metrics(step, metrics)

    def visual_eval_fn(unet, pipe, val_vis_samples, step):
        visual_eval(
            unet,
            pipe,
            val_vis_samples,
            step,
            cfg["output"]["eval_dir"],
            eval_cfg=cfg.get("output", {}),
        )

    train_loop(
        unet=unet,
        ref_unet=ref_unet,
        vae=vae,
        text_enc=text_enc,
        scheduler=scheduler,
        optimizer=optimizer,
        lr_sched=lr_sched,
        scaler=scaler,
        train_loader=train_loader,
        pipe=pipe,
        val_vis_samples=val_vis_samples,
        wandb_log_fn=wandb_log_fn,
        save_fn=save_fn,
        visual_eval_fn=visual_eval_fn,
        cfg=cfg,
        device=device,
    )

    output_cfg = cfg.get("output", {})
    if output_cfg.get("plot_metrics", True):
        if plot_training_metrics is None:
            print("Metric plotting is enabled but matplotlib/seaborn are unavailable. Skipping plots.")
        else:
            metrics_plot_dir = output_cfg.get(
                "metrics_plot_dir",
                os.path.join(output_cfg.get("eval_dir", "./eval_outputs"), "metrics_plots"),
            )
            smoothing_window = int(output_cfg.get("plot_smoothing_window", 5))
            try:
                generated_files = plot_training_metrics(
                    metric_history,
                    metrics_plot_dir,
                    smoothing_window=smoothing_window,
                )
                if generated_files:
                    print(f"Saved {len(generated_files)} metric artifacts to {metrics_plot_dir}")
                else:
                    print("No metric history was available, so no plots were generated.")
            except Exception as err:
                print(f"Metric plotting failed: {err}")

    finish_wandb()

if __name__ == "__main__":
    main()