import os
import yaml
import copy
import torch
from torch.utils.data import DataLoader
from diffusers import StableDiffusionInpaintPipeline, UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from transformers import CLIPTextModel
from peft import LoraConfig, get_peft_model

from data.dataset import LatentInpaintDataset
from data.collate import latent_collate
from utils.seed import seed_everything
from utils.logging import init_wandb, log_metrics, finish_wandb
from engine.checkpoint import save_checkpoint
from engine.evaluate import visual_eval
from engine.train_one_epoch import train_loop

def main():
    with open("configs/inpaint.yaml", "r") as f: #opens the yaml config file 
        cfg = yaml.safe_load(f) # loads cfg with configurations from yaml 

    seed_everything(42) # sets all the seeds to 42 for reproducability 
    device = "cuda" 

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

    train_loader = DataLoader(
        train_ds, 
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=True,
        collate_fn=latent_collate,
    )

    val_vis_samples = [val_ds[i] for i in range(min(4, len(val_ds)))]

    # setting up the stable diffusion pipeline 
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        cfg["model"]["base_model"], 
        torch_dtype=torch.float16,
        safety_checker=None,
    )

    #initialized the unet 
    unet = pipe.unet
    
    # ================================================
    # LoRA injection setup
    # ================================================
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
    
    # make a deep copy for reference net 
    ref_unet = copy.deepcopy(unet).eval()
    for p in ref_unet.parameters():
        '''
        turn of gradients for the reference model 
        '''
        p.requires_grad = False

    # set up common image encoder for both the models 
    vae = pipe.vae.eval()
    
    #setting up common text encoder 
    text_enc = pipe.text_encoder.eval()
    # setting up the schedular
    scheduler = DDPMScheduler.from_pretrained(cfg["model"]["base_model"], subfolder="scheduler")

    optimizer = torch.optim.AdamW(unet.parameters(), lr=cfg["training"]["lr"])
    total_steps = cfg["training"]["max_steps"]
    warmup_steps = cfg["training"]["warmup_steps"]

    def lr_lambda(step):
        if step < warmup_steps:
            return max(1e-8, float(step + 1) / float(max(1, warmup_steps)))
        return 1.0

    lr_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scaler = torch.amp.GradScaler("cuda")

    os.makedirs(cfg["output"]["checkpoint_dir"], exist_ok=True)
    os.makedirs(cfg["output"]["eval_dir"], exist_ok=True)

    def save_fn(global_step, unet, optimizer, scheduler, scaler, epoch):
        ckpt_path = os.path.join(cfg["output"]["checkpoint_dir"], f"step_{global_step}.pt")
        save_checkpoint(ckpt_path, unet, optimizer, scheduler, scaler, global_step, epoch)
        unet.save_pretrained(f"checkpoint--{global_step}")

    def wandb_log_fn(metrics, step):
        log_metrics(step, metrics)

    def visual_eval_fn(unet, pipe, val_vis_samples, step):
        visual_eval(unet, pipe, val_vis_samples, step, cfg["output"]["eval_dir"])

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

    finish_wandb()

if __name__ == "__main__":
    main()