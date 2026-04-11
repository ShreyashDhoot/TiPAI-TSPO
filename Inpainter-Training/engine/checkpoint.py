import os
import torch

def save_checkpoint(path, unet, optimizer, scheduler, scaler, global_step, epoch):
    """Save training checkpoint.
    
    Saves the UNet model, optimizer, scheduler, and scaler states along
    with training metadata to enable resuming training.
    For LoRA-adapted models, saves only the LoRA weights.
    
    Args:
        path (str): Path to save checkpoint file.
        unet (UNet2DConditionModel or PeftModel): The UNet model (may be LoRA-adapted).
        optimizer (torch.optim.Optimizer): The optimizer.
        scheduler (torch.optim.lr_scheduler.LambdaLR): The learning rate scheduler.
        scaler (torch.amp.GradScaler): The gradient scaler for mixed precision.
        global_step (int): Current training step.
        epoch (int): Current epoch number.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Check if model is PEFT LoRA wrapped
    is_lora = hasattr(unet, 'save_pretrained')
    if is_lora:
        # Save LoRA adapters separately
        lora_dir = os.path.join(os.path.dirname(path), f"lora_step_{global_step}")
        unet.save_pretrained(lora_dir)
        unet_state = None  # Don't save full model state for LoRA
    else:
        # Save full model state
        unet_state = unet.state_dict()
    
    torch.save({
        "unet": unet_state,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "global_step": global_step,
        "epoch": epoch,
        "is_lora": is_lora,
    }, path)

def load_checkpoint(path, unet, optimizer=None, scheduler=None, scaler=None, device="cuda"):
    """Load training checkpoint.
    
    Restores the UNet model, optimizer, scheduler, and scaler states from
    a saved checkpoint to resume training. Handles both full models and LoRA-adapted models.
    
    Args:
        path (str): Path to checkpoint file.
        unet (UNet2DConditionModel or PeftModel): The UNet model to load state into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to restore.
        scheduler (torch.optim.lr_scheduler.LambdaLR, optional): The scheduler to restore.
        scaler (torch.amp.GradScaler, optional): The scaler to restore.
        device (str): Device to load checkpoint on. Defaults to 'cuda'.
    
    Returns:
        tuple: (global_step, epoch) - Training step and epoch from checkpoint.
    """
    ckpt = torch.load(path, map_location=device)
    
    # Load UNet state (skip if LoRA, as LoRA adapters are loaded separately)
    if ckpt.get("unet") is not None:
        unet.load_state_dict(ckpt["unet"])
    
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt.get("global_step", 0), ckpt.get("epoch", 0)