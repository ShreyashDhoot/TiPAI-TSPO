import os
import torch
from PIL import Image
import numpy as np

def decode_latent_to_pil(vae, latent):
    """Decode a latent representation to a PIL image.
    
    Uses the VAE decoder to convert latent tensor to image pixel space,
    then converts to PIL Image format.
    
    Args:
        vae (AutoencoderKL): The VAE decoder.
        latent (torch.Tensor): Latent tensor, shape (4, H, W) or (batch_size, 4, H, W).
    
    Returns:
        PIL.Image: Decoded image in RGB format.
    """
    latent = latent.unsqueeze(0) if latent.dim() == 3 else latent
    latent = latent.to(device=vae.device, dtype=vae.dtype)
    latent = latent / vae.config.scaling_factor
    img = vae.decode(latent).sample
    img = (img / 2 + 0.5).clamp(0, 1)
    img = img[0].permute(1, 2, 0).detach().cpu().numpy()
    return Image.fromarray((img * 255).astype("uint8"))

def latent_mask_to_pil(mask_latent):
    """Convert a latent mask to a PIL image.
    
    Converts mask tensor to PIL Image format, handling both normalized
    (0-1) and 8-bit (0-255) representations.
    
    Args:
        mask_latent (torch.Tensor): Mask tensor, shape (1, H, W) or (H, W).
                                   Values in range [0, 1] or [0, 255].
    
    Returns:
        PIL.Image: Mask as grayscale PIL Image (mode 'L').
    """
    from PIL import Image
    import numpy as np
    m = mask_latent.detach().cpu()
    if m.dim() == 3:
        m = m[0]
    m = m.numpy()
    if m.max() <= 1.0:
        m = (m * 255).astype("uint8")
    else:
        m = m.astype("uint8")
    return Image.fromarray(m)


def preprocess_mask_for_eval(mask_latent, target_size, threshold=0.5, invert=False):
    """Convert latent mask to a clean binary mask for inpainting eval.

    Args:
        mask_latent (torch.Tensor): Latent-space mask tensor.
        target_size (tuple): (width, height) to match source image.
        threshold (float): Threshold in [0, 1] for binarization.
        invert (bool): Whether to invert mask polarity.

    Returns:
        PIL.Image: Binary grayscale mask image in mode 'L'.
    """
    m = mask_latent.detach().float().cpu()
    if m.dim() == 3:
        m = m[0]
    m = m.numpy()

    # Normalize to [0, 1] regardless of whether mask came as [0, 1] or [0, 255].
    if m.max() > 1.0:
        m = m / 255.0
    m = np.clip(m, 0.0, 1.0)

    m = (m >= float(threshold)).astype(np.uint8)
    if invert:
        m = 1 - m

    m = (m * 255).astype(np.uint8)
    mask = Image.fromarray(m, mode="L")
    if mask.size != target_size:
        mask = mask.resize(target_size, Image.NEAREST)
    return mask

def visual_eval(unet, pipe, val_vis_samples, step, out_dir, eval_cfg=None):
    """Generate and save inpainting results for visual evaluation.
    
    Performs inference on validation samples using the current UNet model
    and saves generated images for visual inspection during training.
    
    Args:
        unet (UNet2DConditionModel): The trainable UNet model.
        pipe (StableDiffusionInpaintPipeline): The inpainting pipeline.
        val_vis_samples (list): List of validation samples containing 'z0',
                              'mask_latent', and optionally 'prompt' keys.
        step (int): Current training step (used in filename).
        out_dir (str): Directory to save generated images.
    """
    os.makedirs(out_dir, exist_ok=True)
    unet.eval()
    
    # Default prompt if not provided in dataset
    default_prompt = "a high-quality inpainted image"
    
    eval_cfg = eval_cfg or {}
    eval_seed = int(eval_cfg.get("eval_seed", 1234))
    eval_threshold = float(eval_cfg.get("eval_mask_threshold", 0.5))
    eval_invert_mask = bool(eval_cfg.get("eval_invert_mask", False))
    eval_steps = int(eval_cfg.get("eval_num_inference_steps", 25))
    eval_guidance = float(eval_cfg.get("eval_guidance_scale", 7.5))

    with torch.no_grad():
        for i, sample in enumerate(val_vis_samples[:4]):
            z0 = sample["z0"]
            mask_latent = sample["mask_latent"]
            # Use provided prompt or fall back to default
            prompt = sample.get("prompt", default_prompt)

            img = decode_latent_to_pil(pipe.vae, z0)
            mask = preprocess_mask_for_eval(
                mask_latent,
                target_size=img.size,
                threshold=eval_threshold,
                invert=eval_invert_mask,
            )

            generator = torch.Generator(device=pipe.device).manual_seed(eval_seed + i)

            out = pipe(
                prompt=prompt,
                image=img,
                mask_image=mask,
                num_inference_steps=eval_steps,
                guidance_scale=eval_guidance,
                generator=generator,
            ).images[0]
            out.save(os.path.join(out_dir, f"eval_step{step}_sample{i}.png"))
    unet.train()