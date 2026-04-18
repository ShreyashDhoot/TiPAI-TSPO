import torch
import torch.nn.functional as F


def kto_loss(pred_train, pred_ref, noise, label, mask_l, beta=1000.0, mask_weight=1.0):
    """KTO (Kahneman-Tversky Optimization) loss with KL-centering and preference weighting.
    
    Implements KTO loss that encourages the trainable model to outperform the
    reference model on safe/preferred samples (label=1) while matching or
    underperforming on unsafe/adversarial samples (label=0).
    
    Key improvements over simple preference learning:
    - KL-centering: normalizes preference signal relative to KL divergence from reference
    - Asymmetric label handling: different gradient directions for safe vs. unsafe samples
    - High-precision scaling: large beta value enables sharp preference signal
    
    Args:
        pred_train (torch.Tensor): Noise predictions from trainable UNet,
                                  shape (batch_size, 4, H, W).
        pred_ref (torch.Tensor): Noise predictions from reference UNet,
                                shape (batch_size, 4, H, W).
        noise (torch.Tensor): Original noise target, shape (batch_size, 4, H, W).
        label (torch.Tensor): Preference labels, shape (batch_size,).
                             1.0 for safe/preferred, 0.0 for unsafe/adversarial.
        mask_l (torch.Tensor): Inpainting mask in latent space,
                      shape (batch_size, 1, H, W).
        beta (float): Scaling factor for preference signal. Defaults to 1000.0.
                     Higher values create sharper preference gradients.
        mask_weight (float): Extra weight applied to masked regions. Defaults to 1.0.
    
    Returns:
        torch.Tensor: Scalar KTO loss value.
    """
    # Compute MSE losses
    mse_train = F.mse_loss(pred_train, noise, reduction="none")
    mse_ref = F.mse_loss(pred_ref, noise, reduction="none")

    # Apply mask weighting to emphasize inpainted regions
    if mask_l is not None:
        weight = 1.0 + mask_weight * mask_l.to(dtype=pred_train.dtype)
        mse_train = mse_train * weight.expand_as(mse_train)
        mse_ref = mse_ref * weight.expand_as(mse_ref)

    # Average over spatial dimensions to get per-sample scalar loss
    mse_train = mse_train.mean(dim=[1, 2, 3])
    mse_ref = mse_ref.mean(dim=[1, 2, 3])
    
    # Preference gap: positive when reference is better (trainable is worse)
    # g_term = mse_ref - mse_train
    # For safe samples, we want g_term < 0 (trainable model better)
    # For unsafe samples, we want g_term >= 0 (trainable model worse or equal)
    g_term = mse_ref - mse_train
    
    # KL-centering: compute mean KL divergence and normalize
    # This stabilizes training by preventing extreme values from dominating
    kl = g_term.mean().detach()
    kl = torch.clamp(kl, min=0.0)  # Prevent negative KL from destabilizing
    g_term_centered = g_term - kl
    
    # Asymmetric label handling: convert binary labels to gradient direction
    # label=1 (safe): label_sgn = +1 (want to maximize model quality)
    # label=0 (unsafe): label_sgn = -1 (want to minimize model quality / stay near reference)
    labels_float = label.float()
    label_sgn = 2.0 * labels_float - 1.0  # Converts [0, 1] to [-1, +1]
    
    # Scale preference signal by beta and apply sigmoid
    label_scale_g = label_sgn * beta * g_term_centered
    h = torch.sigmoid(label_scale_g)
    
    # KTO loss: (1 - h)
    # For safe samples (label_sgn=+1):
    #   - When g_term < kl (model better): h→1, loss→0 (good!)
    #   - When g_term > kl (model worse): h→0, loss→1 (bad!)
    # For unsafe samples (label_sgn=-1):
    #   - When g_term > kl (model worse): h→0, loss→1 (good!)
    #   - When g_term < kl (model better): h→1, loss→0 (bad!)
    loss = (1.0 - h).mean()
    
    return loss