import torch
import torch.nn.functional as F


def kto_loss(pred_train, pred_ref, noise, label, mask_l, beta=0.1, mask_weight=1.0):
    """KTO (Kahneman-Tversky Optimization) loss for preference learning.
    
    Computes a preference-based loss that encourages the trainable model to
    make better predictions than the reference model on positive samples and
    worse predictions on negative samples. Uses a log-sigmoid formulation.
    
    Args:
        pred_train (torch.Tensor): Noise predictions from trainable UNet,
                                  shape (batch_size, 4, H, W).
        pred_ref (torch.Tensor): Noise predictions from reference UNet,
                                shape (batch_size, 4, H, W).
        noise (torch.Tensor): Original noise, shape (batch_size, 4, H, W).
        label (torch.Tensor): Quality labels, shape (batch_size,).
                             1.0 for positive samples, 0.0 for negative.
        mask_l (torch.Tensor): Inpainting mask in latent space,
                      shape (batch_size, 1, H, W).
        beta (float): Temperature/scaling parameter for the loss.
                     Controls the strength of preference signal. Defaults to 0.1.
        mask_weight (float): Extra weight applied to masked pixels. The masked
                     region gets weight 1.0 + mask_weight.
    
    Returns:
        torch.Tensor: Scalar loss value.
    """
    mse_train = F.mse_loss(pred_train, noise, reduction="none")
    mse_ref = F.mse_loss(pred_ref, noise, reduction="none")

    if mask_l is not None:
        weight = 1.0 + mask_weight * mask_l.to(dtype=pred_train.dtype)
        mse_train = mse_train * weight.expand_as(mse_train)
        mse_ref = mse_ref * weight.expand_as(mse_ref)

    mse_train = mse_train.mean(dim=[1, 2, 3])
    mse_ref = mse_ref.mean(dim=[1, 2, 3])
    log_ratio = -(mse_train - mse_ref) #<== trainable model learns better -> mse_train < mse_ref => log_ratio becomes -ve

    pos_mask = label.bool()
    neg_mask = ~pos_mask

    zero = torch.tensor(0.0, device=pred_train.device)
    pos_loss = -F.logsigmoid(beta * log_ratio[pos_mask]).mean() if pos_mask.any() else zero
    neg_loss = -F.logsigmoid(-beta * log_ratio[neg_mask]).mean() if neg_mask.any() else zero

    return 0.5 * (pos_loss + neg_loss)