import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _predict_z0(pred_noise, zt, t, scheduler):
    """Reconstruct denoised latent z0 from noise prediction using DDPM formula."""
    alpha_bar = scheduler.alphas_cumprod.to(device=zt.device, dtype=zt.dtype)
    a     = alpha_bar[t].view(-1, 1, 1, 1).sqrt()
    sigma = (1 - alpha_bar[t]).view(-1, 1, 1, 1).sqrt()
    return (zt - sigma * pred_noise) / a.clamp(min=1e-6)


def _to_per_sample_weight(weight, reference_tensor):
    """
    Coerce weight into a [B] float tensor on the same device/dtype as
    reference_tensor.  Accepts float/int (broadcast) or [B] Tensor.
    """
    if isinstance(weight, torch.Tensor):
        return weight.to(dtype=reference_tensor.dtype,
                         device=reference_tensor.device)
    B = reference_tensor.shape[0]
    return torch.full((B,), float(weight),
                      dtype=reference_tensor.dtype,
                      device=reference_tensor.device)


# ─────────────────────────────────────────────────────────────────────────────
# BCO Loss
# ─────────────────────────────────────────────────────────────────────────────

def bco_loss(
    pred_train,
    pred_ref,
    noise,
    label,
    mask_l,
    z0,
    zt,
    t,
    scheduler,
    beta=7,
    recon_weight=200.0,         # float | [B] tensor; unsafe zeroed internally
    identity_weight=30.0,       # float | [B] tensor; unsafe forced to 0
    reward_shift_ema=None,      # float | None
    mask_weight=0.5,
    bco_coeffs=None,
):
    """
    Fidelity-Guided Binary Classifier Optimization (BCO) Loss.

    Loss components:
        L = 4*L_bco  +  L_recon_safe  +  L_identity_safe

    Aesthetic loss has been removed. The policy is shaped purely by the
    BCO alignment signal, reconstruction anchor, and identity guardrail.

    Identity loss:
        Applied to safe samples at full weight; applied to unsafe at a small
        weight (identity_weight_unsafe) to anchor the unmasked background region,
        preventing suppression from bleeding into arms/hands outside the mask.

    Args:
        pred_train       : [B, C, H, W] trainable UNet prediction
        pred_ref         : [B, C, H, W] frozen reference prediction
        noise            : [B, C, H, W] ground-truth noise
        label            : [B, 3] one-hot OR [B] binary (1=safe)
        mask_l           : [B, 1, H, W] inpainting mask
        z0               : [B, C, H, W] original clean latent
        zt               : [B, C, H, W] noised latent at t
        t                : [B] timesteps
        scheduler        : diffusion scheduler (.alphas_cumprod)
        beta             : BCE logit scale
        recon_weight     : background anchor weight (safe only)
        identity_weight  : identity guardrail weight; pass [B] tensor for per-sample control
        reward_shift_ema : delta EMA float from trainer, or None
        mask_weight      : extra weight on masked pixels in MSE
        bco_coeffs       : per-class BCE weights dict

    Returns:
        loss      : scalar Tensor
        debug     : dict of float scalars
        delta_raw : float for EMA update
    """

    # ── 0. Parse label ─────────────────────────────────────────────────────
    if label.dim() == 1:
        is_safe     = label.bool()
        is_nudity   = ~is_safe
        is_violence = torch.zeros_like(is_safe)
        tri_class   = False
    else:
        is_safe     = label[:, 0].bool()
        is_nudity   = label[:, 1].bool()
        is_violence = label[:, 2].bool()
        tri_class   = True

    if bco_coeffs is None:
        bco_coeffs = {"safe": 1.0, "nudity": 5.0, "violence": 12.0}

    # ── 1. z0 predictions ─────────────────────────────────────────────────
    with torch.no_grad():
        z0_pred_ref = _predict_z0(pred_ref.float(), zt.float(), t, scheduler)

    z0_pred_train = _predict_z0(pred_train.float(), zt.float(), t, scheduler)

    # ── 2. Masked MSE against z0 ──────────────────────────────────────────
    mask_expanded = mask_l.expand_as(z0).to(dtype=torch.float32)
    mask_pixels   = mask_expanded.sum(dim=[1, 2, 3]).clamp(min=1)

    mse_train_z0_sample = F.mse_loss(z0_pred_train.float(), z0.float(), reduction="none")
    mse_ref_z0_sample   = F.mse_loss(z0_pred_ref.float(),   z0.float(), reduction="none")

    if mask_weight > 0:
        w = 1.0 + mask_weight * mask_l.to(dtype=torch.float32)
        mse_train_z0_sample = mse_train_z0_sample * w.expand_as(mse_train_z0_sample)
        mse_ref_z0_sample   = mse_ref_z0_sample   * w.expand_as(mse_ref_z0_sample)

    mse_train_masked = (mse_train_z0_sample * mask_expanded).sum(dim=[1, 2, 3]) / mask_pixels
    mse_ref_masked   = (mse_ref_z0_sample   * mask_expanded).sum(dim=[1, 2, 3]) / mask_pixels

    # ── 3. Reward ──────────────────────────────────────────────────────────
    reward = mse_ref_masked - mse_train_masked

    # ── 4. Hinge cap on unsafe reward ─────────────────────────────────────
    ref_reward_mean = (
        reward[~is_safe].mean().detach()
        if (~is_safe).any()
        else torch.tensor(0., device=reward.device)
    )
    unsafe_reward_cap = ref_reward_mean.abs() * 1.5
    reward = torch.where(is_safe, reward, reward.clamp(max=unsafe_reward_cap.item()))

    # ── 5. Reward shift delta ──────────────────────────────────────────────
    mean_safe_reward = (
        reward[is_safe].mean().detach() if is_safe.any()
        else torch.tensor(0., device=reward.device)
    )
    mean_unsafe_reward = (
        reward[~is_safe].mean().detach() if (~is_safe).any()
        else torch.tensor(0., device=reward.device)
    )
    delta_raw = ((mean_safe_reward + mean_unsafe_reward) / 2.0).item()

    if reward_shift_ema is not None:
        delta = torch.tensor(reward_shift_ema, device=reward.device, dtype=reward.dtype)
    else:
        delta = torch.tensor(delta_raw, device=reward.device, dtype=reward.dtype)

    delta = delta.clamp(min=-0.03, max=0.03)

    # ── 6. BCO BCE Loss ────────────────────────────────────────────────────
    label_sgn = torch.where(is_safe, torch.ones_like(reward), -torch.ones_like(reward))
    r_shifted      = reward - delta
    bce_per_sample = F.softplus(-label_sgn * beta * r_shifted)

    # ── 7. Per-class weighting ─────────────────────────────────────────────
    w_y = torch.ones_like(bce_per_sample)
    if tri_class:
        w_y[is_safe]     = bco_coeffs["safe"]
        w_y[is_nudity]   = bco_coeffs["nudity"]
        w_y[is_violence] = bco_coeffs["violence"]
    else:
        w_y[is_safe]  = bco_coeffs["safe"]
        w_y[~is_safe] = bco_coeffs["nudity"]

    bco_loss_final = (w_y * bce_per_sample).mean()

    # ── 8. Reconstruction Loss — safe images only ─────────────────────────
    unmask           = (1.0 - mask_l).expand_as(pred_train)
    sq_err_recon     = ((pred_train * unmask) - (noise * unmask)) ** 2
    recon_per_sample = sq_err_recon.mean(dim=[1, 2, 3])

    rw     = _to_per_sample_weight(recon_weight, recon_per_sample)
    # recon_weight is already zeroed for unsafe by the caller (train_loop passes
    # a [B] tensor with 0 for unsafe samples). Don't override here.
    rw_sum = rw.sum().clamp(min=1e-6)
    recon_loss = (rw * recon_per_sample).sum() / rw_sum

    # ── 9. Identity Guardrail — safe images only ──────────────────────────
    sq_err_id           = (z0_pred_train.float() - z0_pred_ref.float().detach()) ** 2
    identity_per_sample = sq_err_id.mean(dim=[1, 2, 3])
    drift_threshold     = 0.02
    hinged_per_sample   = (F.relu(identity_per_sample - drift_threshold)) ** 2

    iw     = _to_per_sample_weight(identity_weight, hinged_per_sample)
    # identity_weight is a [B] tensor from the caller: full weight for safe,
    # identity_weight_unsafe (small, e.g. 3.0) for unsafe unmasked region.
    # Don't force unsafe to zero here — the caller controls the magnitude.
    iw_sum = iw.sum().clamp(min=1e-6)
    identity_loss = (iw * hinged_per_sample).sum() / iw_sum

    identity_gap = identity_per_sample.mean().detach()

    # ── 10. Combined Loss ─────────────────────────────────────────────────
    loss = 4.0 * bco_loss_final + recon_loss + identity_loss

    # ── Debug ─────────────────────────────────────────────────────────────
    h = torch.sigmoid(label_sgn * beta * r_shifted).detach()

    debug = {
        "reward_mean":            reward.mean().item(),
        "reward_std":             reward.std().item(),
        "reward_safe":            reward[is_safe].mean().item()     if is_safe.any()    else float("nan"),
        "reward_unsafe":          reward[~is_safe].mean().item()    if (~is_safe).any() else float("nan"),
        "delta":                  delta.item(),
        "delta_raw":              delta_raw,
        "mse_train_z0":           mse_train_masked.mean().item(),
        "mse_ref_z0":             mse_ref_masked.mean().item(),
        "h_safe":                 h[is_safe].mean().item()      if is_safe.any()     else float("nan"),
        "h_nudity":               h[is_nudity].mean().item()    if is_nudity.any()   else float("nan"),
        "h_violence":             h[is_violence].mean().item()  if is_violence.any() else float("nan"),
        "bco_loss":               bco_loss_final.item(),
        "recon_loss":             recon_loss.item(),
        "identity_loss":          identity_loss.item(),
        "identity_gap":           identity_gap.item(),
        "recon_per_safe":         recon_per_sample[is_safe].mean().item()    if is_safe.any()    else float("nan"),
        "recon_per_unsafe":       recon_per_sample[~is_safe].mean().item()   if (~is_safe).any() else float("nan"),
        "identity_per_safe":      identity_per_sample[is_safe].mean().item() if is_safe.any()    else float("nan"),
        "identity_per_unsafe":    identity_per_sample[~is_safe].mean().item() if (~is_safe).any() else float("nan"),
        "rw_mean_safe":           rw[is_safe].mean().item()    if is_safe.any()    else float("nan"),
        "rw_mean_unsafe":         rw[~is_safe].mean().item()   if (~is_safe).any() else float("nan"),
        "iw_mean_safe":           iw[is_safe].mean().item()    if is_safe.any()    else float("nan"),
        "iw_mean_unsafe":         iw[~is_safe].mean().item()   if (~is_safe).any() else float("nan"),
    }

    return loss, debug, delta_raw


# ─────────────────────────────────────────────────────────────────────────────
# Reward Shift EMA Helper
# ─────────────────────────────────────────────────────────────────────────────

class RewardShiftEMA:
    """
    Maintains an EMA of the BCO reward shift delta.

    Usage:
        ema = RewardShiftEMA(momentum=0.99)
        loss, debug, delta_raw = bco_loss(..., reward_shift_ema=ema.value)
        ema.update(delta_raw)
    """

    def __init__(self, momentum: float = 0.99, init: float = 0.0):
        self.momentum     = momentum
        self.value        = init
        self._initialized = False

    def update(self, delta_raw: float):
        if not self._initialized:
            self.value        = delta_raw
            self._initialized = True
        else:
            self.value = self.momentum * self.value + (1.0 - self.momentum) * delta_raw

    def state_dict(self):
        return {"value": self.value, "initialized": self._initialized}

    def load_state_dict(self, d):
        self.value        = d["value"]
        self._initialized = d["initialized"]