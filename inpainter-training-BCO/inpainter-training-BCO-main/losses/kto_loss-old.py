import torch
import torch.nn.functional as F

def _predict_z0(pred_noise, zt, t, scheduler):
    """Reconstruct denoised latent z0 from noise prediction using DDPM formula."""
    alpha_bar = scheduler.alphas_cumprod.to(device=zt.device, dtype=zt.dtype)
    a = alpha_bar[t].view(-1, 1, 1, 1).sqrt()
    sigma = (1 - alpha_bar[t]).view(-1, 1, 1, 1).sqrt()
    return (zt - sigma * pred_noise) / a.clamp(min=1e-6)

def kto_loss(
    pred_train,
    pred_ref,
    noise,
    label,           # [B, 3] One-Hot: (Safe, Nudity, Violence)
    mask_l,
    z0,
    zt,
    t,
    scheduler,
    beta=7,
    recon_weight=200.0,
    identity_weight=30.0,
    kl_ema=None,
    mask_weight=0.5,
    kto_coeffs=None  # {'safe': 1.0, 'nudity': 5.0, 'violence': 12.0}
):
    """
    Tri-Class Fidelity-Guided KTO Loss.
    
    Enforces refusal while maintaining image manifold via:
    - Reconstruction penalty against individual image z0
    - Global Identity Guardrail
    - Class-weighted KTO signal
    """
    if kto_coeffs is None:
        kto_coeffs = {"safe": 1.0, "nudity": 5.0, "violence": 12.0}

    # ── 1. Reconstruct z0 predictions ─────────────────────────────────────
    with torch.no_grad():
        z0_pred_ref = _predict_z0(pred_ref.float(), zt.float(), t, scheduler)

    z0_pred_train = _predict_z0(pred_train.float(), zt.float(), t, scheduler)

    # ── 2. Per-sample MSE against original z0 (The Anchor) ────────────────
    mask_expanded = mask_l.expand_as(z0).to(dtype=torch.float32)
    mask_pixels   = mask_expanded.sum(dim=[1, 2, 3]).clamp(min=1)

    mse_train_z0_sample = F.mse_loss(z0_pred_train.float(), z0.float(), reduction="none")
    mse_ref_z0_sample   = F.mse_loss(z0_pred_ref.float(),   z0.float(), reduction="none")

    if mask_weight > 0:
        w = 1.0 + mask_weight * mask_l.to(dtype=torch.float32)
        mse_train_z0_sample = mse_train_z0_sample * w.expand_as(mse_train_z0_sample)
        mse_ref_z0_sample   = mse_ref_z0_sample   * w.expand_as(mse_ref_z0_sample)

    mse_train_masked = (mse_train_z0_sample * mask_expanded).sum(dim=[1,2,3]) / mask_pixels
    mse_ref_masked   = (mse_ref_z0_sample   * mask_expanded).sum(dim=[1,2,3]) / mask_pixels

    # ── 3. g_term logic ──────────────────────────────────────────────────
    # Positive = train drifted further from z0 than ref.
    g_term = mse_train_masked - mse_ref_masked

    # ── 4. Hinge Clamp (per class) ───────────────────────────────────────
    # We detach the ref failure mean to use as a baseline for the hinge
    ref_mse_mean = mse_ref_masked.mean().detach()
    g_term_unsafe_cap = ref_mse_mean * 1.5
    
    # Apply hinge only to unsafe (Nudity/Violence)
    is_safe = label[:, 0].bool()
    g_term = torch.where(is_safe, g_term, g_term.clamp(max=g_term_unsafe_cap))

    # ── 5. KL Centering (Safe-only baseline) ──────────────────────────────
    if is_safe.any():
        kl_baseline_raw = g_term[is_safe].mean().detach()
    else:
        kl_baseline_raw = torch.tensor(0., device=label.device, dtype=g_term.dtype)
    
    if kl_ema is not None:
        kl_baseline = torch.tensor(kl_ema, device=label.device, dtype=g_term.dtype)
    else:
        kl_baseline = kl_baseline_raw.clamp(min=-0.02, max=0.05)
    
    g_term_centered = g_term - kl_baseline

    # ── 6. KTO sigmoid & Class-Wise Torque ────────────────────────────────
    # label_sgn: Safe=+1, Nudity/Violence=-1
    # Note: label[:,0] is Safe. label[:,1:] are Unsafe.
    label_sgn = torch.where(is_safe, 1.0, -1.0)
    
    # h is the KTO satisfaction probability
    h = torch.sigmoid(label_sgn * beta * g_term_centered)

    # Apply class weights (w_y)
    w_y = torch.zeros_like(h)
    w_y[label[:, 0] == 1] = kto_coeffs["safe"]
    w_y[label[:, 1] == 1] = kto_coeffs["nudity"]
    w_y[label[:, 2] == 1] = kto_coeffs["violence"]

    kto_loss_per_sample = w_y * (1.0 - h)
    kto_loss_final = kto_loss_per_sample.mean()

    # ── 7. Global Reconstruction Loss (Steel Anchor) ──────────────────────
    # Noise-space recon for unmasked background across ALL images
    unmask = (1.0 - mask_l).expand_as(pred_train)
    recon_loss = F.mse_loss(
        pred_train * unmask,
        noise      * unmask,
        reduction="mean",
    )

    # ── 8. Identity Guardrail (Global z0-space protection) ────────────────
    identity_gap = F.mse_loss(
        z0_pred_train.float(),
        z0_pred_ref.float().detach(),
        reduction="mean",
    )
    drift_threshold = 0.02
    identity_loss = (F.relu(identity_gap - drift_threshold)) ** 2

    # ── 9. Final Combined Loss ────────────────────────────────────────────
    # User weight of 4 on KTO term
    loss = 4.0 * kto_loss_final + recon_weight * recon_loss + identity_weight * identity_loss

    # ── Debug Metrics ─────────────────────────────────────────────────────
    safe_mask = label[:, 0].bool()
    nudity_mask = label[:, 1].bool()
    violence_mask = label[:, 2].bool()

    debug = {
        "g_term_mean":     g_term.mean().item(),
        "g_term_std":      g_term.std().item(),
        "kl_current":      kl_baseline_raw.item(),
        "mse_train_z0":    mse_train_masked.mean().item(),
        "mse_ref_z0":      mse_ref_masked.mean().item(),
        "h_safe":          h[safe_mask].mean().item() if safe_mask.any() else 0.5,
        "h_nudity":        h[nudity_mask].mean().item() if nudity_mask.any() else 0.5,
        "h_violence":      h[violence_mask].mean().item() if violence_mask.any() else 0.5,
        "kto_loss":        kto_loss_final.item(),
        "recon_loss":      recon_loss.item(),
        "identity_loss":   identity_loss.item(),
        "identity_gap":    identity_gap.item(),
    }

    return loss, debug