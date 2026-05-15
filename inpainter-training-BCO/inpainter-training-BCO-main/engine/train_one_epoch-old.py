import torch
import torch.nn.functional as F
import copy

from models.unet_wrapper import unet_forward
from models.diffusion_utils import q_sample
from losses.bco_loss import bco_loss, RewardShiftEMA


def train_loop(
    unet,
    ref_unet,          # ← NEW: frozen static reference (pass from train.py)
    vae,
    text_enc,
    scheduler,
    optimizer,
    lr_sched,
    scaler,
    train_loader,
    pipe,
    val_vis_samples,
    wandb_log_fn,
    save_fn,
    visual_eval_fn,
    cfg,
    device="cuda",
):
    unet.train()
    ref_unet.eval()    # always frozen
    vae.eval()
    text_enc.eval()
    g_std_ema=None 

    global_step   = 0
    micro_step    = 0
    accum_loss    = 0.0
    grad_norm     = torch.tensor(0.0, device=device)
    reward_shift_ema = RewardShiftEMA(momentum=0.999)   # for bco loss was kl_ema_scaler=None before 

    max_epochs       = cfg["training"].get("max_epochs", 100)
    grad_accum_steps = cfg["training"]["grad_accum_steps"]
    log_every        = cfg["training"]["log_every"]
    beta             = cfg["training"]["beta"]
    recon_weight     = cfg["training"].get("recon_weight", 150.0)
    identity_weight  = cfg["training"].get("identity_weight", 0.1)
    kto_coeffs = cfg["training"].get("kto_coeffs", {"safe": 1.0, "nudity": 5.0, "violence": 12.0})

    # Accumulators for logging
    label_pos_count = 0
    label_neg_count = 0
    mse_gap_sum     = 0.0
    mse_gap_count   = 0
    debug_accum     = {}

    for epoch in range(max_epochs):
        for batch in train_loader:
            z0            = batch["z0"].to(device, non_blocking=True)
            masked_latent = batch["masked_latent"].to(device, non_blocking=True)
            mask_l        = batch["mask_latent"].to(device, non_blocking=True)
            input_ids     = batch["input_ids"].to(device, non_blocking=True)
            label         = batch["label"].to(device, non_blocking=True)

            with torch.no_grad():
                enc_hidden = text_enc(input_ids).last_hidden_state

            t     = torch.randint(200,900, (z0.shape[0],), device=device)
            noise = torch.randn_like(z0)
            zt    = q_sample(z0, t, noise, scheduler)

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                # Trainable forward pass
                pred_train = unet_forward(unet, zt, t, enc_hidden, mask_l, masked_latent)

                # Frozen reference forward pass — no weight swapping needed
                with torch.no_grad():
                    pred_ref = unet_forward(ref_unet, zt, t, enc_hidden, mask_l, masked_latent)
'''
                loss, debug = kto_loss(
                    pred_train     = pred_train,
                    pred_ref       = pred_ref,
                    noise          = noise,
                    label          = label,
                    mask_l         = mask_l,
                    z0             = z0,
                    zt             = zt,
                    t              = t,
                    scheduler      = scheduler,
                    beta           = beta,
                    recon_weight   = recon_weight,
                    identity_weight= identity_weight,
                    reward_shift_ema   = kl_ema,
                    kto_coeffs=kto_coeffs,
                )
'''
                loss = loss / grad_accum_steps
                reward_shift_ema.update(delta_raw)

                g_std_val = debug["reward_std"]
'''
                if g_std_ema is None:
                    g_std_ema = g_std_val
                else:
                    g_std_ema = 0.99 * g_std_ema + 0.01 * g_std_val
'''
                '''
                if g_std_ema > 1e-9:
                    beta = float(torch.clamp(
                        torch.tensor(0.35 / g_std_ema),
                        min=1.0, max=cfg["training"]["beta"]   # config beta = ceiling, not fixed value
                    ))
                '''
                beta=cfg["training"]["beta"]

            # ── KL EMA scalar update (replaces the old weight-swap EMA) ──────
            
            with torch.no_grad():
                kl_now = debug["kl_current"]
                if kl_ema_scalar is None:
                    kl_ema_scalar = kl_now
                else:
                    kl_ema_scalar = 0.999 * kl_ema_scalar + 0.001 * kl_now

            # ── Logging accumulators ─────────────────────────────────────────
            with torch.no_grad():
                is_safe = label[:, 0].bool()
                is_nudity = label[:, 1].bool()
                is_violence = label[:, 2].bool()

                mse_gap = abs(debug["mse_train_z0"] - debug["mse_ref_z0"])
                mse_gap_sum   += mse_gap
                mse_gap_count += 1

                for k, v in debug.items():
                    debug_accum[k] = debug_accum.get(k, 0.0) + v

            scaler.scale(loss).backward()
            accum_loss += loss.item() * grad_accum_steps
            micro_step += 1

            if micro_step % grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    unet.parameters(), cfg["training"]["grad_clip_norm"]
                )
                scaler.step(optimizer)
                scaler.update()
                lr_sched.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % log_every == 0:
                    avg_loss    = accum_loss / float(log_every)
                    avg_mse_gap = mse_gap_sum / float(max(1, mse_gap_count))
                    avg_debug   = {k: v / float(log_every) for k, v in debug_accum.items()}
                    h_s = avg_debug.get("h_safe", 0.5)
                    delta_n = avg_debug.get("h_nudity", 0.5) - h_s
                    delta_v = avg_debug.get("h_violence", 0.5) - h_s

                    wandb_log_fn({
                        "train/loss":              avg_loss,
                        "train/reward_gap":        avg_debug.get("g_term_mean", 0),
                        "train/grad_norm":         grad_norm.item(),
                        "train/lr":                lr_sched.get_last_lr()[0],
                        "train/epoch":             epoch,
                        "train/label_pos_count":   label_pos_count,
                        "train/label_neg_count":   label_neg_count,
                        "train/mse_gap_avg":       avg_mse_gap,
                        "debug/g_term_mean":       avg_debug.get("g_term_mean", 0),
                        "debug/g_term_std":        avg_debug.get("g_term_std", 0),
                        "debug/kl_current":        avg_debug.get("kl_current", 0),
                        "debug/kl_ema":            kl_ema_scalar or 0,
                        "debug/mse_train_z0":      avg_debug.get("mse_train_z0", 0),
                        "debug/mse_ref_z0":        avg_debug.get("mse_ref_z0", 0),
                        "debug/h_safe":            avg_debug.get("h_safe", 0),
                        "debug/h_unsafe":          avg_debug.get("h_unsafe", 0),
                        "debug/identity_gap":      avg_debug.get("identity_gap", 0),
                        "debug/identity_loss":     avg_debug.get("identity_loss", 0),
                        "debug/ref_mse_cap":       avg_debug.get("ref_mse_cap", 0),
                    }, step=global_step)

                    print(
                        f"step={global_step} loss={avg_loss:.4f} "
                        f"h_S={h_s:.3f} ΔN={delta_n:.3f} ΔV={delta_v:.3f} "
                        f"id_gap={avg_debug.get('identity_gap',0):.4f} "
                        f"lr={lr_sched.get_last_lr()[0]:.2e} beta={beta:.1f}"
                    )

                    accum_loss = 0.0
                    mse_gap_sum = 0.0
                    mse_gap_count = 0
                    debug_accum = {}

                if global_step % cfg["training"]["save_every"] == 0:
                    save_fn(global_step, unet, optimizer, lr_sched, scaler, epoch)
                    visual_eval_fn(unet, pipe, val_vis_samples, global_step)

                if global_step >= cfg["training"]["max_steps"]:
                    return