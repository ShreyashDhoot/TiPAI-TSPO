import torch
import torch.nn.functional as F
import copy

from models.unet_wrapper import unet_forward
from models.diffusion_utils import q_sample
from losses.bco_loss import bco_loss, RewardShiftEMA


# ─────────────────────────────────────────────────────────────────────────────
# Multi-step denoising helper
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _multistep_denoise(
    unet,
    scheduler,
    z_start,        # [B, C, H, W]  noisy latent at t_start
    t_start,        # [B]            starting timesteps (int tensor)
    enc_hidden,     # [B, S, D]
    mask_l,         # [B, 1, H, W]
    masked_latent,  # [B, C, H, W]
    num_steps,      # int  how many denoising steps to unroll (2 or 3)
    device,
):
    """
    Run `num_steps` DDPM denoising steps starting from (z_start, t_start).

    For each sample the timestep schedule is evenly spaced from t_start[i]
    down to 0, e.g. for num_steps=2 with t_start=800:  [800, 400].

    Returns the predicted z0 after the final step (no grad).
    """
    B = z_start.shape[0]
    zt = z_start.clone()

    # Build per-sample step schedules  shape [B, num_steps]
    t_np = t_start.cpu().tolist()
    step_grids = []
    for ti in t_np:
        steps = torch.linspace(ti, 0, num_steps + 1, dtype=torch.long)[:-1]
        step_grids.append(steps)
    step_grids = torch.stack(step_grids, dim=0).to(device)  # [B, num_steps]

    alpha_bar = scheduler.alphas_cumprod.to(device=device, dtype=zt.dtype)

    for s in range(num_steps):
        t_s = step_grids[:, s]
        
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            pred = unet_forward(unet, zt, t_s, enc_hidden, mask_l, masked_latent)

        # 1. Predict clean z0 (same as before)
        a = alpha_bar[t_s].view(B, 1, 1, 1).sqrt()
        sigma = (1 - alpha_bar[t_s]).view(B, 1, 1, 1).sqrt()
        z0_hat = (zt - sigma * pred) / a.clamp(min=1e-6)

        # 2. DDIM Step (Deterministic η=0)
        if s < num_steps - 1:
            t_next = step_grids[:, s + 1]
            a_next = alpha_bar[t_next].view(B, 1, 1, 1).sqrt()
            
            # This is the "Direction pointing to x_t" in DDIM formula
            # It replaces the random noise injection
            sigma_next = (1 - alpha_bar[t_next]).view(B, 1, 1, 1).sqrt()
            zt = a_next * z0_hat + sigma_next * pred  # <--- DETERMINISTIC PATH
            
    return z0_hat  # [B, C, H, W]  predicted clean latent after final step


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_loop(
    unet,
    ref_unet,          # frozen static reference
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
    ref_unet.eval()
    text_enc.eval()
    g_std_ema = None

    global_step   = 0
    micro_step    = 0
    accum_loss    = 0.0
    grad_norm     = torch.tensor(0.0, device=device)

    # ── BCO reward-shift EMA ──────────────────────────────────────────────────
    reward_shift_ema = RewardShiftEMA(momentum=0.999)

    # ── VAE on training device (no aesthetic scorer) ──────────────────────────
    vae.to(device)
    vae.eval()

    # ── Pre-compute null (empty) prompt embedding for prompt dropping ─────────
    # Tokenize a single empty string once; reuse every batch.
    # Shape: [1, S, D] → broadcast over batch during dropped steps.
    with torch.no_grad():
        null_ids = pipe.tokenizer(
            [""],
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(device)
        null_hidden = text_enc(null_ids).last_hidden_state  # [1, S, D]

    max_epochs          = cfg["training"].get("max_epochs", 100)
    grad_accum_steps    = cfg["training"]["grad_accum_steps"]
    log_every           = cfg["training"]["log_every"]
    beta                = cfg["training"]["beta"]
    recon_weight        = cfg["training"].get("recon_weight",   200.0)
    identity_weight     = cfg["training"].get("identity_weight", 30.0)
    bco_coeffs = cfg["training"].get("kto_coeffs", {"safe": 1.0, "nudity": 5.0, "violence": 12.0})

    # ── Timestep range: full diffusion schedule ───────────────────────────────
    # Previously [200, 900]; expanded to [0, 1000) so the model learns the
    # alignment policy at all noise levels, matching inference behaviour.
    t_min = cfg["training"].get("t_min", 0)
    t_max = cfg["training"].get("t_max", 1000)

    # ── Multi-step unrolling config ───────────────────────────────────────────
    # Every `unroll_every` global steps, run `unroll_steps` actual denoising
    # steps and compute BCO on the multi-step output.  This directly bridges
    # the single-step training / multi-step inference distribution gap.
    #
    # Defaults: unroll_every=10 (10 % of steps), unroll_steps=2 (2-step).
    # Override in config:
    #   training:
    #     unroll_every: 10
    #     unroll_steps: 2
    unroll_every = cfg["training"].get("unroll_every", 10)
    unroll_steps = cfg["training"].get("unroll_steps", 2)

    # ── Prompt dropping probability ───────────────────────────────────────────
    # With probability p_drop, replace the text conditioning with the null
    # embedding.  Forces the model to use spatial/visual context to decide
    # WHERE to suppress, fixing the spatial grounding failure (arm smudging).
    p_drop = cfg["training"].get("prompt_drop_prob", 0.10)

    # ── Identity weight on unsafe unmasked region ─────────────────────────────
    # Small identity penalty on unsafe samples' background prevents the model
    # from bleeding suppression into arms/hands outside the mask.
    identity_weight_unsafe = cfg["training"].get("identity_weight_unsafe", 5.0)

    label_pos_count = 0
    label_neg_count = 0
    mse_gap_sum     = 0.0
    mse_gap_count   = 0
    debug_accum     = {}
    unroll_count    = 0
    drop_count      = 0   # samples dropped (prompt replaced with null) this log window

    for epoch in range(max_epochs):
        for batch in train_loader:
            z0            = batch["z0"].to(device, non_blocking=True)
            masked_latent = batch["masked_latent"].to(device, non_blocking=True)
            mask_l        = batch["mask_latent"].to(device, non_blocking=True)
            input_ids     = batch["input_ids"].to(device, non_blocking=True)
            label         = batch["label"].to(device, non_blocking=True)

            B = z0.shape[0]

            with torch.no_grad():
                enc_hidden = text_enc(input_ids).last_hidden_state

            # ── Prompt dropping ───────────────────────────────────────────────
            # Per-sample: independently drop each sample's conditioning.
            # Dropped samples get the null embedding (broadcast [1,S,D] → [1,S,D]).
            n_dropped = 0
            if p_drop > 0.0:
                drop_mask = torch.rand(B, device=device) < p_drop  # [B] bool
                if drop_mask.any():
                    n_dropped = drop_mask.sum().item()
                    null_exp = null_hidden.expand(B, -1, -1)        # [B, S, D]
                    enc_hidden = torch.where(
                        drop_mask.view(B, 1, 1).expand_as(enc_hidden),
                        null_exp,
                        enc_hidden,
                    )
            drop_count += n_dropped

            # ── Sample from full timestep range [t_min, t_max) ───────────────
            t     = torch.randint(t_min, t_max, (z0.shape[0],), device=device)
            noise = torch.randn_like(z0)
            zt    = q_sample(z0, t, noise, scheduler)

            if label.dim() == 2:
                is_safe = label[:, 0].bool()
            else:
                is_safe = label.bool()

            recon_weight_t = torch.where(
                is_safe,
                torch.full((B,), float(recon_weight), device=device),
                torch.zeros(B,                         device=device),
            )
            # Safe samples: full identity weight.
            # Unsafe samples: small identity weight on unmasked region only —
            # prevents suppression from bleeding into arms/background outside mask.
            identity_weight_t = torch.where(
                is_safe,
                torch.full((B,), float(identity_weight),        device=device),
                torch.full((B,), float(identity_weight_unsafe), device=device),
            )

            # ── Decide whether to use multi-step unrolling this step ─────────
            use_unroll = (global_step % unroll_every == 0) and (unroll_steps > 1)

            # ── PASS 1: standard single-step BCO ─────────────────────────────
            # Compute loss, immediately scale and accumulate gradients, then
            # FREE the activation graph before pass 2.  This is the key fix:
            # never hold two full UNet activation graphs in VRAM simultaneously.
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                pred_train = unet_forward(unet, zt, t, enc_hidden, mask_l, masked_latent)
                with torch.no_grad():
                    pred_ref = unet_forward(ref_unet, zt, t, enc_hidden, mask_l, masked_latent)

                loss_single, debug, delta_raw = bco_loss(
                    pred_train       = pred_train,
                    pred_ref         = pred_ref,
                    noise            = noise,
                    label            = label,
                    mask_l           = mask_l,
                    z0               = z0,
                    zt               = zt,
                    t                = t,
                    scheduler        = scheduler,
                    beta             = beta,
                    recon_weight     = recon_weight_t,
                    identity_weight  = identity_weight_t,
                    reward_shift_ema = reward_shift_ema.value,
                    bco_coeffs       = bco_coeffs,
                )
                # Weight: 0.5 on unroll steps so combined gradient scale is stable
                w_single = 0.5 if use_unroll else 1.0
                loss1 = (w_single * loss_single) / grad_accum_steps

            # Backward pass 1 — frees pred_train activation graph entirely
            scaler.scale(loss1).backward()
            debug["unroll_bco_loss"] = float("nan")

            # ── PASS 2: multi-step unrolling (only every unroll_every steps) ──
            # Pass-1 graph is fully freed; we now have headroom for a second
            # UNet forward at a lower (closer-to-inference) noise level.
            if use_unroll:
                # 2a: build z_mid entirely under no_grad
                with torch.no_grad():
                    if unroll_steps > 1 and (t >= 100).any():
                        z_mid = _multistep_denoise(
                            unet          = unet,
                            scheduler     = scheduler,
                            z_start       = zt,
                            t_start       = t,
                            enc_hidden    = enc_hidden,
                            mask_l        = mask_l,
                            masked_latent = masked_latent,
                            num_steps     = unroll_steps - 1,
                            device        = device,
                        )
                    else:
                        z_mid = zt

                # 2b: re-noise z_mid to t_final
                t_final     = (t // unroll_steps).clamp(min=0, max=t_max - 1)
                noise_final = torch.randn_like(z0)
                alpha_bar   = scheduler.alphas_cumprod.to(device=device, dtype=z0.dtype)
                a_f   = alpha_bar[t_final].view(B, 1, 1, 1).sqrt()
                sig_f = (1 - alpha_bar[t_final]).view(B, 1, 1, 1).sqrt()
                zt_final = a_f * z_mid.detach() + sig_f * noise_final

                # 2c: grad-tracked forward at the lower noise level
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    pred_train_unroll = unet_forward(
                        unet, zt_final, t_final, enc_hidden, mask_l, masked_latent
                    )
                    with torch.no_grad():
                        pred_ref_unroll = unet_forward(
                            ref_unet, zt_final, t_final, enc_hidden, mask_l, masked_latent
                        )

                    loss_unroll, debug_unroll, _ = bco_loss(
                        pred_train       = pred_train_unroll,
                        pred_ref         = pred_ref_unroll,
                        noise            = noise_final,
                        label            = label,
                        mask_l           = mask_l,
                        z0               = z0,
                        zt               = zt_final,
                        t                = t_final,
                        scheduler        = scheduler,
                        beta             = beta,
                        recon_weight     = recon_weight_t,
                        identity_weight  = identity_weight_t,
                        reward_shift_ema = reward_shift_ema.value,
                        bco_coeffs       = bco_coeffs,
                    )
                    loss2 = (0.5 * loss_unroll) / grad_accum_steps

                # Backward pass 2 — gradients accumulate into same params
                scaler.scale(loss2).backward()
                debug["unroll_bco_loss"] = debug_unroll.get("bco_loss", float("nan"))
                unroll_count += 1
                loss_single = loss_single + loss_unroll  # for accum_loss logging

            loss = loss_single  # for accum_loss tracking below

            g_std_val = debug["reward_std"]
            if g_std_ema is None:
                g_std_ema = g_std_val
            else:
                g_std_ema = 0.99 * g_std_ema + 0.01 * g_std_val

            beta = cfg["training"]["beta"]

            with torch.no_grad():
                reward_shift_ema.update(delta_raw)

            with torch.no_grad():
                mse_gap = abs(debug["mse_train_z0"] - debug["mse_ref_z0"])
                mse_gap_sum   += mse_gap
                mse_gap_count += 1
                for k, v in debug.items():
                    if not (isinstance(v, float) and v != v):
                        debug_accum[k] = debug_accum.get(k, 0.0) + v

            # Backward passes have already been called above (pass 1 always,
            # pass 2 on unroll steps). Just track loss for logging.
            accum_loss += loss.item()
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

                    h_s     = avg_debug.get("h_safe",    0.5)
                    delta_n = avg_debug.get("h_nudity",  0.5) - h_s
                    delta_v = avg_debug.get("h_violence", 0.5) - h_s

                    wandb_log_fn({
                        # Core training
                        "train/loss":               avg_loss,
                        "train/reward_gap":         avg_debug.get("reward_mean",  0),
                        "train/grad_norm":          grad_norm.item(),
                        "train/lr":                 lr_sched.get_last_lr()[0],
                        "train/epoch":              epoch,
                        "train/label_pos_count":    label_pos_count,
                        "train/label_neg_count":    label_neg_count,
                        "train/mse_gap_avg":        avg_mse_gap,
                        "train/unroll_steps_done":  unroll_count,
                        "train/prompt_drops":       drop_count,
                        "train/prompt_drop_rate":   drop_count / max(1, log_every * B),
                        # Reward diagnostics
                        "debug/reward_mean":        avg_debug.get("reward_mean",  0),
                        "debug/reward_std":         avg_debug.get("reward_std",   0),
                        "debug/reward_safe":        avg_debug.get("reward_safe",  0),
                        "debug/reward_unsafe":      avg_debug.get("reward_unsafe",0),
                        # Reward shift
                        "debug/delta":              avg_debug.get("delta",        0),
                        "debug/delta_raw":          avg_debug.get("delta_raw",    0),
                        "debug/delta_ema":          reward_shift_ema.value,
                        # MSE
                        "debug/mse_train_z0":       avg_debug.get("mse_train_z0", 0),
                        "debug/mse_ref_z0":         avg_debug.get("mse_ref_z0",   0),
                        # Satisfaction h
                        "debug/h_safe":             avg_debug.get("h_safe",       0),
                        "debug/h_nudity":           avg_debug.get("h_nudity",     0),
                        "debug/h_violence":         avg_debug.get("h_violence",   0),
                        # Component losses
                        "debug/bco_loss":           avg_debug.get("bco_loss",     0),
                        "debug/unroll_bco_loss":    avg_debug.get("unroll_bco_loss", 0),
                        "debug/recon_loss":         avg_debug.get("recon_loss",   0),
                        "debug/identity_loss":      avg_debug.get("identity_loss",0),
                        "debug/identity_gap":       avg_debug.get("identity_gap", 0),
                        # Dynamic weight monitors
                        "debug/recon_per_safe":     avg_debug.get("recon_per_safe",    0),
                        "debug/recon_per_unsafe":   avg_debug.get("recon_per_unsafe",  0),
                        "debug/identity_per_safe":  avg_debug.get("identity_per_safe", 0),
                        "debug/identity_per_unsafe":avg_debug.get("identity_per_unsafe",0),
                        "debug/rw_mean_safe":       avg_debug.get("rw_mean_safe",  0),
                        "debug/rw_mean_unsafe":     avg_debug.get("rw_mean_unsafe",0),
                        "debug/iw_mean_safe":       avg_debug.get("iw_mean_safe",  0),
                        "debug/iw_mean_unsafe":     avg_debug.get("iw_mean_unsafe",0),
                    }, step=global_step)

                    print(
                        f"step={global_step} loss={avg_loss:.4f} "
                        f"h_S={h_s:.3f} ΔN={delta_n:.3f} ΔV={delta_v:.3f} "
                        f"δ_ema={reward_shift_ema.value:.4f} "
                        f"id_gap={avg_debug.get('identity_gap', 0):.4f} "
                        f"unrolls={unroll_count} "
                        f"drops={drop_count}({100*drop_count/max(1,log_every*B):.1f}%) "
                        f"lr={lr_sched.get_last_lr()[0]:.2e} beta={beta:.1f}"
                    )

                    accum_loss    = 0.0
                    mse_gap_sum   = 0.0
                    mse_gap_count = 0
                    debug_accum   = {}
                    drop_count    = 0

                if global_step % cfg["training"]["save_every"] == 0:
                    save_fn(global_step, unet, optimizer, lr_sched, scaler, epoch)
                    visual_eval_fn(unet, pipe, val_vis_samples, global_step)

                if global_step >= cfg["training"]["max_steps"]:
                    return
