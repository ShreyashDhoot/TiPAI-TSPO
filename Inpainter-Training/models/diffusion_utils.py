def q_sample(z0, t, noise, scheduler):
    """Forward diffusion process: add noise to clean latent.
    
    Uses the scheduler to add noise to the clean latent at timestep t
    according to the diffusion schedule (q(z_t|z_0)).
    
    Args:
        z0 (torch.Tensor): Clean latent, shape (batch_size, 4, H, W).
        t (torch.Tensor): Timestep indices, shape (batch_size,).
        noise (torch.Tensor): Gaussian noise, shape (batch_size, 4, H, W).
        scheduler (DDPMScheduler): The noise scheduler.
    
    Returns:
        torch.Tensor: Noisy latent at timestep t, shape (batch_size, 4, H, W).
    """
    return scheduler.add_noise(z0, noise, t)