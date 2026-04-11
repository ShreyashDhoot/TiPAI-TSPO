import torch

def unet_forward(unet_model, zt, t, encoder_hidden, mask_l, masked_latent):
    """Forward pass through the inpainting UNet.
    
    Concatenates the noisy latent, mask, and masked latent as input to the UNet
    and performs a forward pass with text encoder embeddings.
    
    Args:
        unet_model (UNet2DConditionModel): The UNet model.
        zt (torch.Tensor): Noisy latent at timestep t, shape (batch_size, 4, H, W).
        t (torch.Tensor): Timestep indices, shape (batch_size,).
        encoder_hidden (torch.Tensor): Text encoder hidden states,
                                      shape (batch_size, seq_len, hidden_dim).
        mask_l (torch.Tensor): Mask in latent space, shape (batch_size, 1, H, W)
                              or (batch_size, H, W).
        masked_latent (torch.Tensor): Masked version of original latent,
                                     shape (batch_size, 4, H, W).
    
    Returns:
        torch.Tensor: Model prediction (noise), shape (batch_size, 4, H, W).
    """
    if mask_l.dim() == 3:
        mask_l = mask_l.unsqueeze(1)
    unet_in = torch.cat([zt, mask_l, masked_latent], dim=1)
    return unet_model(unet_in, t, encoder_hidden_states=encoder_hidden).sample