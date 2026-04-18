def latent_collate(batch):
    """Collate function for batching LatentInpaintDataset samples.
    
    Converts a list of sample dictionaries into a single batched dictionary
    where each value is a stacked tensor.
    
    Args:
        batch (list): List of dictionaries from LatentInpaintDataset,
                     each containing keys: 'z0', 'masked_latent', 'mask_latent',
                     'input_ids', 'label'.
    
    Returns:
        dict: Dictionary with same keys as input samples, but values are
             stacked tensors of shape (batch_size, ...).
    """
    keys = batch[0].keys()
    out = {}
    for k in keys:
        out[k] = [b[k] for b in batch]
    return {k: __import__("torch").stack(v) for k, v in out.items()}