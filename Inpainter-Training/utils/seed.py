import random
import numpy as np
import torch

def seed_everything(seed=42):
    """Set random seeds for reproducibility across all libraries.
    
    Sets seeds for Python's random module, NumPy, PyTorch CPU, and CUDA
    to ensure reproducible results during training.
    
    Args:
        seed (int): The random seed value. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)