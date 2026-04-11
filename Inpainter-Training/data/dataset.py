import os
import glob
import bisect
from pathlib import Path

import torch
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download


class LatentInpaintDataset(torch.utils.data.Dataset):
    """PyTorch dataset for loading inpainting data from Parquet files.
    
    Loads latent representations and masks from Parquet files, either from
    a HuggingFace Hub repository or a local directory. Stores cumulative
    row indices for efficient indexing across multiple files.
    """
    
    def __init__(self, repo_id=None, local_dir=None, split="train", cache_dir="./hf_cache"):
        """Initialize the dataset.
        
        Args:
            repo_id (str, optional): HuggingFace Hub dataset repository ID.
                                    Either repo_id or local_dir must be provided.
            local_dir (str, optional): Local directory containing Parquet files
                                      or split subdirectories.
            split (str): Name of the split subdirectory (e.g., 'train', 'val').
                        Defaults to 'train'.
            cache_dir (str): Directory to cache downloaded dataset.
                           Defaults to './hf_cache'.
        
        Raises:
            ValueError: If both repo_id and local_dir are None.
            FileNotFoundError: If no Parquet files are found in the dataset directory.
        """
        self.repo_id = repo_id
        self.split = split
        self.cache_dir = cache_dir

        if local_dir is None:
            if repo_id is None:
                raise ValueError("Provide either repo_id or local_dir")
            local_dir = snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=cache_dir,
                local_dir_use_symlinks=False,
            )

        self.root = os.path.join(local_dir, split) if os.path.isdir(os.path.join(local_dir, split)) else local_dir
        self.files = sorted(glob.glob(os.path.join(self.root, "*.parquet")))
        if not self.files:
            raise FileNotFoundError(f"No parquet files found in {self.root}")

        self.cum_rows = [0]
        self.row_counts = []

        for f in self.files:
            pf = pq.ParquetFile(f)
            n = pf.metadata.num_rows
            self.row_counts.append(n)
            self.cum_rows.append(self.cum_rows[-1] + n)

    def __len__(self):
        """Return total number of samples in the dataset.
        
        Returns:
            int: Total number of rows across all Parquet files.
        """
        return self.cum_rows[-1]

    def _locate(self, idx):
        """Locate which file and local index corresponds to a global index.
        
        Uses binary search on cumulative row counts to efficiently find
        which Parquet file contains a sample and its local index within that file.
        
        Args:
            idx (int): Global sample index.
        
        Returns:
            tuple: (file_idx, local_idx) - Parquet file index and local row index.
        """
        file_idx = bisect.bisect_right(self.cum_rows, idx) - 1
        local_idx = idx - self.cum_rows[file_idx]
        return file_idx, local_idx

    def _read_row(self, f, local_idx):
        """Read a single row from a Parquet file.
        
        Args:
            f (str): Path to Parquet file.
            local_idx (int): Row index within the file.
        
        Returns:
            dict: Dictionary with keys 'z0', 'masked_latent', 'mask_latent',
                 'input_ids', 'label' containing raw data from the Parquet file.
        """
        table = pq.read_table(f)
        row = table.slice(local_idx, 1).to_pylist()[0]
        return row

    def __getitem__(self, idx):
        """Retrieve a sample from the dataset.
        
        Locates the sample, reads it from disk, and converts to PyTorch tensors.
        
        Args:
            idx (int): Sample index.
        
        Returns:
            dict: Dictionary containing:
                - 'z0' (torch.Tensor): Clean latent, shape (4, H, W).
                - 'masked_latent' (torch.Tensor): Masked latent, shape (4, H, W).
                - 'mask_latent' (torch.Tensor): Inpainting mask, shape (1, H, W).
                - 'input_ids' (torch.Tensor): Text prompt token IDs.
                - 'label' (torch.Tensor): Quality label for KTO loss.
        """
        file_idx, local_idx = self._locate(idx)
        row = self._read_row(self.files[file_idx], local_idx)

        z0 = torch.tensor(row["z0"], dtype=torch.float32)
        masked_latent = torch.tensor(row["masked_latent"], dtype=torch.float32)
        mask_latent = torch.tensor(row["mask_latent"], dtype=torch.float32)
        if mask_latent.ndim == 2:
            mask_latent = mask_latent.unsqueeze(0)
        input_ids = torch.tensor(row["input_ids"], dtype=torch.long)
        label = torch.tensor(row["label"], dtype=torch.float32)

        return {
            "z0": z0,
            "masked_latent": masked_latent,
            "mask_latent": mask_latent,
            "input_ids": input_ids,
            "label": label,
        }