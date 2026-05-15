import os
import glob
import bisect

import numpy as np
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
        self.parquet_files = []
        self.row_group_cums = {}
        self.file_to_idx = {}
        self._row_group_cache = {}

        for f in self.files:
            pf = pq.ParquetFile(f)
            n = pf.metadata.num_rows
            self.row_counts.append(n)
            self.cum_rows.append(self.cum_rows[-1] + n)
            self.file_to_idx[f] = len(self.parquet_files)
            self.parquet_files.append(pf)
            if len(self.parquet_files) == 1:
                print(f"DEBUG: Found columns in {f}:")
                print(pf.schema.names)

            rg_cum = [0]
            for rg_idx in range(pf.num_row_groups):
                rg_rows = pf.metadata.row_group(rg_idx).num_rows
                rg_cum.append(rg_cum[-1] + rg_rows)
            self.row_group_cums[f] = rg_cum

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
        file_idx = self.file_to_idx[f]
        pf = self.parquet_files[file_idx]
        rg_cum = self.row_group_cums[f]

        rg_idx = bisect.bisect_right(rg_cum, local_idx) - 1
        rg_local_idx = local_idx - rg_cum[rg_idx]

        cache_key = (f, rg_idx)
        table = self._row_group_cache.get(cache_key)
        if table is None:
            table = pf.read_row_group(rg_idx)
            self._row_group_cache.clear()
            self._row_group_cache[cache_key] = table

        row = table.slice(rg_local_idx, 1).to_pylist()[0]
        return row

    def __getitem__(self, idx):
        """Retrieve a sample from the dataset.
        
        Updated to return one-hot encoded labels: [safe, nudity, violence].
        """
        file_idx, local_idx = self._locate(idx)
        row = self._read_row(self.files[file_idx], local_idx)

        z0 = torch.from_numpy(np.asarray(row["z0"], dtype=np.float32))
        masked_latent = torch.from_numpy(np.asarray(row["masked_latent"], dtype=np.float32))
        mask_latent = torch.from_numpy(np.asarray(row["mask_latent"], dtype=np.float32))
        
        if mask_latent.ndim == 2:
            mask_latent = mask_latent.unsqueeze(0)
            
        input_ids = torch.from_numpy(np.asarray(row["input_ids"], dtype=np.int64))

        # NEW: Construct the 3-element one-hot label for the training loop
        # Format: [safe, nudity, violence]
        label = torch.tensor([
            float(row["safe"]), 
            float(row["nudity"]), 
            float(row["violence"])
        ], dtype=torch.float32)

        return {
            "z0": z0,
            "masked_latent": masked_latent,
            "mask_latent": mask_latent,
            "input_ids": input_ids,
            "label": label,  # This is now shape [3]
        }

    def get_all_labels(self):
        """
        Extracts all one-hot labels and returns an (N, 3) matrix.
        Crucial for the TriClassBatchSampler.
        """
        all_labels = []
        
        print(f"[Dataset] Loading all labels for stratified sampling...")
        for pf in self.parquet_files:
            # Efficiently read only the necessary columns
            table = pf.read(columns=["safe", "nudity", "violence"])
            
            # Convert columns to numpy arrays
            s = np.array(table.column("safe").to_pylist(), dtype=np.int64)
            n = np.array(table.column("nudity").to_pylist(), dtype=np.int64)
            v = np.array(table.column("violence").to_pylist(), dtype=np.int64)
            
            # Stack into (File_N, 3) matrix
            file_labels = np.stack([s, n, v], axis=1)
            all_labels.append(file_labels)
        
        # Combine all files into a single (Total_N, 3) matrix
        final_labels = np.concatenate(all_labels, axis=0)
        print(f"[Dataset] Successfully loaded labels matrix with shape: {final_labels.shape}")
        
        return final_labels