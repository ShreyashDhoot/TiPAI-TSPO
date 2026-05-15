import random
import math
import torch
from torch.utils.data import Sampler

class StratifiedBatchSampler(Sampler):
    """
    Surgical Batching for Safe(8), Nudity(4), and Violence(4).
    The epoch length is defined by the largest minority class to ensure 
    high-signal diversity.
    """

    def __init__(self, labels, batch_size=16, safe_count=8, nudity_count=4, violence_count=4, shuffle=True):
        # Assuming labels is a 2D array/tensor [N, 3] -> (Safe, Nudity, Violence)
        # We extract indices based on the one-hot position
        # Convert to tensor immediately so [:, 0] works
        labels = torch.as_tensor(labels) 
        
        # 2. Safety Check: If it's 1D, the code will fail. 
        # We need it to be (N, 3)
        if labels.ndim == 1:
            raise ValueError(
                f"Labels must be 2D (N, 3) for Tri-Class sampling, "
                f"but got 1D tensor of shape {labels.shape}. "
                f"Check your get_all_labels() method!"
            )
        
        self.safe_idx = torch.where(labels[:, 0] == 1)[0].tolist()
        self.nudity_idx = torch.where(labels[:, 1] == 1)[0].tolist()
        self.violence_idx = torch.where(labels[:, 2] == 1)[0].tolist()
        
        self.batch_size = batch_size
        self.safe_count = safe_count
        self.nudity_count = nudity_count
        self.violence_count = violence_count
        self.shuffle = shuffle

        # The 'Long' class determines epoch length to ensure we don't repeat the same 
        # combinations too often. 
        max_class_len = max(len(self.safe_idx), len(self.nudity_idx), len(self.violence_idx))
        
        # We'll base our length on the class that needs the most 'batches' to be fully seen
        self._len = max(
            math.ceil(len(self.safe_idx) / safe_count),
            math.ceil(len(self.nudity_idx) / nudity_count),
            math.ceil(len(self.violence_idx) / violence_count)
        )

        print(f"[Sampler] S:{len(self.safe_idx)} | N:{len(self.nudity_idx)} | V:{len(self.violence_idx)}")
        print(f"[Sampler] Batch Structure: {safe_count}/{nudity_count}/{violence_count} | Batches/Epoch: {self._len}")

    def __iter__(self):
        # Create shuffled pools
        s_pool = self.safe_idx.copy()
        n_pool = self.nudity_idx.copy()
        v_pool = self.violence_idx.copy()

        if self.shuffle:
            random.shuffle(s_pool)
            random.shuffle(n_pool)
            random.shuffle(v_pool)

        # Cycle pools to match the required total per epoch
        def get_cycle(pool, count_needed):
            cycle = pool * ( (self._len * count_needed) // len(pool) + 2)
            if self.shuffle: random.shuffle(cycle)
            return cycle

        s_cycle = get_cycle(s_pool, self.safe_count)
        n_cycle = get_cycle(n_pool, self.nudity_count)
        v_cycle = get_cycle(v_pool, self.violence_count)

        s_ptr, n_ptr, v_ptr = 0, 0, 0

        for _ in range(self._len):
            batch = (
                s_cycle[s_ptr : s_ptr + self.safe_count] +
                n_cycle[n_ptr : n_ptr + self.nudity_count] +
                v_cycle[v_ptr : v_ptr + self.violence_count]
            )
            
            s_ptr += self.safe_count
            n_ptr += self.nudity_count
            v_ptr += self.violence_count

            if self.shuffle: random.shuffle(batch)
            yield batch

    def __len__(self):
        return self._len