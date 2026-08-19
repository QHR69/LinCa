"""
Dataset for training the invertible decomposition network

Data format:
    linca_data/data/cache_data/
    ├── prompt_0000/
    │   ├── cond/
    │   │   ├── step_00.pt  # [6889, 3072]
    │   │   ├── step_01.pt
    │   │   └── ...
    │   └── uncond/
    │       ├── step_00.pt
    │       └── ...
    └── prompt_0001/
        └── ...

Training sample construction:
    Build prediction samples for a given interval:
    - Full-compute steps: 0, 1, 2, then every interval after first_enhance
    - Prediction steps: every remaining step
    
    For each prediction step t:
    - Input: features from the 2-3 most recent full-compute steps
    - Target: the true feature at step t
"""

import os
import torch
import json
import random
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional
import numpy as np


class CacheDataset(Dataset):
    """
    Dataset used to train the invertible decomposition network
    
    Each sample contains:
    - target_feature: the feature to predict [seq_len, 3072]
    - cache_features: historical features used for prediction
    - step_distances: distance of each history feature from the target step
    - interval: interval used for this sample
    - target_step: target step index
    - branch: 'cond' or 'uncond'
    """
    
    def __init__(
        self,
        data_dir: str,
        intervals: List[int] = [5, 7, 10],
        first_enhance: int = 3,
        num_steps: int = 50,
        branches: List[str] = ['cond', 'uncond'],
        prompt_indices: Optional[List[int]] = None,
        max_cache_points: int = 3,  # maximum number of history points
    ):
        """
        Args:
            data_dir: path to the data directory (cache_data/)
            intervals: list of intervals to train on
            first_enhance: number of leading full-compute steps
            num_steps: total number of steps
            branches: branches to train
            prompt_indices: prompt indices to use; None means all
            max_cache_points: max history points used at prediction time
        """
        self.data_dir = data_dir
        self.intervals = intervals
        self.first_enhance = first_enhance
        self.num_steps = num_steps
        self.branches = branches
        self.max_cache_points = max_cache_points
        
        # Scan all prompt directories
        all_prompts = sorted([
            d for d in os.listdir(data_dir) 
            if d.startswith('prompt_') and os.path.isdir(os.path.join(data_dir, d))
        ])
        
        if prompt_indices is not None:
            self.prompt_dirs = [f'prompt_{i:04d}' for i in prompt_indices if f'prompt_{i:04d}' in all_prompts]
        else:
            self.prompt_dirs = all_prompts
        
        # Precompute every training sample
        self.samples = self._build_samples()
        
        print(f"Dataset: {len(self.prompt_dirs)} prompts, {len(self.samples)} samples")
        print(f"Intervals: {self.intervals}")
        print(f"Branches: {self.branches}")
    
    def _get_active_steps(self, interval: int) -> List[int]:
        """
        Full-compute steps for a given interval
        
        Same logic as LinCA:
        - Leading first_enhance steps: 0, 1, 2
        - Then every interval: first_enhance, first_enhance+interval, ...
        """
        active = list(range(self.first_enhance))  # [0, 1, 2]
        
        # From first_enhance onward, every interval steps
        step = self.first_enhance
        while step < self.num_steps:
            if step not in active:
                active.append(step)
            step += interval
        
        return sorted(active)
    
    def _get_cache_info(self, target_step: int, active_steps: List[int]) -> List[Tuple[int, int]]:
        """
        Collect cache info used to predict target_step
        
        Returns:
            List of (step_distance, active_step) tuples, sorted by distance (nearest first)
        """
        # Find every full-compute step earlier than target_step
        available = [s for s in active_steps if s < target_step]
        
        if not available:
            return []
        
        # Sort by distance (nearest first)
        available.sort(key=lambda s: target_step - s)
        
        # Return a list of (distance, step) pairs
        return [(target_step - s, s) for s in available[:self.max_cache_points]]
    
    def _build_samples(self) -> List[Dict]:
        """Build every training sample"""
        samples = []
        
        for prompt_dir in self.prompt_dirs:
            for branch in self.branches:
                for interval in self.intervals:
                    active_steps = self._get_active_steps(interval)
                    
                    # Create one sample for every non-full-compute step
                    for target_step in range(self.num_steps):
                        if target_step in active_steps:
                            continue  # skip full-compute steps
                        
                        cache_info = self._get_cache_info(target_step, active_steps)
                        if len(cache_info) == 0:
                            continue  # no usable cache
                        
                        samples.append({
                            'prompt_dir': prompt_dir,
                            'branch': branch,
                            'interval': interval,
                            'target_step': target_step,
                            'cache_info': cache_info,  # [(distance, step), ...]
                        })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        sample_info = self.samples[idx]
        
        prompt_dir = sample_info['prompt_dir']
        branch = sample_info['branch']
        target_step = sample_info['target_step']
        cache_info = sample_info['cache_info']
        interval = sample_info['interval']
        
        # Load the target feature
        target_path = os.path.join(
            self.data_dir, prompt_dir, branch, f'step_{target_step:02d}.pt'
        )
        target_feature = torch.load(target_path, map_location='cpu')
        # Cast to float32 (data is stored as bfloat16)
        target_feature = target_feature.float()
        
        # Load cached features
        cache_features = []
        step_distances = []
        for distance, step in cache_info:
            cache_path = os.path.join(
                self.data_dir, prompt_dir, branch, f'step_{step:02d}.pt'
            )
            cache_feature = torch.load(cache_path, map_location='cpu')
            # Cast to float32
            cache_feature = cache_feature.float()
            cache_features.append(cache_feature)
            step_distances.append(distance)
        
        return {
            'target_feature': target_feature,  # [seq_len, 3072]
            'cache_features': cache_features,  # List of [seq_len, 3072]
            'step_distances': step_distances,  # List of int
            'interval': interval,
            'target_step': target_step,
            'branch': branch,
            'prompt_dir': prompt_dir,
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function
    
    seq_len is fixed at 6889, so tensors can be stacked directly
    cache_features length may vary and needs special handling
    """
    target_features = torch.stack([item['target_feature'] for item in batch])
    
    # Find the largest cache count
    max_cache = max(len(item['cache_features']) for item in batch)
    
    # Pad cache features
    batch_size = len(batch)
    seq_len, dim = batch[0]['target_feature'].shape
    
    cache_features = torch.zeros(batch_size, max_cache, seq_len, dim)
    cache_mask = torch.zeros(batch_size, max_cache, dtype=torch.bool)
    step_distances = torch.zeros(batch_size, max_cache, dtype=torch.long)
    
    for i, item in enumerate(batch):
        num_cache = len(item['cache_features'])
        for j in range(num_cache):
            cache_features[i, j] = item['cache_features'][j]
            cache_mask[i, j] = True
            step_distances[i, j] = item['step_distances'][j]
    
    intervals = torch.tensor([item['interval'] for item in batch])
    target_steps = torch.tensor([item['target_step'] for item in batch])
    
    return {
        'target_features': target_features,      # [batch, seq_len, 3072]
        'cache_features': cache_features,        # [batch, max_cache, seq_len, 3072]
        'cache_mask': cache_mask,                # [batch, max_cache]
        'step_distances': step_distances,        # [batch, max_cache]
        'intervals': intervals,                  # [batch]
        'target_steps': target_steps,            # [batch]
        'branches': [item['branch'] for item in batch],
        'prompt_dirs': [item['prompt_dir'] for item in batch],
    }


class MemoryEfficientCacheDataset(Dataset):
    """
    Memory-efficient dataset (supports random interval sampling)
    
    Load data on demand instead of prefetching everything
    Speed up via a precomputed index
    
    Supports two modes:
    - random_interval=False: pre-build samples per interval (original behaviour)
    - random_interval=True: resample interval on every draw (fewer samples, more diversity)
    """
    
    def __init__(
        self,
        data_dir: str,
        intervals: List[int] = [5, 7, 10],
        first_enhance: int = 3,
        num_steps: int = 50,
        branches: List[str] = ['cond', 'uncond'],
        prompt_indices: Optional[List[int]] = None,
        max_cache_points: int = 3,
        subsample_seq: Optional[int] = None,  # subsample seq_len to reduce memory
        random_interval: bool = False,  # whether to sample interval at random
    ):
        self.data_dir = data_dir
        self.intervals = intervals
        self.first_enhance = first_enhance
        self.num_steps = num_steps
        self.branches = branches
        self.max_cache_points = max_cache_points
        self.subsample_seq = subsample_seq
        self.random_interval = random_interval
        
        # Scan all prompt directories
        all_prompts = sorted([
            d for d in os.listdir(data_dir) 
            if d.startswith('prompt_') and os.path.isdir(os.path.join(data_dir, d))
        ])
        
        if prompt_indices is not None:
            self.prompt_dirs = [f'prompt_{i:04d}' for i in prompt_indices if f'prompt_{i:04d}' in all_prompts]
        else:
            self.prompt_dirs = all_prompts
        
        # Precompute the sample index
        self.samples = self._build_sample_indices()
        
        mode_str = "random interval" if random_interval else "fixed interval"
        print(f"Dataset: {len(self.prompt_dirs)} prompts, {len(self.samples)} samples ({mode_str})")
        print(f"Intervals: {self.intervals}")
    
    def _get_active_steps(self, interval: int) -> List[int]:
        active = list(range(self.first_enhance))
        step = self.first_enhance
        while step < self.num_steps:
            if step not in active:
                active.append(step)
            step += interval
        return sorted(active)
    
    def _get_cache_info(self, target_step: int, active_steps: List[int]) -> List[Tuple[int, int]]:
        available = [s for s in active_steps if s < target_step]
        if not available:
            return []
        available.sort(key=lambda s: target_step - s)
        return [(target_step - s, s) for s in available[:self.max_cache_points]]
    
    def _build_sample_indices(self) -> List[Tuple]:
        """Build the sample index (do not load actual data)"""
        samples = []
        
        if self.random_interval:
            # Random-interval mode: one sample per (prompt, branch, target_step)
            # interval is sampled randomly inside __getitem__
            # Use the smallest interval to decide which target_steps are valid
            min_interval = min(self.intervals)
            active_steps_min = self._get_active_steps(min_interval)
            
            for prompt_idx, prompt_dir in enumerate(self.prompt_dirs):
                for branch in self.branches:
                    for target_step in range(self.num_steps):
                        if target_step in active_steps_min:
                            continue
                        
                        # Do not store interval/cache_info; compute them in __getitem__
                        samples.append((
                            prompt_idx,
                            branch,
                            target_step,
                        ))
        else:
            # Original mode: create samples for every interval
            for prompt_idx, prompt_dir in enumerate(self.prompt_dirs):
                for branch in self.branches:
                    for interval in self.intervals:
                        active_steps = self._get_active_steps(interval)
                        
                        for target_step in range(self.num_steps):
                            if target_step in active_steps:
                                continue
                            
                            cache_info = self._get_cache_info(target_step, active_steps)
                            if len(cache_info) == 0:
                                continue
                            
                            # Store indices only, not the tensors
                            samples.append((
                                prompt_idx,
                                branch,
                                interval,
                                target_step,
                                cache_info,
                            ))
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        if self.random_interval:
            # Random-interval mode
            prompt_idx, branch, target_step = self.samples[idx]
            
            # Sample interval uniformly at random
            interval = random.choice(self.intervals)
            
            # Compute active_steps and cache_info on the fly
            active_steps = self._get_active_steps(interval)
            
            # If target_step is a full-compute step under this interval, pick another interval
            attempts = 0
            while target_step in active_steps and attempts < len(self.intervals):
                interval = random.choice(self.intervals)
                active_steps = self._get_active_steps(interval)
                attempts += 1
            
            # If no interval works, fall back to the smallest interval
            if target_step in active_steps:
                # This should not happen; kept as a guard
                interval = min(self.intervals)
                active_steps = self._get_active_steps(interval)
            
            cache_info = self._get_cache_info(target_step, active_steps)
        else:
            # Original mode
            prompt_idx, branch, interval, target_step, cache_info = self.samples[idx]
        
        prompt_dir = self.prompt_dirs[prompt_idx]
        
        # Load the target feature
        target_path = os.path.join(
            self.data_dir, prompt_dir, branch, f'step_{target_step:02d}.pt'
        )
        target_feature = torch.load(target_path, map_location='cpu')
        # Cast to float32 (data is stored as bfloat16)
        target_feature = target_feature.float()
        
        # Subsample
        if self.subsample_seq is not None and target_feature.shape[0] > self.subsample_seq:
            indices = torch.randperm(target_feature.shape[0])[:self.subsample_seq]
            indices = indices.sort().values
            target_feature = target_feature[indices]
        
        # Load cached features
        cache_features = []
        step_distances = []
        for distance, step in cache_info:
            cache_path = os.path.join(
                self.data_dir, prompt_dir, branch, f'step_{step:02d}.pt'
            )
            cache_feature = torch.load(cache_path, map_location='cpu')
            # Cast to float32
            cache_feature = cache_feature.float()
            
            if self.subsample_seq is not None and cache_feature.shape[0] > self.subsample_seq:
                cache_feature = cache_feature[indices]  # Use the same indices
            
            cache_features.append(cache_feature)
            step_distances.append(distance)
        
        return {
            'target_feature': target_feature,
            'cache_features': cache_features,
            'step_distances': step_distances,
            'interval': interval,
            'target_step': target_step,
            'branch': branch,
            'prompt_dir': prompt_dir,
        }


def create_dataloader(
    data_dir: str,
    batch_size: int = 4,
    intervals: List[int] = [5, 7, 10],
    num_workers: int = 4,
    shuffle: bool = True,
    prompt_indices: Optional[List[int]] = None,
    subsample_seq: Optional[int] = None,
    random_interval: bool = False,
) -> DataLoader:
    """
    Convenience helper that builds a DataLoader
    
    Args:
        random_interval: Whether to resample interval on every draw
                        True: fewer samples, more interval diversity (recommended for training)
                        False: create independent samples per interval (original behaviour)
    """
    
    dataset = MemoryEfficientCacheDataset(
        data_dir=data_dir,
        intervals=intervals,
        prompt_indices=prompt_indices,
        subsample_seq=subsample_seq,
        random_interval=random_interval,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=False,  # Disable pin_memory to save host memory
    )
    
    return dataloader


if __name__ == "__main__":
    # Test the dataset
    data_dir = "/root/autodl-tmp/linca_data/data/cache_data"
    
    print("Testing MemoryEfficientCacheDataset...")
    dataset = MemoryEfficientCacheDataset(
        data_dir=data_dir,
        intervals=[5, 7, 10],
        prompt_indices=list(range(10)),  # use the first 10 prompts for a smoke test
        subsample_seq=1000,  # subsample to speed up the smoke test
    )
    
    print(f"\nTotal samples: {len(dataset)}")
    
    # Test a single sample
    sample = dataset[0]
    print(f"\nSample 0:")
    print(f"  target_feature shape: {sample['target_feature'].shape}")
    print(f"  num cache features: {len(sample['cache_features'])}")
    print(f"  step_distances: {sample['step_distances']}")
    print(f"  interval: {sample['interval']}")
    print(f"  target_step: {sample['target_step']}")
    print(f"  branch: {sample['branch']}")
    
    # Test the DataLoader
    print("\nTesting DataLoader...")
    dataloader = create_dataloader(
        data_dir=data_dir,
        batch_size=2,
        intervals=[5, 10],
        num_workers=0,
        prompt_indices=list(range(5)),
        subsample_seq=500,
    )
    
    batch = next(iter(dataloader))
    print(f"\nBatch:")
    print(f"  target_features: {batch['target_features'].shape}")
    print(f"  cache_features: {batch['cache_features'].shape}")
    print(f"  cache_mask: {batch['cache_mask'].shape}")
    print(f"  step_distances: {batch['step_distances']}")
    print(f"  intervals: {batch['intervals']}")
    
    print("\n✓ Dataset tests passed!")
