"""
Dataset for training the invertible decomposition network (qwen_edit version)

Data format:
    cache_data/qwen_edit_202/
    ├── sample_0000/
    │   ├── cond/
    │   │   ├── step_00.pt  # [4096, 3072]
    │   │   ├── step_01.pt
    │   │   └── ...
    │   └── uncond/
    │       ├── step_00.pt
    │       └── ...
    └── sample_0001/
        └── ...

与 qwen_image 的差异: prompt_ -> sample_, seq_len 6889 -> 4096
"""

import os
import torch
import json
import random
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional
import numpy as np


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function
    qwen_edit: seq_len=4096
    """
    target_features = torch.stack([item['target_feature'] for item in batch])
    max_cache = max(len(item['cache_features']) for item in batch)
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
        'target_features': target_features,
        'cache_features': cache_features,
        'cache_mask': cache_mask,
        'step_distances': step_distances,
        'intervals': intervals,
        'target_steps': target_steps,
        'branches': [item['branch'] for item in batch],
        'sample_dirs': [item['sample_dir'] for item in batch],
    }


class MemoryEfficientCacheDataset(Dataset):
    """
    内存高效版本的数据集 (qwen_edit: sample_XXXX, seq_len=4096)
    支持 interval 随机采样，cond/uncond 都做训练
    """

    def __init__(
        self,
        data_dir: str,
        intervals: List[int] = [5, 7, 10],
        first_enhance: int = 3,
        num_steps: int = 50,
        branches: List[str] = ['cond', 'uncond'],
        sample_indices: Optional[List[int]] = None,
        max_cache_points: int = 3,
        subsample_seq: Optional[int] = None,
        random_interval: bool = False,
    ):
        self.data_dir = data_dir
        self.intervals = intervals
        self.first_enhance = first_enhance
        self.num_steps = num_steps
        self.branches = branches
        self.max_cache_points = max_cache_points
        self.subsample_seq = subsample_seq
        self.random_interval = random_interval

        # 扫描所有 sample 目录 (qwen_edit 用 sample_ 前缀)
        all_samples = sorted([
            d for d in os.listdir(data_dir)
            if d.startswith('sample_') and os.path.isdir(os.path.join(data_dir, d))
        ])

        if sample_indices is not None:
            self.sample_dirs = [f'sample_{i:04d}' for i in sample_indices if f'sample_{i:04d}' in all_samples]
        else:
            self.sample_dirs = all_samples

        self.samples = self._build_sample_indices()
        mode_str = "random interval" if random_interval else "fixed interval"
        print(f"Dataset: {len(self.sample_dirs)} samples, {len(self.samples)} training samples ({mode_str})")
        print(f"Intervals: {self.intervals}, Branches: {self.branches}")

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
        samples = []
        if self.random_interval:
            min_interval = min(self.intervals)
            active_steps_min = self._get_active_steps(min_interval)
            for sample_idx, sample_dir in enumerate(self.sample_dirs):
                for branch in self.branches:
                    for target_step in range(self.num_steps):
                        if target_step in active_steps_min:
                            continue
                        samples.append((sample_idx, branch, target_step))
        else:
            for sample_idx, sample_dir in enumerate(self.sample_dirs):
                for branch in self.branches:
                    for interval in self.intervals:
                        active_steps = self._get_active_steps(interval)
                        for target_step in range(self.num_steps):
                            if target_step in active_steps:
                                continue
                            cache_info = self._get_cache_info(target_step, active_steps)
                            if len(cache_info) == 0:
                                continue
                            samples.append((sample_idx, branch, interval, target_step, cache_info))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        if self.random_interval:
            sample_idx, branch, target_step = self.samples[idx]
            interval = random.choice(self.intervals)
            active_steps = self._get_active_steps(interval)
            attempts = 0
            while target_step in active_steps and attempts < len(self.intervals):
                interval = random.choice(self.intervals)
                active_steps = self._get_active_steps(interval)
                attempts += 1
            if target_step in active_steps:
                interval = min(self.intervals)
                active_steps = self._get_active_steps(interval)
            cache_info = self._get_cache_info(target_step, active_steps)
        else:
            sample_idx, branch, interval, target_step, cache_info = self.samples[idx]

        sample_dir = self.sample_dirs[sample_idx]

        target_path = os.path.join(
            self.data_dir, sample_dir, branch, f'step_{target_step:02d}.pt'
        )
        target_feature = torch.load(target_path, map_location='cpu')
        target_feature = target_feature.float()

        if self.subsample_seq is not None and target_feature.shape[0] > self.subsample_seq:
            indices = torch.randperm(target_feature.shape[0])[:self.subsample_seq]
            indices = indices.sort().values
            target_feature = target_feature[indices]

        cache_features = []
        step_distances = []
        for distance, step in cache_info:
            cache_path = os.path.join(
                self.data_dir, sample_dir, branch, f'step_{step:02d}.pt'
            )
            cache_feature = torch.load(cache_path, map_location='cpu')
            cache_feature = cache_feature.float()
            if self.subsample_seq is not None and cache_feature.shape[0] > self.subsample_seq:
                cache_feature = cache_feature[indices]
            cache_features.append(cache_feature)
            step_distances.append(distance)

        return {
            'target_feature': target_feature,
            'cache_features': cache_features,
            'step_distances': step_distances,
            'interval': interval,
            'target_step': target_step,
            'branch': branch,
            'sample_dir': sample_dir,
        }


def create_dataloader(
    data_dir: str,
    batch_size: int = 4,
    intervals: List[int] = [5, 7, 10],
    num_workers: int = 4,
    shuffle: bool = True,
    sample_indices: Optional[List[int]] = None,
    subsample_seq: Optional[int] = None,
    random_interval: bool = False,
) -> DataLoader:
    dataset = MemoryEfficientCacheDataset(
        data_dir=data_dir,
        intervals=intervals,
        sample_indices=sample_indices,
        subsample_seq=subsample_seq,
        random_interval=random_interval,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=False,
    )
