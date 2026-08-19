"""
Dataset for training the invertible decomposition network

Data format:
    cache_data/
    ├── prompt_0000/
    │   ├── step_00.pt  # [seq_len, 3072]
    │   ├── step_01.pt
    │   └── ...
    └── prompt_0001/
        └── ...

Training sample construction:
    给定interval,构造预测样本:
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
        prompt_indices: Optional[List[int]] = None,
        max_cache_points: int = 3,  # maximum number of history points
    ):
        """
        Args:
            data_dir: path to the data directory (cache_data/)
            intervals: list of intervals to train on
            first_enhance: number of leading full-compute steps
            num_steps: total number of steps
            prompt_indices: 要使用的prompt索引,None表示全部
            max_cache_points: max history points used at prediction time
        """
        self.data_dir = data_dir
        self.intervals = intervals
        self.first_enhance = first_enhance
        self.num_steps = num_steps
        self.max_cache_points = max_cache_points

        # Scan all prompt directories
        all_prompts = sorted([
            d for d in os.listdir(data_dir)
            if d.startswith('prompt_') and os.path.isdir(os.path.join(data_dir, d))
        ])

        if prompt_indices is not None:
            # Build index -> dir_name mapping (3-digit or 4-digit names)
            index_to_dir = {}
            for d in all_prompts:
                try:
                    idx = int(d.split('_')[1])
                    index_to_dir[idx] = d
                except (IndexError, ValueError):
                    continue
            self.prompt_dirs = [index_to_dir[i] for i in prompt_indices if i in index_to_dir]
        else:
            self.prompt_dirs = all_prompts

        # Precompute every training sample
        self.samples = self._build_samples()

        print(f"Dataset: {len(self.prompt_dirs)} prompts, {len(self.samples)} samples")
        print(f"Intervals: {self.intervals}")

    def _get_active_steps(self, interval: int) -> List[int]:
        """
        获取给定interval下的激活步(full计算的步)

        逻辑与FreqCa一致:
        - Leading first_enhance steps: 0, 1, 2
        - Then every interval: first_enhance, first_enhance+interval, ...
        """
        active = list(range(self.first_enhance))  # [0, 1, 2]

        # 从first_enhance开始,每interval步
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

        # 按距离排序(最近的在前)
        available.sort(key=lambda s: target_step - s)

        # Return a list of (distance, step) pairs
        return [(target_step - s, s) for s in available[:self.max_cache_points]]

    def _build_samples(self) -> List[Dict]:
        """Build every training sample"""
        samples = []

        for prompt_dir in self.prompt_dirs:
            for interval in self.intervals:
                active_steps = self._get_active_steps(interval)

                # 对于每个非激活步,创建一个样本
                for target_step in range(self.num_steps):
                    if target_step in active_steps:
                        continue  # skip full-compute steps

                    cache_info = self._get_cache_info(target_step, active_steps)
                    if len(cache_info) == 0:
                        continue  # no usable cache

                    samples.append({
                        'prompt_dir': prompt_dir,
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
        target_step = sample_info['target_step']
        cache_info = sample_info['cache_info']
        interval = sample_info['interval']

        # Load the target feature
        target_path = os.path.join(
            self.data_dir, prompt_dir, f'step_{target_step:02d}.pt'
        )
        data = torch.load(target_path, map_location='cpu', weights_only=True)
        # Extract the feature field from a dict
        if isinstance(data, dict) and 'feature' in data:
            target_feature = data['feature']
        else:
            target_feature = data
        # Cast to float32 (data is stored as bfloat16)
        target_feature = target_feature.float()
        # Drop the batch dim [1, N, D] -> [N, D]
        if target_feature.dim() == 3 and target_feature.shape[0] == 1:
            target_feature = target_feature.squeeze(0)

        # Load cached features
        cache_features = []
        step_distances = []
        for distance, step in cache_info:
            cache_path = os.path.join(
                self.data_dir, prompt_dir, f'step_{step:02d}.pt'
            )
            data = torch.load(cache_path, map_location='cpu', weights_only=True)
            # Extract the feature field from a dict
            if isinstance(data, dict) and 'feature' in data:
                cache_feature = data['feature']
            else:
                cache_feature = data
            # Cast to float32
            cache_feature = cache_feature.float()
            # Drop the batch dim [1, N, D] -> [N, D]
            if cache_feature.dim() == 3 and cache_feature.shape[0] == 1:
                cache_feature = cache_feature.squeeze(0)
            cache_features.append(cache_feature)
            step_distances.append(distance)

        return {
            'target_feature': target_feature,  # [seq_len, 3072]
            'cache_features': cache_features,  # List of [seq_len, 3072]
            'step_distances': step_distances,  # List of int
            'interval': interval,
            'target_step': target_step,
            'prompt_dir': prompt_dir,
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function

    由于seq_len固定(6889),可以直接stack
    但cache_features数量可能不同,需要特殊处理
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
        'prompt_dirs': [item['prompt_dir'] for item in batch],
    }


class MemoryEfficientCacheDataset(Dataset):
    """
    Memory-efficient dataset (supports random interval sampling)

    不预加载所有数据,而是按需加载
    Speed up via a precomputed index

    Supports two modes:
    - random_interval=False: pre-build samples per interval (original behaviour)
    - random_interval=True: 每次取样时随机选择interval (减少样本量,增加多样性)
    """

    def __init__(
        self,
        data_dir: str,
        intervals: List[int] = [5, 7, 10],
        first_enhance: int = 3,
        num_steps: int = 50,
        prompt_indices: Optional[List[int]] = None,
        max_cache_points: int = 3,
        subsample_seq: Optional[int] = None,  # subsample seq_len to reduce memory
        random_interval: bool = False,  # whether to sample interval at random
    ):
        self.data_dir = data_dir
        self.intervals = intervals
        self.first_enhance = first_enhance
        self.num_steps = num_steps
        self.max_cache_points = max_cache_points
        self.subsample_seq = subsample_seq
        self.random_interval = random_interval

        # Scan all prompt directories
        all_prompts = sorted([
            d for d in os.listdir(data_dir)
            if d.startswith('prompt_') and os.path.isdir(os.path.join(data_dir, d))
        ])

        if prompt_indices is not None:
            # Build index -> dir_name mapping (3-digit or 4-digit names)
            index_to_dir = {}
            for d in all_prompts:
                try:
                    idx = int(d.split('_')[1])
                    index_to_dir[idx] = d
                except (IndexError, ValueError):
                    continue
            self.prompt_dirs = [index_to_dir[i] for i in prompt_indices if i in index_to_dir]
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
            # Random-interval mode: 只为每个(prompt, target_step)创建一个样本
            # interval is sampled randomly inside __getitem__
            # Use the smallest interval to decide which target_steps are valid
            min_interval = min(self.intervals)
            active_steps_min = self._get_active_steps(min_interval)

            for prompt_idx, prompt_dir in enumerate(self.prompt_dirs):
                for target_step in range(self.num_steps):
                    if target_step in active_steps_min:
                        continue

                    # 存储时不包含interval和cache_info,这些在__getitem__时动态计算
                    samples.append((
                        prompt_idx,
                        target_step,
                    ))
        else:
            # Original mode: create samples for every interval
            for prompt_idx, prompt_dir in enumerate(self.prompt_dirs):
                for interval in self.intervals:
                    active_steps = self._get_active_steps(interval)

                    for target_step in range(self.num_steps):
                        if target_step in active_steps:
                            continue

                        cache_info = self._get_cache_info(target_step, active_steps)
                        if len(cache_info) == 0:
                            continue

                        # 只存储索引,不存储数据
                        samples.append((
                            prompt_idx,
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
            prompt_idx, target_step = self.samples[idx]

            # Sample interval uniformly at random
            interval = random.choice(self.intervals)

            # Compute active_steps and cache_info on the fly
            active_steps = self._get_active_steps(interval)

            # 如果target_step在当前interval下是active step,随机选择另一个interval
            attempts = 0
            while target_step in active_steps and attempts < len(self.intervals):
                interval = random.choice(self.intervals)
                active_steps = self._get_active_steps(interval)
                attempts += 1

            # 如果所有interval都不行,使用最小interval的fallback
            if target_step in active_steps:
                # 这种情况理论上不应该发生,但作为保护
                interval = min(self.intervals)
                active_steps = self._get_active_steps(interval)

            cache_info = self._get_cache_info(target_step, active_steps)
        else:
            # Original mode
            prompt_idx, interval, target_step, cache_info = self.samples[idx]

        prompt_dir = self.prompt_dirs[prompt_idx]

        # Load the target feature
        target_path = os.path.join(
            self.data_dir, prompt_dir, f'step_{target_step:02d}.pt'
        )
        data = torch.load(target_path, map_location='cpu', weights_only=True)
        # Extract the feature field from a dict
        if isinstance(data, dict) and 'feature' in data:
            target_feature = data['feature']
        else:
            target_feature = data
        # Cast to float32 (data is stored as bfloat16)
        target_feature = target_feature.float()
        # Drop the batch dim [1, N, D] -> [N, D]
        if target_feature.dim() == 3 and target_feature.shape[0] == 1:
            target_feature = target_feature.squeeze(0)

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
                self.data_dir, prompt_dir, f'step_{step:02d}.pt'
            )
            data = torch.load(cache_path, map_location='cpu', weights_only=True)
            # Extract the feature field from a dict
            if isinstance(data, dict) and 'feature' in data:
                cache_feature = data['feature']
            else:
                cache_feature = data
            # Cast to float32
            cache_feature = cache_feature.float()
            # Drop the batch dim [1, N, D] -> [N, D]
            if cache_feature.dim() == 3 and cache_feature.shape[0] == 1:
                cache_feature = cache_feature.squeeze(0)

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
                        True: 减少样本量,增加interval多样性 (推荐用于训练)
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
