"""
Data split: 180 train + 22 val + 11 display.
- val: one Chinese and one English entry per task type, seed fixed at 42
- display: one entry per task type, either language
- train: the remaining 180 of the 202 samples
"""

import os
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple


TASK_TYPES = [
    'background_change', 'color_alter', 'material_alter', 'motion_change',
    'ps_human', 'style_change', 'subject-add', 'subject-remove',
    'subject-replace', 'text_change', 'tone_transfer',
]


def load_cache_metadata(cache_data_dir: str) -> List[Dict]:
    """Load metadata for the 202 cached samples from index.json."""
    index_path = os.path.join(cache_data_dir, 'index.json')
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"index.json not found: {index_path}")

    with open(index_path, 'r', encoding='utf-8') as f:
        index_data = json.load(f)

    samples = []
    for s in index_data['samples']:
        sample_idx = s['sample_idx']
        sample_dir = s['dir']
        meta_path = os.path.join(cache_data_dir, sample_dir, 'metadata.json')
        if os.path.exists(meta_path):
            with open(meta_path, 'r', encoding='utf-8') as mf:
                meta = json.load(mf)
            samples.append({
                'sample_idx': sample_idx,
                'dir': sample_dir,
                'task_type': meta.get('task_type', ''),
                'language': meta.get('language', ''),
            })
        else:
            samples.append({
                'sample_idx': sample_idx,
                'dir': sample_dir,
                'task_type': '',
                'language': '',
            })
    return samples


def build_splits(
    cache_data_dir: str,
    val_seed: int = 42,
    display_seed: int = 43,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Build the val, display and train splits.
    Returns:
        val_sample_indices: 22 sample indices
        display_sample_indices: 11 sample indices, one per task type
        train_sample_indices: 180 sample indices
    """
    samples = load_cache_metadata(cache_data_dir)
    if len(samples) != 202:
        print(f"Warning: expected 202 samples, got {len(samples)}")

    # Group by task type and language
    by_tt_lang = {}
    for s in samples:
        tt = s['task_type']
        lang = s['language']
        if tt not in by_tt_lang:
            by_tt_lang[tt] = {'cn': [], 'en': []}
        if lang in ('cn', 'en'):
            by_tt_lang[tt][lang].append(s['sample_idx'])

    val_rng = random.Random(val_seed)
    display_rng = random.Random(display_seed)

    val_indices = []
    display_indices = []

    for tt in TASK_TYPES:
        if tt not in by_tt_lang:
            continue
        cn_list = by_tt_lang[tt]['cn']
        en_list = by_tt_lang[tt]['en']
        # val: one Chinese and one English entry per task type
        if cn_list:
            val_indices.append(val_rng.choice(cn_list))
        if en_list:
            val_indices.append(val_rng.choice(en_list))
        # display: one per task type, preferring Chinese and falling back to English
        if cn_list:
            display_indices.append(display_rng.choice(cn_list))
        elif en_list:
            display_indices.append(display_rng.choice(en_list))

    val_set = set(val_indices)
    train_indices = [s['sample_idx'] for s in samples if s['sample_idx'] not in val_set]

    return val_indices, display_indices, train_indices


def save_splits(
    output_dir: str,
    val_indices: List[int],
    display_indices: List[int],
    train_indices: List[int],
):
    """Write the split to splits.json."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'splits.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({
            'val_sample_indices': val_indices,
            'display_sample_indices': display_indices,
            'train_sample_indices': train_indices,
    }, f, indent=2)
    print(f"Saved splits to {path}")


def load_splits(output_dir: str) -> Tuple[List[int], List[int], List[int]]:
    """Load the split from splits.json."""
    path = os.path.join(output_dir, 'splits.json')
    with open(path, 'r', encoding='utf-8') as f:
        d = json.load(f)
    return d['val_sample_indices'], d['display_sample_indices'], d['train_sample_indices']


if __name__ == "__main__":
    val, disp, train = build_splits("./cache_data/qwen_edit")
    print(f"val: {len(val)}, display: {len(disp)}, train: {len(train)}")
