"""
数据划分: 22 验证 + 11 展示 + 180 训练
- 验证: 11 task_type，每个 cn+en 各 1 条，固定 seed=42
- 展示: 11 task_type 各 1 条，可混 cn/en
- 训练: 202 - 22 = 180
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
    """从 index.json 和 metadata 加载 202 条 cache 样本的元信息"""
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
    构建验证、展示、训练划分
    Returns:
        val_sample_indices: 22 个 sample_idx
        display_sample_indices: 11 个 sample_idx，每类 1 个
        train_sample_indices: 180 个 sample_idx
    """
    samples = load_cache_metadata(cache_data_dir)
    if len(samples) != 202:
        print(f"Warning: expected 202 samples, got {len(samples)}")

    # 按 task_type + language 分组
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
        # 验证: 每个 task_type 选 1 cn + 1 en
        if cn_list:
            val_indices.append(val_rng.choice(cn_list))
        if en_list:
            val_indices.append(val_rng.choice(en_list))
        # 展示: 每类 1 个，优先 cn 若有则选 cn 否则 en
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
    """保存划分到 splits.json"""
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
    """从 splits.json 加载划分"""
    path = os.path.join(output_dir, 'splits.json')
    with open(path, 'r', encoding='utf-8') as f:
        d = json.load(f)
    return d['val_sample_indices'], d['display_sample_indices'], d['train_sample_indices']


if __name__ == "__main__":
    val, disp, train = build_splits("./cache_data/qwen_edit")
    print(f"val: {len(val)}, display: {len(disp)}, train: {len(train)}")
