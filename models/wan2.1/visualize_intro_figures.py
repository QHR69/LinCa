#!/usr/bin/env python3
"""
Introduction-figure script — flux_v4_multi_stage

- Panel 1: dynamics across denoising — 1-1 trajectory, 1-2 multi-stage, 1-3 step deltas
- Panel 3: 3 raw vs 3 decomposed plots (depends on split_dims and the checkpoint)

Usage:
    python visualize_intro_figures.py \
        --cache_dir ./cache_data/flux \
        --checkpoint ./checkpoints/flux/best_predictor.pt \
        --split_dims 2304,384,384 \
        --output_dir ./intro_figures
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional

try:
    from sklearn.decomposition import PCA
except ImportError:
    def pca_fit_transform(X, n_components=2):
        Xc = X - X.mean(axis=0)
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        var_ratio = (S[:n_components]**2) / (S**2).sum()
        return Xc @ Vt.T[:, :n_components], var_ratio
    PCA = None

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'src'))
from flux.modules.invertible_net import LearnedDecompositionPredictor

PALETTE = {
    'primary': '#2563eb', 'primary_light': '#60a5fa', 'primary_dark': '#1d4ed8',
    'secondary': '#0ea5e9', 'accent': '#38bdf8', 'warm': '#f59e0b', 'warm2': '#f97316',
    'gray': '#64748b', 'gray_light': '#94a3b8',
    'stage1': '#1e40af', 'stage2': '#3b82f6', 'stage3': '#0ea5e9', 'stage4': '#06b6d4', 'stage5': '#22d3ee',
    'raw': '#64748b', 'decomposed': '#2563eb',
}
import matplotlib.colors as mpl_colors
_colors_traj = ['#93c5fd', '#60a5fa', '#3b82f6', '#1d4ed8', '#1e40af']
CMAP_STEP = mpl_colors.LinearSegmentedColormap.from_list('traj_blue', _colors_traj, N=256)
_colors_aspect3 = ['#FFEB3B', '#FFC107', '#8BC34A', '#2196F3', '#1565C0', '#0D47A1']
CMAP_STEP_ASPECT3 = mpl_colors.LinearSegmentedColormap.from_list('traj_ygb', _colors_aspect3, N=256)
FONT_SIZE = 11
TICK_SIZE = 10
FIG_DPI = 300
OUTPUT_SVG = True
OUTPUT_PNG = True
COLOR_Z0, COLOR_Z1, COLOR_Z2 = '#1e40af', '#f59e0b', '#059669'


def _savefig(fig, out_path_base: str):
    if OUTPUT_SVG:
        fig.savefig(out_path_base + '.svg', dpi=FIG_DPI, bbox_inches='tight', format='svg')
    if OUTPUT_PNG:
        fig.savefig(out_path_base + '.png', dpi=150, bbox_inches='tight', format='png')


def load_prompt_cache(cache_dir: str, prompt_idx: int) -> torch.Tensor:
    """Load a flux/collected_features cache (prompt_xxx/step_xx.pt, dict layout)"""
    prompt_dir = os.path.join(cache_dir, f'prompt_{prompt_idx:03d}')
    if not os.path.isdir(prompt_dir):
        raise FileNotFoundError(f"Cache not found: {prompt_dir}")
    feats = []
    for step in range(50):
        p = os.path.join(prompt_dir, f'step_{step:02d}.pt')
        if not os.path.exists(p):
            raise FileNotFoundError(f"Step file not found: {p}")
        data = torch.load(p, map_location='cpu', weights_only=True)
        # dict layout: {'feature': Tensor[1, 4096, 3072], 'step': int, ...}
        x = data['feature'].float().squeeze(0)  # [4096, 3072]
        feats.append(x)
    return torch.stack(feats, dim=0)  # [50, 4096, 3072]


def mean_over_tokens(feats: torch.Tensor) -> np.ndarray:
    """Mean over the token axis: [50, 4096, 3072] -> [50, 3072]"""
    return feats.mean(dim=1).numpy()


def split_three(x: np.ndarray, split_dims_list: List[int]):
    """Split [..., dim] into three parts given split_dims_list"""
    d0, d1, d2 = split_dims_list
    return x[..., :d0], x[..., d0:d0+d1], x[..., d0+d1:d0+d1+d2]


def _do_pca(X: np.ndarray, n_components: int = 2):
    if X.shape[0] < n_components:
        raise ValueError(f"Too few samples ({X.shape[0]}) for PCA with n_components={n_components}")
    if PCA is not None:
        pca = PCA(n_components=n_components)
        coords = pca.fit_transform(X)
        return coords, pca.explained_variance_ratio_
    coords, var_ratio = pca_fit_transform(X, n_components)
    return coords, var_ratio


# ========== Panel 1: dynamics ==========

def fig_1_1_trajectory_pca(feats_mean: np.ndarray, out_path: str, prompt_text: str = ""):
    coords, var_ratio = _do_pca(feats_mean, 2)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    steps = np.arange(50)
    norm = mpl_colors.Normalize(vmin=0, vmax=49)
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=steps, cmap=CMAP_STEP, norm=norm, s=28, alpha=0.92, edgecolors='white', linewidths=0.3)
    ax.plot(coords[:, 0], coords[:, 1], color=PALETTE['gray'], alpha=0.5, linewidth=1.2)
    color_t0, color_t49 = CMAP_STEP(norm(0)), CMAP_STEP(norm(49))
    ax.scatter(coords[0, 0], coords[0, 1], c=[color_t0], s=100, marker='o', edgecolors='white', linewidths=2, zorder=5, label='t=0')
    ax.scatter(coords[-1, 0], coords[-1, 1], c=[color_t49], s=100, marker='s', edgecolors='white', linewidths=2, zorder=5, label='t=49')
    cbar = plt.colorbar(sc, ax=ax, shrink=0.75)
    cbar.set_label('Denoising step $t$', fontsize=FONT_SIZE)
    ax.set_xlabel(f'PC1 ({var_ratio[0]*100:.1f}% var.)', fontsize=FONT_SIZE)
    ax.set_ylabel(f'PC2 ({var_ratio[1]*100:.1f}% var.)', fontsize=FONT_SIZE)
    ax.set_title('Latent Trajectory (Full 50 Steps)', fontsize=FONT_SIZE)
    if prompt_text:
        txt = prompt_text[:72] + '...' if len(prompt_text) > 72 else prompt_text
        ax.text(0.5, 0.98, 'Prompt: ' + txt, transform=ax.transAxes, ha='center', va='top', fontsize=8, color=PALETTE['gray'])
    ax.legend(loc='lower left', fontsize=9)
    ax.tick_params(axis='both', labelsize=TICK_SIZE)
    ax.set_aspect('equal', adjustable='datalim')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    _savefig(fig, out_path)
    np.savez(out_path + '_data.npz', steps=steps, pc1=coords[:, 0], pc2=coords[:, 1], var_ratio=var_ratio)
    plt.close()


def fig_1_2_stage_trajectories(feats_mean: np.ndarray, out_path: str, n_stages: int, prompt_text: str = ""):
    coords, var_ratio = _do_pca(feats_mean, 2)
    stage_colors = [PALETTE[f'stage{i+1}'] for i in range(min(n_stages, 5))]
    boundaries = np.linspace(0, 50, n_stages + 1, dtype=int)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for i in range(n_stages):
        s, e = boundaries[i], boundaries[i + 1]
        idx = np.arange(s, e)
        if len(idx) < 2:
            continue
        ax.plot(coords[idx, 0], coords[idx, 1], '-', color=stage_colors[i], linewidth=2.2, label=f'Stage {i+1} (t={s}-{e-1})')
        ax.scatter(coords[idx, 0], coords[idx, 1], c=stage_colors[i], s=22, alpha=0.9, edgecolors='white', linewidths=0.3)
    ax.set_xlabel(f'PC1 ({var_ratio[0]*100:.1f}% var.)', fontsize=FONT_SIZE)
    ax.set_ylabel(f'PC2 ({var_ratio[1]*100:.1f}% var.)', fontsize=FONT_SIZE)
    ax.set_title(f'Trajectory by Stage ({n_stages} stages)', fontsize=FONT_SIZE)
    if prompt_text:
        txt = prompt_text[:72] + '...' if len(prompt_text) > 72 else prompt_text
        fig.text(0.5, 1.02, 'Prompt: ' + txt, transform=fig.transFigure, ha='center', va='bottom', fontsize=9, color=PALETTE['gray'])
    ax.legend(loc='best', fontsize=9)
    ax.tick_params(axis='both', labelsize=TICK_SIZE)
    ax.set_aspect('equal', adjustable='datalim')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    _savefig(fig, out_path)
    np.savez(out_path + '_data.npz', pc1=coords[:, 0], pc2=coords[:, 1],
             var_ratio=var_ratio, boundaries=boundaries)
    plt.close()


def fig_1_3_step_change_magnitude(feats_mean: np.ndarray, out_path: str, prompt_text: str = ""):
    diffs = np.linalg.norm(np.diff(feats_mean, axis=0), axis=1)
    steps = np.arange(1, 50)
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.fill_between(steps, diffs, alpha=0.25, color=PALETTE['primary'])
    ax.plot(steps, diffs, 'o-', color=PALETTE['primary'], markersize=5, linewidth=1.5)
    ax.axvline(x=24.5, color=PALETTE['gray'], linestyle='--', alpha=0.8, linewidth=1.2, label='Early/Late boundary')
    ax.set_xlabel('Denoising step $t$', fontsize=FONT_SIZE)
    ax.set_ylabel(r'$\|\mathbf{x}_t - \mathbf{x}_{t-1}\|_2$', fontsize=FONT_SIZE)
    ax.set_title('Step-wise Change Magnitude', fontsize=FONT_SIZE)
    if prompt_text:
        txt = prompt_text[:72] + '...' if len(prompt_text) > 72 else prompt_text
        fig.text(0.5, 1.02, 'Prompt: ' + txt, transform=fig.transFigure, ha='center', va='bottom', fontsize=9, color=PALETTE['gray'])
    ax.legend(loc='best', fontsize=9)
    ax.tick_params(axis='both', labelsize=TICK_SIZE)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    _savefig(fig, out_path)
    np.savez(out_path + '_data.npz', steps=steps, diffs=diffs)
    plt.close()


# ========== Panel 3: decomposition ==========

def _similarity_per_step(z: np.ndarray) -> np.ndarray:
    if z.shape[0] < 2 or z.shape[-1] == 0:
        return np.array([])
    a, b = z[:-1], z[1:]
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return np.sum(an * bn, axis=1)


def _first_order_error_per_step(z: np.ndarray) -> np.ndarray:
    if z.shape[0] < 3 or z.shape[-1] == 0:
        return np.array([])
    pred = z[1:-1] + (z[1:-1] - z[:-2])
    return np.linalg.norm(pred - z[2:], axis=1)


def _aspect3_ax_style(ax, spine_lw: int, with_grid: bool = True):
    ax.set_xlabel('Principal Component 1', fontsize=FONT_SIZE)
    ax.set_ylabel('Principal Component 2', fontsize=FONT_SIZE)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(spine_lw)
    if with_grid:
        ax.grid(True, which='major', color='#888888', linestyle='-', linewidth=0.9, alpha=1.0)
        ax.set_axisbelow(True)
    ax.tick_params(axis='both', which='major', length=4, labelsize=TICK_SIZE)
    ax.set_aspect('equal', adjustable='datalim')


def fig_similarity_lines(z0, z1, z2, out_base: str, prompt_text: str = "", title_suffix: str = ""):
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    s0, s1, s2 = _similarity_per_step(z0), _similarity_per_step(z1), _similarity_per_step(z2)
    steps = np.arange(1, 50)
    if len(s0): ax.plot(steps, s0, '-', color=COLOR_Z0, label=f'z0 (0-order, dim={z0.shape[-1]})', linewidth=1.5)
    if len(s1): ax.plot(steps, s1, '-', color=COLOR_Z1, label=f'z1 (1-order, dim={z1.shape[-1]})', linewidth=1.5)
    if len(s2): ax.plot(steps, s2, '-', color=COLOR_Z2, label=f'z2 (2-order, dim={z2.shape[-1]})', linewidth=1.5)
    ax.set_xlabel('Denoising step $t$', fontsize=FONT_SIZE)
    ax.set_ylabel('Cosine similarity (adjacent steps)', fontsize=FONT_SIZE)
    ax.set_title('0-order Similarity' + (' ' + title_suffix if title_suffix else ''), fontsize=FONT_SIZE)
    if prompt_text:
        txt = prompt_text[:72] + '...' if len(prompt_text) > 72 else prompt_text
        fig.text(0.5, 1.02, 'Prompt: ' + txt, transform=fig.transFigure, ha='center', va='bottom', fontsize=9, color=PALETTE['gray'])
    ax.legend(loc='best', fontsize=9)
    ax.tick_params(axis='both', labelsize=TICK_SIZE)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    _savefig(fig, out_base)
    np.savez(out_base + '_data.npz', steps=steps, s0=s0, s1=s1, s2=s2)
    plt.close()


def fig_continuity_pca(z0, z1, z2, out_base: str, prompt_text: str = "", title_suffix: str = "", is_raw: bool = True):
    subplot_titles = ('Dim1 (z0)', 'Dim2 (z1)', 'Dim3 (z2)')
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.8))
    spine_lw = 4
    norm_step = mpl_colors.Normalize(vmin=0, vmax=49)
    _save_data = {}
    for zi, (ax, z, title, c) in enumerate([(axes[0], z0, subplot_titles[0], COLOR_Z0),
                             (axes[1], z1, subplot_titles[1], COLOR_Z1),
                             (axes[2], z2, subplot_titles[2], COLOR_Z2)]):
        if z.shape[0] < 2 or z.shape[-1] < 2:
            ax.set_title(f'{title} (dim={z.shape[-1] if z.ndim>1 else 0}, skipped)', fontsize=FONT_SIZE)
            _aspect3_ax_style(ax, spine_lw)
            ax.set_xticks([]); ax.set_yticks([])
            continue
        if PCA is not None:
            pca = PCA(n_components=2).fit(z)
            proj = pca.transform(z)
        else:
            proj, _ = pca_fit_transform(z, 2)
        steps = np.arange(z.shape[0])
        _save_data[f'z{zi}_pc1'] = proj[:, 0]
        _save_data[f'z{zi}_pc2'] = proj[:, 1]
        _save_data[f'z{zi}_steps'] = steps
        ax.plot(proj[:, 0], proj[:, 1], color='#b0b0b0', linewidth=1.2, zorder=0)
        sc = ax.scatter(proj[:, 0], proj[:, 1], c=steps, cmap=CMAP_STEP_ASPECT3, norm=norm_step, s=32, alpha=0.95, edgecolors='white', linewidths=0.4, zorder=1)
        cbar = plt.colorbar(sc, ax=ax, shrink=0.75)
        cbar.set_label('step', fontsize=FONT_SIZE)
        ax.set_title(f'{title} (dim={z.shape[-1]})', fontsize=FONT_SIZE)
        _aspect3_ax_style(ax, spine_lw)
        ax.locator_params(axis='both', nbins=6)
        ax.tick_params(axis='both', labelbottom=False, labelleft=False)
    if prompt_text:
        txt = prompt_text[:72] + '...' if len(prompt_text) > 72 else prompt_text
        fig.text(0.5, 1.02, 'Prompt: ' + txt, transform=fig.transFigure, ha='center', va='bottom', fontsize=9, color=PALETTE['gray'])
    fig.suptitle('2-order Continuity (PCA trajectory)' + (' ' + title_suffix if title_suffix else ''), fontsize=FONT_SIZE, y=0.98)
    plt.tight_layout()
    _savefig(fig, out_base)
    np.savez(out_base + '_data.npz', **_save_data)
    plt.close()


def fig_first_order_lines(z0, z1, z2, out_base: str, prompt_text: str = "", title_suffix: str = ""):
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    e0 = _first_order_error_per_step(z0)
    e1 = _first_order_error_per_step(z1)
    e2 = _first_order_error_per_step(z2)
    steps = np.arange(2, 50)
    if len(e0): ax.plot(steps, e0, '-', color=COLOR_Z0, label=f'z0 (dim={z0.shape[-1]})', linewidth=1.5)
    if len(e1): ax.plot(steps, e1, '-', color=COLOR_Z1, label=f'z1 (dim={z1.shape[-1]})', linewidth=1.5)
    if len(e2): ax.plot(steps, e2, '-', color=COLOR_Z2, label=f'z2 (dim={z2.shape[-1]})', linewidth=1.5)
    ax.set_xlabel('Denoising step $t$', fontsize=FONT_SIZE)
    ax.set_ylabel('1st-order extrap. error', fontsize=FONT_SIZE)
    ax.set_title('1st-order Predictability' + (' ' + title_suffix if title_suffix else ''), fontsize=FONT_SIZE)
    if prompt_text:
        txt = prompt_text[:72] + '...' if len(prompt_text) > 72 else prompt_text
        fig.text(0.5, 1.02, 'Prompt: ' + txt, transform=fig.transFigure, ha='center', va='bottom', fontsize=9, color=PALETTE['gray'])
    ax.legend(loc='best', fontsize=9)
    ax.tick_params(axis='both', labelsize=TICK_SIZE)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    _savefig(fig, out_base)
    np.savez(out_base + '_data.npz', steps=steps, e0=e0, e1=e1, e2=e2)
    plt.close()


# ========== Feature PCA Trajectories ==========

def fig_feature_pca_trajectories(feats, out_dir: str, n_select: int = 20):
    """
    Evenly sample n_select dims from D; for each dim take the [50, N] matrix
    (50 steps as samples, N tokens as features) PCA -> [50, 2],
    Draw a trajectory plot in the same style as fig_1_1_trajectory_pca.

    Args:
        feats: [n_steps, N, D] tensor/ndarray (e.g. [50, 4096, 3072])
        out_dir: output directory
        n_select: number of feature dims sampled at equal spacing
    """
    feats_np = feats.float().numpy() if isinstance(feats, torch.Tensor) else np.asarray(feats)
    n_steps, N, D = feats_np.shape

    # Sample feature dimensions at equal spacing
    n_sel = min(n_select, D)
    dim_indices = np.linspace(0, D - 1, n_sel, dtype=int)

    os.makedirs(out_dir, exist_ok=True)
    steps = np.arange(n_steps)
    norm = mpl_colors.Normalize(vmin=0, vmax=n_steps - 1)

    for dim_idx in dim_indices:
        # [50, N]: the N tokens of each step at dim_idx
        feat_slice = feats_np[:, :, dim_idx]  # [n_steps, N]

        coords, var_ratio = _do_pca(feat_slice, 2)  # [n_steps, 2]

        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=steps, cmap=CMAP_STEP,
                        norm=norm, s=28, alpha=0.92, edgecolors='white', linewidths=0.3)
        ax.plot(coords[:, 0], coords[:, 1], color=PALETTE['gray'], alpha=0.5, linewidth=1.2)
        color_t0 = CMAP_STEP(norm(0))
        color_tlast = CMAP_STEP(norm(n_steps - 1))
        ax.scatter(coords[0, 0], coords[0, 1], c=[color_t0], s=100, marker='o',
                   edgecolors='white', linewidths=2, zorder=5, label='t=0')
        ax.scatter(coords[-1, 0], coords[-1, 1], c=[color_tlast], s=100, marker='s',
                   edgecolors='white', linewidths=2, zorder=5, label=f't={n_steps - 1}')
        cbar = plt.colorbar(sc, ax=ax, shrink=0.75)
        cbar.set_label('Denoising step $t$', fontsize=FONT_SIZE)
        ax.set_xlabel(f'PC1 ({var_ratio[0] * 100:.1f}% var.)', fontsize=FONT_SIZE)
        ax.set_ylabel(f'PC2 ({var_ratio[1] * 100:.1f}% var.)', fontsize=FONT_SIZE)
        ax.set_title(f'Feature dim {dim_idx} — Trajectory ({n_steps} Steps)', fontsize=FONT_SIZE)
        ax.legend(loc='lower left', fontsize=9)
        ax.tick_params(axis='both', labelsize=TICK_SIZE)
        ax.set_aspect('equal', adjustable='datalim')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        out_path = os.path.join(out_dir, f'dim_{dim_idx:04d}')
        _savefig(fig, out_path)
        np.savez(out_path + '_data.npz', steps=steps,
                 pc1=coords[:, 0], pc2=coords[:, 1], var_ratio=var_ratio)
        plt.close()


# ========== Model loading ==========

def load_model(checkpoint_path: str, split_dims: List[int], device: str = 'cuda') -> LearnedDecompositionPredictor:
    """Load a flux_v4_multi_stage LearnedDecompositionPredictor"""
    # Try to read the config json written by save_pretrained
    config_path = checkpoint_path.replace('.pt', '_config.json')
    if not config_path.endswith('_config.json'):
        config_path = checkpoint_path + '_config.json'

    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        split_dims_use = config.get('split_dims', split_dims)
        print(f"Loaded config from {config_path}: split_dims={split_dims_use}")
    else:
        split_dims_use = split_dims
        config = {}
        print(f"No config found, using split_dims={split_dims_use}")

    model = LearnedDecompositionPredictor(
        dim=config.get('dim', 3072),
        num_blocks=config.get('num_blocks', 1),
        hidden_dim=config.get('hidden_dim', 128),
        split_dims=split_dims_use,
        dropout=config.get('dropout', 0.0),
    )

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    # save_pretrained writes a raw state_dict (no 'model_state_dict' wrapper)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state = ckpt['model_state_dict']
    elif isinstance(ckpt, dict) and any(k.startswith('net.') for k in ckpt.keys()):
        state = ckpt
    else:
        state = ckpt

    # Drop cached weight_inv (a None buffer is absent from the state_dict)
    inv_weights = {k: v for k, v in state.items() if k.endswith('.weight_inv')}
    main_weights = {k: v for k, v in state.items() if not k.endswith('.weight_inv')}
    model.load_state_dict(main_weights, strict=True)
    for i, block in enumerate(model.net.blocks):
        key = f'net.blocks.{i}.conv1x1.weight_inv'
        if key in inv_weights:
            block.conv1x1.weight_inv = inv_weights[key]

    model.to(device)
    model.eval()
    return model, split_dims_use


# ========== Main ==========

def process_one_prompt(
    prompt_idx: int,
    cache_dir: str,
    checkpoint_path: Optional[str],
    split_dims: List[int],
    output_dir: str,
    prompts_list: List[str],
    device: str,
    model: Optional[LearnedDecompositionPredictor] = None,
):
    prompt_text = prompts_list[prompt_idx] if prompt_idx < len(prompts_list) else f"prompt_{prompt_idx}"
    prefix = f"prompt_{prompt_idx:03d}"

    feats = load_prompt_cache(cache_dir, prompt_idx)   # [50, 4096, 3072]
    feats_mean = mean_over_tokens(feats)                # [50, 3072]

    # Panel 1: dynamics (no model needed)
    asp1_dir = os.path.join(output_dir, 'aspect1')
    os.makedirs(asp1_dir, exist_ok=True)
    fig_1_1_trajectory_pca(feats_mean, os.path.join(asp1_dir, f'{prefix}_1_1_trajectory'), prompt_text)
    for n in [2, 3, 4, 5]:
        fig_1_2_stage_trajectories(feats_mean, os.path.join(asp1_dir, f'{prefix}_1_2_stages{n}'), n_stages=n, prompt_text=prompt_text)
    fig_1_3_step_change_magnitude(feats_mean, os.path.join(asp1_dir, f'{prefix}_1_3_step_change'), prompt_text)

    # Panel 3: raw split (no model, just slice)
    z0_raw, z1_raw, z2_raw = split_three(feats_mean, split_dims)
    asp3_dir = os.path.join(output_dir, 'aspect3')
    os.makedirs(asp3_dir, exist_ok=True)
    sd_tag = f"{split_dims[0]}_{split_dims[1]}_{split_dims[2]}"

    fig_similarity_lines(z0_raw, z1_raw, z2_raw, os.path.join(asp3_dir, f'{prefix}_raw_similarity'), prompt_text, f'(Raw, split={sd_tag})')
    fig_continuity_pca(z0_raw, z1_raw, z2_raw, os.path.join(asp3_dir, f'{prefix}_raw_continuity'), prompt_text, f'(Raw, split={sd_tag})', is_raw=True)
    fig_first_order_lines(z0_raw, z1_raw, z2_raw, os.path.join(asp3_dir, f'{prefix}_raw_first_order'), prompt_text, f'(Raw, split={sd_tag})')

    # Panel 3: after learned decomposition
    if checkpoint_path and os.path.exists(checkpoint_path):
        with torch.no_grad():
            x_in = torch.from_numpy(feats_mean).float().to(device)  # [50, 3072]
            z = model.forward(x_in)                                  # [50, 3072]
            parts = torch.split(z, split_dims, dim=-1)
            z0_dec = parts[0].cpu().float().numpy()
            z1_dec = parts[1].cpu().float().numpy()
            z2_dec = parts[2].cpu().float().numpy()
        fig_similarity_lines(z0_dec, z1_dec, z2_dec, os.path.join(asp3_dir, f'{prefix}_decomposed_similarity'), prompt_text, f'(Decomposed, split={sd_tag})')
        fig_continuity_pca(z0_dec, z1_dec, z2_dec, os.path.join(asp3_dir, f'{prefix}_decomposed_continuity'), prompt_text, f'(Decomposed, split={sd_tag})', is_raw=False)
        fig_first_order_lines(z0_dec, z1_dec, z2_dec, os.path.join(asp3_dir, f'{prefix}_decomposed_first_order'), prompt_text, f'(Decomposed, split={sd_tag})')
    else:
        print(f"  No checkpoint, skipping decomposed figures for {prefix}")

    # Feature PCA trajectories (per-dim, PCA trajectory over N tokens for each feature dim)
    pca_traj_dir = os.path.join(output_dir, 'feature_pca')
    fig_feature_pca_trajectories(feats, pca_traj_dir)

    print(f"  Done {prefix}")


def main():
    parser = argparse.ArgumentParser(description='Generate Introduction figures (flux_v4_multi_stage)')
    parser.add_argument('--cache_dir', type=str,
                        default='./cache_data/flux',
                        help='Cache data root (prompt_xxx/step_xx.pt)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='path to best_predictor_stageX.pt (optional, for decomposition plots)')
    parser.add_argument('--split_dims', type=str, default='2304,384,384',
                        help='three split dims, comma-separated, must match the checkpoint (default: 2304,384,384)')
    parser.add_argument('--prompts_file', type=str,
                        default='prompts/DrawBench200.txt',
                        help='Prompts text file')
    parser.add_argument('--output_dir', type=str, default='./intro_figures',
                        help='Output directory for figures')
    parser.add_argument('--prompt_indices', type=str, default='0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19',
                        help='Comma-separated prompt indices')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    split_dims = [int(x) for x in args.split_dims.split(',')]
    assert len(split_dims) == 3, "split_dims must contain exactly three values"
    assert sum(split_dims) == 3072, f"split_dims must sum to 3072, got {sum(split_dims)}"

    if os.path.exists(args.prompts_file):
        with open(args.prompts_file, 'r', encoding='utf-8') as f:
            prompts_list = [line.strip() for line in f if line.strip()]
    else:
        print(f"Warning: prompts file not found: {args.prompts_file}, using empty prompts")
        prompts_list = []

    indices = [int(x.strip()) for x in args.prompt_indices.split(',')]
    print(f"Processing {len(indices)} prompts: {indices}")
    print(f"Cache: {args.cache_dir}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Split dims: {split_dims}")
    print(f"Output: {args.output_dir}\n")

    os.makedirs(args.output_dir, exist_ok=True)

    model = None
    model_split_dims = split_dims
    if args.checkpoint and os.path.exists(args.checkpoint):
        model, model_split_dims = load_model(args.checkpoint, split_dims, args.device)
        print(f"Loaded model, effective split_dims={model_split_dims}\n")
    elif args.checkpoint:
        print(f"Warning: checkpoint not found: {args.checkpoint}, skipping decomposed figures\n")

    for idx in indices:
        prompt_out = os.path.join(args.output_dir, f'prompt_{idx:03d}')
        os.makedirs(prompt_out, exist_ok=True)
        try:
            process_one_prompt(idx, args.cache_dir, args.checkpoint, model_split_dims,
                               prompt_out, prompts_list, args.device, model=model)
        except Exception as e:
            print(f"  Error prompt_{idx:03d}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 50)
    print("All figures saved to:", os.path.abspath(args.output_dir))
    print("  Each prompt: prompt_000/, ...; under each: aspect1/, aspect3/")


if __name__ == '__main__':
    main()
