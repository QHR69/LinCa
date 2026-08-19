#!/usr/bin/env python3
"""
Introduction-figure script — HunyuanVideo (hunyuan_v4)

- Panel 1: dynamics across denoising — 1-1 trajectory, 1-2 multi-stage, 1-3 step deltas
- Panel 3: 3 raw vs 3 decomposed plots

Usage:
    python visualize_intro_figures_hunyuan.py \
        --cache_dir ./cache_data/hunyuan \
        --checkpoint ./checkpoints/hunyuan/best_predictor.pt \
        --output_dir ./intro_figures_hunyuan
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
from typing import List, Optional

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

# HunyuanVideo partition widths
SPLIT_DIMS_LIST = [2304, 384, 384]

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
    """HunyuanVideo cache layout: prompt_{idx:04d}/step_{step:02d}.pt, shape [20400, 3072], bfloat16"""
    prompt_dir = os.path.join(cache_dir, f'prompt_{prompt_idx:04d}')
    if not os.path.isdir(prompt_dir):
        raise FileNotFoundError(f"Cache not found: {prompt_dir}")
    feats = []
    for step in range(50):
        p = os.path.join(prompt_dir, f'step_{step:02d}.pt')
        if not os.path.exists(p):
            raise FileNotFoundError(f"Step file not found: {p}")
        x = torch.load(p, map_location='cpu', weights_only=True)
        feats.append(x.float())
    return torch.stack(feats, dim=0)  # [50, 20400, 3072]


def mean_over_tokens(feats: torch.Tensor) -> torch.Tensor:
    """[50, tokens, dim] → [50, dim]"""
    return feats.mean(dim=1)


def split_dims(x, split_dims_list: List[int]):
    d0, d1, d2 = split_dims_list
    return x[..., :d0], x[..., d0:d0+d1], x[..., d0+d1:]


def _do_pca(X: np.ndarray, n_components: int = 2):
    if PCA is not None:
        pca = PCA(n_components=n_components)
        coords = pca.fit_transform(X)
        return coords, pca.explained_variance_ratio_
    coords, var_ratio = pca_fit_transform(X, n_components)
    return coords, var_ratio


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


def _similarity_per_step(z: np.ndarray) -> np.ndarray:
    if z.shape[0] < 2:
        return np.array([])
    a, b = z[:-1], z[1:]
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return np.sum(an * bn, axis=1)


def _first_order_error_per_step(z: np.ndarray) -> np.ndarray:
    if z.shape[0] < 3:
        return np.array([])
    pred = z[1:-1] + (z[1:-1] - z[:-2])
    return np.linalg.norm(pred - z[2:], axis=1)


def fig_similarity_lines(z0, z1, z2, out_base: str, prompt_text: str = "", title_suffix: str = ""):
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    s0, s1, s2 = _similarity_per_step(z0), _similarity_per_step(z1), _similarity_per_step(z2)
    steps = np.arange(1, 50)
    if len(s0): ax.plot(steps, s0, '-', color=COLOR_Z0, label='z0 (main)', linewidth=1.5)
    if len(s1): ax.plot(steps, s1, '-', color=COLOR_Z1, label='z1', linewidth=1.5)
    if len(s2): ax.plot(steps, s2, '-', color=COLOR_Z2, label='z2', linewidth=1.5)
    ax.set_xlabel('Denoising step $t$', fontsize=FONT_SIZE)
    ax.set_ylabel('Cosine similarity (adjacent steps)', fontsize=FONT_SIZE)
    ax.set_title('Cosine Similarity' + (' ' + title_suffix if title_suffix else ''), fontsize=FONT_SIZE)
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


def fig_continuity_pca(z0, z1, z2, out_base: str, prompt_text: str = "", title_suffix: str = "", is_raw: bool = True):
    subplot_titles = ('Dim1 (main)', 'Dim2', 'Dim3') if is_raw else ('Dim1: z0 (decomposed)', 'Dim2: z1', 'Dim3: z2')
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.8))
    spine_lw = 4
    norm_step = mpl_colors.Normalize(vmin=0, vmax=49)
    _save_data = {}
    for zi, (ax, z, title) in enumerate([(axes[0], z0, subplot_titles[0]),
                             (axes[1], z1, subplot_titles[1]),
                             (axes[2], z2, subplot_titles[2])]):
        if z.shape[0] < 2:
            ax.set_title(title)
            _aspect3_ax_style(ax, spine_lw)
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
        ax.set_title(title, fontsize=FONT_SIZE)
        _aspect3_ax_style(ax, spine_lw)
        ax.locator_params(axis='both', nbins=6)
        ax.tick_params(axis='both', labelbottom=False, labelleft=False)
    if prompt_text:
        txt = prompt_text[:72] + '...' if len(prompt_text) > 72 else prompt_text
        fig.text(0.5, 1.02, 'Prompt: ' + txt, transform=fig.transFigure, ha='center', va='bottom', fontsize=9, color=PALETTE['gray'])
    fig.suptitle('PCA Trajectory' + (' ' + title_suffix if title_suffix else ''), fontsize=FONT_SIZE, y=0.98)
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
    if len(e0): ax.plot(steps, e0, '-', color=COLOR_Z0, label='z0 (main)', linewidth=1.5)
    if len(e1): ax.plot(steps, e1, '-', color=COLOR_Z1, label='z1', linewidth=1.5)
    if len(e2): ax.plot(steps, e2, '-', color=COLOR_Z2, label='z2', linewidth=1.5)
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
    Evenly sample n_select dims from D; for each dim take the [n_steps, N] matrix
    (n_steps as samples, N tokens as features) PCA -> [n_steps, 2],
    Draw a trajectory plot in the same style as fig_1_1_trajectory_pca.

    Args:
        feats: [n_steps, N, D] tensor/ndarray (e.g. [50, 20400, 3072])
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
        # [n_steps, N]: the N tokens of each step at dim_idx
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


def load_hunyuan_model(checkpoint_path: str, device: str = 'cuda') -> LearnedDecompositionPredictor:
    """Load a HunyuanVideo LearnedDecompositionPredictor via from_pretrained (handles weight_inv + config)"""
    return LearnedDecompositionPredictor.from_pretrained(checkpoint_path, device=device)


def process_one_prompt(
    prompt_idx: int,
    cache_dir: str,
    checkpoint_path: Optional[str],
    output_dir: str,
    prompts_list: List[str],
    device: str,
    model: Optional[LearnedDecompositionPredictor] = None,
):
    prompt_text = prompts_list[prompt_idx] if prompt_idx < len(prompts_list) else f"prompt_{prompt_idx}"
    prefix = f"prompt_{prompt_idx:04d}"

    feats = load_prompt_cache(cache_dir, prompt_idx)        # [50, 20400, 3072]
    feats_mean = mean_over_tokens(feats).numpy()            # [50, 3072]

    # Panel 1
    asp1_dir = os.path.join(output_dir, 'aspect1')
    os.makedirs(asp1_dir, exist_ok=True)
    fig_1_1_trajectory_pca(feats_mean, os.path.join(asp1_dir, f'{prefix}_1_1_trajectory'), prompt_text)
    for n in [2, 3, 4, 5]:
        fig_1_2_stage_trajectories(feats_mean, os.path.join(asp1_dir, f'{prefix}_1_2_stages{n}'), n_stages=n, prompt_text=prompt_text)
    fig_1_3_step_change_magnitude(feats_mean, os.path.join(asp1_dir, f'{prefix}_1_3_step_change'), prompt_text)

    # Panel 3: raw split
    z0_raw, z1_raw, z2_raw = split_dims(feats_mean, SPLIT_DIMS_LIST)
    asp3_dir = os.path.join(output_dir, 'aspect3')
    os.makedirs(asp3_dir, exist_ok=True)
    fig_similarity_lines(z0_raw, z1_raw, z2_raw, os.path.join(asp3_dir, f'{prefix}_raw_similarity'), prompt_text, '(Raw)')
    fig_continuity_pca(z0_raw, z1_raw, z2_raw, os.path.join(asp3_dir, f'{prefix}_raw_continuity'), prompt_text, '(Raw)', is_raw=True)
    fig_first_order_lines(z0_raw, z1_raw, z2_raw, os.path.join(asp3_dir, f'{prefix}_raw_first_order'), prompt_text, '(Raw)')

    # Panel 3: decomposed (needs a checkpoint)
    if checkpoint_path and os.path.exists(checkpoint_path):
        if model is None:
            model = load_hunyuan_model(checkpoint_path, device)
        with torch.no_grad():
            inp = torch.from_numpy(feats_mean).float().to(device)  # [50, 3072]
            z = model.forward(inp)
            z0_dec, z1_dec, z2_dec = split_dims(z.cpu().float().numpy(), SPLIT_DIMS_LIST)
        fig_similarity_lines(z0_dec, z1_dec, z2_dec, os.path.join(asp3_dir, f'{prefix}_decomposed_similarity'), prompt_text, '(Decomposed)')
        fig_continuity_pca(z0_dec, z1_dec, z2_dec, os.path.join(asp3_dir, f'{prefix}_decomposed_continuity'), prompt_text, '(Decomposed)', is_raw=False)
        fig_first_order_lines(z0_dec, z1_dec, z2_dec, os.path.join(asp3_dir, f'{prefix}_decomposed_first_order'), prompt_text, '(Decomposed)')
    else:
        print(f"  [WARN] No checkpoint, skipping decomposed figures for {prefix}")

    # Feature PCA trajectories (per-dim, PCA trajectory over N tokens for each feature dim)
    pca_traj_dir = os.path.join(output_dir, 'feature_pca')
    fig_feature_pca_trajectories(feats, pca_traj_dir)

    print(f"  Done {prefix}: {prompt_text[:60]}")


def main():
    parser = argparse.ArgumentParser(description='HunyuanVideo Introduction figures')
    parser.add_argument('--cache_dir', type=str,
                        default='./cache_data/hunyuan',
                        help='Cache data root (prompt_xxxx/step_xx.pt)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='LearnedDecompositionPredictor checkpoint (.pt)')
    parser.add_argument('--vbench_json', type=str,
                        default='./data/VBench_full_info.json',
                        help='VBench prompts JSON')
    parser.add_argument('--output_dir', type=str, default='./intro_figures_hunyuan',
                        help='Output directory for figures')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    with open(args.vbench_json, 'r', encoding='utf-8') as f:
        prompts_data = json.load(f)
    prompts_list = [item['prompt_en'] for item in prompts_data]

    indices = list(range(20))
    print(f"Processing first 20 prompts: {indices}")
    print(f"Cache: {args.cache_dir}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output: {args.output_dir}")
    print(f"Split dims (HunyuanVideo): {SPLIT_DIMS_LIST}\n")

    os.makedirs(args.output_dir, exist_ok=True)

    model = None
    if args.checkpoint and os.path.exists(args.checkpoint):
        model = load_hunyuan_model(args.checkpoint, args.device)
        print("Loaded LearnedDecompositionPredictor\n")

    for idx in indices:
        prompt_dir = os.path.join(args.output_dir, f'prompt_{idx:04d}')
        os.makedirs(prompt_dir, exist_ok=True)
        try:
            process_one_prompt(idx, args.cache_dir, args.checkpoint, prompt_dir, prompts_list, args.device, model=model)
        except Exception as e:
            print(f"  Error prompt_{idx:04d}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 50)
    print("All figures saved to:", os.path.abspath(args.output_dir))


if __name__ == '__main__':
    main()
