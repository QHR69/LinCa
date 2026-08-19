"""
Training script for the invertible decomposition network (two-stage)

Plan: same architecture, separate parameters per stage
- Stage 1: steps 0-24 (first 25 steps)
- Stage 2: steps 25-49 (last 25 steps)
- Two identical invertible nets, trained on the first and last 25 steps respectively
- Early stopping B: stop only after both stages stall for N epochs
"""

import os
import sys
import argparse
import json
import math
import random
import gc
import ctypes
import numpy as np
from tqdm import tqdm
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

# Wandb for monitoring
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available, training will proceed without logging")

# Local imports
from flux.modules.invertible_net import (
    InvertibleDecompositionNet,
    LearnedDecompositionPredictor,
    FixedPredictionStrategy,
)
from flux.dataset import MemoryEfficientCacheDataset, collate_fn, create_dataloader

# Stage-boundary constants
STAGE1_MAX_STEP = 24   # 0-24 is stage1
STAGE2_MIN_STEP = 25   # 25-49 is stage2


def set_seed(seed: int):
    """Set random seeds"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_training_optimizations():
    """Enable training-speed optimisations"""
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        if hasattr(torch, 'set_float32_matmul_precision'):
            torch.set_float32_matmul_precision('high')
        print("✓ Training optimizations enabled")


def aggressive_memory_cleanup():
    """Aggressive memory cleanup"""
    gc.collect()
    gc.collect()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass


def compute_loss(
    predictor: LearnedDecompositionPredictor,
    target_features: torch.Tensor,
    cache_features: torch.Tensor,
    cache_mask: torch.Tensor,
    step_distances: torch.Tensor,
    intervals: torch.Tensor,
    z_loss_weight: float = 0.1,
) -> Dict[str, torch.Tensor]:
    """Compute the training loss (same logic as the gate-free variant)"""
    batch_size = target_features.shape[0]
    device = target_features.device
    
    total_loss = torch.tensor(0.0, device=device)
    z0_loss = torch.tensor(0.0, device=device)
    z1_loss = torch.tensor(0.0, device=device)
    z2_loss = torch.tensor(0.0, device=device)
    reconstruction_loss = torch.tensor(0.0, device=device)
    valid_count = 0
    
    for b in range(batch_size):
        valid_mask = cache_mask[b]
        num_valid = valid_mask.sum().item()
        if num_valid == 0:
            continue
        valid_count += 1
        
        nearest_distance = step_distances[b, 0].item()
        decomposed_cache = []
        for i in range(int(num_valid)):
            dist = step_distances[b, i].item()
            feat = cache_features[b, i]
            z0, z1, z2 = predictor.decompose(feat)
            decomposed_cache.append((dist, z0, z1, z2))
        
        target = target_features[b]
        target_z0, target_z1, target_z2 = predictor.decompose(target)
        strategy = predictor.prediction_strategy
        
        _, z0_prev, _, _ = decomposed_cache[0]
        pred_z0 = strategy.predict_z0(z0_prev)
        loss_z0 = F.mse_loss(pred_z0, target_z0)
        
        if num_valid >= 2:
            dist1, _, z1_1, _ = decomposed_cache[0]
            dist2, _, z1_2, _ = decomposed_cache[1]
            pred_z1 = strategy.predict_z1(z1_1, z1_2, dist1, dist2)
        else:
            _, _, z1_prev, _ = decomposed_cache[0]
            pred_z1 = strategy.predict_z0(z1_prev)
        loss_z1 = F.mse_loss(pred_z1, target_z1)
        
        if num_valid >= 3:
            dist1, _, _, z2_1 = decomposed_cache[0]
            dist2, _, _, z2_2 = decomposed_cache[1]
            dist3, _, _, z2_3 = decomposed_cache[2]
            pred_z2 = strategy.predict_z2(z2_1, z2_2, z2_3, dist1, dist2, dist3)
        elif num_valid >= 2:
            dist1, _, _, z2_1 = decomposed_cache[0]
            dist2, _, _, z2_2 = decomposed_cache[1]
            pred_z2 = strategy.predict_z1(z2_1, z2_2, dist1, dist2)
        else:
            _, _, _, z2_prev = decomposed_cache[0]
            pred_z2 = strategy.predict_z0(z2_prev)
        loss_z2 = F.mse_loss(pred_z2, target_z2)
        
        pred_reconstructed = predictor.compose(pred_z0, pred_z1, pred_z2)
        loss_recon = F.mse_loss(pred_reconstructed, target)
        
        norm_factor = max(1, nearest_distance)
        z0_loss = z0_loss + loss_z0 / norm_factor
        z1_loss = z1_loss + loss_z1 / norm_factor
        z2_loss = z2_loss + loss_z2 / norm_factor
        reconstruction_loss = reconstruction_loss + loss_recon / norm_factor
        
        del decomposed_cache, target, target_z0, target_z1, target_z2
        del pred_z0, pred_z1, pred_z2, pred_reconstructed
    
    if valid_count > 0:
        z0_loss = z0_loss / valid_count
        z1_loss = z1_loss / valid_count
        z2_loss = z2_loss / valid_count
        reconstruction_loss = reconstruction_loss / valid_count
    
    total_loss = reconstruction_loss + z_loss_weight * (z0_loss + z1_loss + z2_loss)
    
    return {
        'total_loss': total_loss,
        'reconstruction_loss': reconstruction_loss,
        'z0_loss': z0_loss,
        'z1_loss': z1_loss,
        'z2_loss': z2_loss,
    }


def train_epoch(
    predictor_stage1: LearnedDecompositionPredictor,
    predictor_stage2: LearnedDecompositionPredictor,
    dataloader: DataLoader,
    optimizer1: optim.Optimizer,
    optimizer2: optim.Optimizer,
    device: torch.device,
    epoch: int,
    log_interval: int = 10,
    use_wandb: bool = False,
    grad_accum_steps: int = 1,
    scaler: Optional[torch.amp.GradScaler] = None,
    use_amp: bool = False,
    z_loss_weight: float = 0.1,
) -> Dict[str, float]:
    """Run one two-stage training epoch"""
    predictor_stage1.train()
    predictor_stage2.train()
    
    accum_s1 = {'loss': 0.0, 'recon': 0.0, 'z0': 0.0, 'z1': 0.0, 'z2': 0.0, 'count': 0}
    accum_s2 = {'loss': 0.0, 'recon': 0.0, 'z0': 0.0, 'z1': 0.0, 'z2': 0.0, 'count': 0}
    
    total_s1 = {'loss': 0.0, 'recon': 0.0, 'z0': 0.0, 'z1': 0.0, 'z2': 0.0}
    total_s2 = {'loss': 0.0, 'recon': 0.0, 'z0': 0.0, 'z1': 0.0, 'z2': 0.0}
    num_batches_s1 = 0
    num_batches_s2 = 0
    
    optimizer1.zero_grad(set_to_none=True)
    optimizer2.zero_grad(set_to_none=True)
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        target_features = batch['target_features'].to(device, non_blocking=True)
        cache_features = batch['cache_features'].to(device, non_blocking=True)
        cache_mask = batch['cache_mask'].to(device, non_blocking=True)
        step_distances = batch['step_distances'].to(device, non_blocking=True)
        intervals = batch['intervals'].to(device, non_blocking=True)
        target_steps = batch['target_steps'].to(device, non_blocking=True)
        
        stage1_mask = (target_steps <= STAGE1_MAX_STEP)
        stage2_mask = (target_steps >= STAGE2_MIN_STEP)
        
        n_s1 = stage1_mask.sum().item()
        n_s2 = stage2_mask.sum().item()
        
        if n_s1 > 0:
            with torch.amp.autocast('cuda', enabled=use_amp):
                losses_s1 = compute_loss(
                    predictor_stage1,
                    target_features[stage1_mask],
                    cache_features[stage1_mask],
                    cache_mask[stage1_mask],
                    step_distances[stage1_mask],
                    intervals[stage1_mask],
                    z_loss_weight=z_loss_weight,
                )
                loss_s1 = losses_s1['total_loss'] / grad_accum_steps
            
            if scaler is not None:
                scaler.scale(loss_s1).backward()
            else:
                loss_s1.backward()
            
            accum_s1['loss'] += losses_s1['total_loss'].detach().item()
            accum_s1['recon'] += losses_s1['reconstruction_loss'].detach().item()
            accum_s1['z0'] += losses_s1['z0_loss'].detach().item()
            accum_s1['z1'] += losses_s1['z1_loss'].detach().item()
            accum_s1['z2'] += losses_s1['z2_loss'].detach().item()
            accum_s1['count'] += 1
        
        if n_s2 > 0:
            with torch.amp.autocast('cuda', enabled=use_amp):
                losses_s2 = compute_loss(
                    predictor_stage2,
                    target_features[stage2_mask],
                    cache_features[stage2_mask],
                    cache_mask[stage2_mask],
                    step_distances[stage2_mask],
                    intervals[stage2_mask],
                    z_loss_weight=z_loss_weight,
                )
                loss_s2 = losses_s2['total_loss'] / grad_accum_steps
            
            if scaler is not None:
                scaler.scale(loss_s2).backward()
            else:
                loss_s2.backward()
            
            accum_s2['loss'] += losses_s2['total_loss'].detach().item()
            accum_s2['recon'] += losses_s2['reconstruction_loss'].detach().item()
            accum_s2['z0'] += losses_s2['z0_loss'].detach().item()
            accum_s2['z1'] += losses_s2['z1_loss'].detach().item()
            accum_s2['z2'] += losses_s2['z2_loss'].detach().item()
            accum_s2['count'] += 1
        
        del target_features, cache_features, cache_mask, step_distances, intervals, target_steps
        if n_s1 > 0:
            del losses_s1, loss_s1
        if n_s2 > 0:
            del losses_s2, loss_s2
        
        if (batch_idx + 1) % grad_accum_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer1)
                scaler.unscale_(optimizer2)
                torch.nn.utils.clip_grad_norm_(predictor_stage1.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(predictor_stage2.parameters(), max_norm=1.0)
                scaler.step(optimizer1)
                scaler.step(optimizer2)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(predictor_stage1.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(predictor_stage2.parameters(), max_norm=1.0)
                optimizer1.step()
                optimizer2.step()
            optimizer1.zero_grad(set_to_none=True)
            optimizer2.zero_grad(set_to_none=True)
            
            if accum_s1['count'] > 0:
                c = accum_s1['count']
                total_s1['loss'] += accum_s1['loss']
                total_s1['recon'] += accum_s1['recon']
                total_s1['z0'] += accum_s1['z0']
                total_s1['z1'] += accum_s1['z1']
                total_s1['z2'] += accum_s1['z2']
                num_batches_s1 += c
            if accum_s2['count'] > 0:
                c = accum_s2['count']
                total_s2['loss'] += accum_s2['loss']
                total_s2['recon'] += accum_s2['recon']
                total_s2['z0'] += accum_s2['z0']
                total_s2['z1'] += accum_s2['z1']
                total_s2['z2'] += accum_s2['z2']
                num_batches_s2 += c
            
            avg_s1 = accum_s1['loss'] / max(accum_s1['count'], 1)
            avg_s2 = accum_s2['loss'] / max(accum_s2['count'], 1)
            pbar.set_postfix({'s1': f"{avg_s1:.4f}", 's2': f"{avg_s2:.4f}"})
            
            if use_wandb and WANDB_AVAILABLE and (batch_idx + 1) % (log_interval * grad_accum_steps) == 0:
                global_step = epoch * len(dataloader) + batch_idx
                log_d = {'train/global_step': global_step}
                if accum_s1['count'] > 0:
                    c = accum_s1['count']
                    log_d['train/batch_total_loss_stage1'] = accum_s1['loss'] / c
                    log_d['train/batch_recon_loss_stage1'] = accum_s1['recon'] / c
                if accum_s2['count'] > 0:
                    c = accum_s2['count']
                    log_d['train/batch_total_loss_stage2'] = accum_s2['loss'] / c
                    log_d['train/batch_recon_loss_stage2'] = accum_s2['recon'] / c
                wandb.log(log_d, step=global_step)
            
            accum_s1 = {'loss': 0.0, 'recon': 0.0, 'z0': 0.0, 'z1': 0.0, 'z2': 0.0, 'count': 0}
            accum_s2 = {'loss': 0.0, 'recon': 0.0, 'z0': 0.0, 'z1': 0.0, 'z2': 0.0, 'count': 0}
        
        if (batch_idx + 1) % 50 == 0:
            torch.cuda.empty_cache()
    
    remaining = len(dataloader) % grad_accum_steps
    if remaining > 0 and (accum_s1['count'] > 0 or accum_s2['count'] > 0):
        if scaler is not None:
            scaler.unscale_(optimizer1)
            scaler.unscale_(optimizer2)
            torch.nn.utils.clip_grad_norm_(predictor_stage1.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(predictor_stage2.parameters(), max_norm=1.0)
            scaler.step(optimizer1)
            scaler.step(optimizer2)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(predictor_stage1.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(predictor_stage2.parameters(), max_norm=1.0)
            optimizer1.step()
            optimizer2.step()
        optimizer1.zero_grad(set_to_none=True)
        optimizer2.zero_grad(set_to_none=True)
        if accum_s1['count'] > 0:
            c = accum_s1['count']
            total_s1['loss'] += accum_s1['loss']
            total_s1['recon'] += accum_s1['recon']
            total_s1['z0'] += accum_s1['z0']
            total_s1['z1'] += accum_s1['z1']
            total_s1['z2'] += accum_s1['z2']
            num_batches_s1 += c
        if accum_s2['count'] > 0:
            c = accum_s2['count']
            total_s2['loss'] += accum_s2['loss']
            total_s2['recon'] += accum_s2['recon']
            total_s2['z0'] += accum_s2['z0']
            total_s2['z1'] += accum_s2['z1']
            total_s2['z2'] += accum_s2['z2']
            num_batches_s2 += c
    
    n1 = max(num_batches_s1, 1)
    n2 = max(num_batches_s2, 1)
    return {
        'total_loss_stage1': total_s1['loss'] / n1,
        'reconstruction_loss_stage1': total_s1['recon'] / n1,
        'z0_loss_stage1': total_s1['z0'] / n1,
        'z1_loss_stage1': total_s1['z1'] / n1,
        'z2_loss_stage1': total_s1['z2'] / n1,
        'total_loss_stage2': total_s2['loss'] / n2,
        'reconstruction_loss_stage2': total_s2['recon'] / n2,
        'z0_loss_stage2': total_s2['z0'] / n2,
        'z1_loss_stage2': total_s2['z1'] / n2,
        'z2_loss_stage2': total_s2['z2'] / n2,
    }


@torch.no_grad()
def validate(
    predictor_stage1: LearnedDecompositionPredictor,
    predictor_stage2: LearnedDecompositionPredictor,
    dataloader: DataLoader,
    device: torch.device,
    z_loss_weight: float = 0.1,
) -> Dict[str, float]:
    """两阶段验证"""
    predictor_stage1.eval()
    predictor_stage2.eval()
    
    total_s1 = {'loss': 0.0, 'recon': 0.0, 'z0': 0.0, 'z1': 0.0, 'z2': 0.0}
    total_s2 = {'loss': 0.0, 'recon': 0.0, 'z0': 0.0, 'z1': 0.0, 'z2': 0.0}
    count_s1 = 0
    count_s2 = 0
    
    for batch in tqdm(dataloader, desc="Validation"):
        target_features = batch['target_features'].to(device, non_blocking=True)
        cache_features = batch['cache_features'].to(device, non_blocking=True)
        cache_mask = batch['cache_mask'].to(device, non_blocking=True)
        step_distances = batch['step_distances'].to(device, non_blocking=True)
        intervals = batch['intervals'].to(device, non_blocking=True)
        target_steps = batch['target_steps'].to(device, non_blocking=True)
        
        stage1_mask = (target_steps <= STAGE1_MAX_STEP)
        stage2_mask = (target_steps >= STAGE2_MIN_STEP)
        
        if stage1_mask.any():
            losses_s1 = compute_loss(
                predictor_stage1,
                target_features[stage1_mask],
                cache_features[stage1_mask],
                cache_mask[stage1_mask],
                step_distances[stage1_mask],
                intervals[stage1_mask],
                z_loss_weight=z_loss_weight,
            )
            n = stage1_mask.sum().item()
            total_s1['loss'] += losses_s1['total_loss'].item() * n
            total_s1['recon'] += losses_s1['reconstruction_loss'].item() * n
            total_s1['z0'] += losses_s1['z0_loss'].item() * n
            total_s1['z1'] += losses_s1['z1_loss'].item() * n
            total_s1['z2'] += losses_s1['z2_loss'].item() * n
            count_s1 += n
        
        if stage2_mask.any():
            losses_s2 = compute_loss(
                predictor_stage2,
                target_features[stage2_mask],
                cache_features[stage2_mask],
                cache_mask[stage2_mask],
                step_distances[stage2_mask],
                intervals[stage2_mask],
                z_loss_weight=z_loss_weight,
            )
            n = stage2_mask.sum().item()
            total_s2['loss'] += losses_s2['total_loss'].item() * n
            total_s2['recon'] += losses_s2['reconstruction_loss'].item() * n
            total_s2['z0'] += losses_s2['z0_loss'].item() * n
            total_s2['z1'] += losses_s2['z1_loss'].item() * n
            total_s2['z2'] += losses_s2['z2_loss'].item() * n
            count_s2 += n
        
        del target_features, cache_features, cache_mask, step_distances, intervals, target_steps
    
    n1 = max(count_s1, 1)
    n2 = max(count_s2, 1)
    return {
        'val_total_loss_stage1': total_s1['loss'] / n1,
        'val_recon_loss_stage1': total_s1['recon'] / n1,
        'val_z0_loss_stage1': total_s1['z0'] / n1,
        'val_z1_loss_stage1': total_s1['z1'] / n1,
        'val_z2_loss_stage1': total_s1['z2'] / n1,
        'val_total_loss_stage2': total_s2['loss'] / n2,
        'val_recon_loss_stage2': total_s2['recon'] / n2,
        'val_z0_loss_stage2': total_s2['z0'] / n2,
        'val_z1_loss_stage2': total_s2['z1'] / n2,
        'val_z2_loss_stage2': total_s2['z2'] / n2,
    }


def _thorough_cleanup_pipeline(pipe):
    """彻底清理 pipeline"""
    if pipe is None:
        return
    try:
        pipe.to('cpu')
    except Exception:
        pass
    del pipe
    gc.collect()
    gc.collect()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass
    gc.collect()
    print(">>> Pipeline fully released")


@torch.no_grad()
def eval_generate_images(
    predictor_stage1: LearnedDecompositionPredictor,
    predictor_stage2: LearnedDecompositionPredictor,
    eval_prompts_file: str,
    output_dir: str,
    model_path: str,
    interval: int = 6,
    device: str = 'cuda',
    epoch: int = 0,
    base_seed: int = 0,
) -> List[Image.Image]:
    """在 eval prompts 上生成图片（两阶段版本）"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    from pipeline.pipeline_qwenimage import QwenImagePipeline
    from cache_functions import cache_init, set_predictor_two_stage, get_predictor, pipeline_with_learned_cache
    
    predictor_stage1.eval()
    predictor_stage2.eval()
    if next(predictor_stage1.parameters()).device.type != device.split(':')[0]:
        predictor_stage1 = predictor_stage1.to(device)
        predictor_stage2 = predictor_stage2.to(device)
    
    set_predictor_two_stage(predictor_stage1, predictor_stage2)
    
    with open(eval_prompts_file, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]
    
    print(f"\n>>> Eval: generating {len(prompts)} images (two-stage)...")
    aggressive_memory_cleanup()
    
    pipe = QwenImagePipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
    ).to(device)
    pipe = pipeline_with_learned_cache(pipe)
    
    cache_kwargs = {
        'num_steps': 50,
        'test_FLOPs': False,
        'monitor_gpu_usage': False,
        'interval': interval,
        'max_order': 2,
        'min_order': 0,
        'first_enhance': 3,
        'forecast_method': 'hermite',
        'decompose_method': 'learned',
        'use_z_cache': False,
        'forecast_steps': 5,
        'z2_forecast_method': 'lagrange',
    }
    
    eval_output_dir = os.path.join(output_dir, f'eval_epoch_{epoch}')
    os.makedirs(eval_output_dir, exist_ok=True)
    
    images = []
    for i, prompt in enumerate(prompts):
        seed = base_seed + i
        generator = torch.Generator(device).manual_seed(seed)
        cache_dic, current = cache_init(cache_kwargs)
        
        with torch.inference_mode():
            result = pipe(
                prompt=prompt,
                negative_prompt=" ",
                height=1328,
                width=1328,
                num_inference_steps=50,
                guidance_scale=1.0,
                true_cfg_scale=4.0,
                generator=generator,
                max_sequence_length=512,
                cache_dic=cache_dic,
                current=current,
            )
            image = result.images[0]
        
        output_path = os.path.join(eval_output_dir, f'eval_{i:02d}.jpg')
        image.save(output_path, quality=95)
        images.append(image)
        print(f"  [{i+1}/{len(prompts)}] {prompt[:40]}...")
        
        del result, generator, cache_dic, current, image
        if (i + 1) % 2 == 0:
            torch.cuda.empty_cache()
    
    _thorough_cleanup_pipeline(pipe)
    pipe = None
    return images


def save_checkpoint(
    predictor_stage1: LearnedDecompositionPredictor,
    predictor_stage2: LearnedDecompositionPredictor,
    optimizer1: optim.Optimizer,
    optimizer2: optim.Optimizer,
    epoch: int,
    loss_s1: float,
    loss_s2: float,
    save_path: str,
):
    """保存两阶段检查点"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'predictor_stage1': predictor_stage1.state_dict(),
        'predictor_stage2': predictor_stage2.state_dict(),
        'optimizer1': optimizer1.state_dict(),
        'optimizer2': optimizer2.state_dict(),
        'loss_stage1': loss_s1,
        'loss_stage2': loss_s2,
        'config': {
            'dim': predictor_stage1.dim,
            'num_blocks': predictor_stage1.num_blocks,
            'hidden_dim': predictor_stage1.hidden_dim,
            'split_dims': predictor_stage1.split_dims,
            'dropout': predictor_stage1.dropout,
        }
    }, save_path)
    print(f"Saved checkpoint to {save_path}")


def save_predictor_snapshot(
    predictor_stage1: LearnedDecompositionPredictor,
    predictor_stage2: LearnedDecompositionPredictor,
    output_dir: str,
    epoch: int,
):
    """保存某个 epoch 的两个 predictor 权重（含 config）"""
    os.makedirs(output_dir, exist_ok=True)
    path_s1 = os.path.join(output_dir, f'predictor_stage1_epoch_{epoch:03d}.pt')
    path_s2 = os.path.join(output_dir, f'predictor_stage2_epoch_{epoch:03d}.pt')
    predictor_stage1.save_pretrained(path_s1)
    predictor_stage2.save_pretrained(path_s2)
    print(f"Saved predictor snapshots for epoch {epoch}:")
    print(f"  {path_s1}")
    print(f"  {path_s2}")


def load_checkpoint(
    predictor_stage1: LearnedDecompositionPredictor,
    predictor_stage2: LearnedDecompositionPredictor,
    optimizer1: optim.Optimizer,
    optimizer2: optim.Optimizer,
    load_path: str,
    device: torch.device,
) -> int:
    """加载两阶段检查点"""
    checkpoint = torch.load(load_path, map_location=device)
    predictor_stage1.load_state_dict(checkpoint['predictor_stage1'])
    predictor_stage2.load_state_dict(checkpoint['predictor_stage2'])
    optimizer1.load_state_dict(checkpoint['optimizer1'])
    optimizer2.load_state_dict(checkpoint['optimizer2'])
    return checkpoint['epoch']


def main():
    parser = argparse.ArgumentParser(description="Train two-stage invertible decomposition network (v4)")
    
    parser.add_argument('--data_dir', type=str, default='./cache_data/wan')
    parser.add_argument('--train_prompts', type=str, default='0-946')
    parser.add_argument('--val_sample_range', type=str, default=None)
    parser.add_argument('--num_val_prompts', type=int, default=20)
    parser.add_argument('--prompts_per_epoch', type=int, default=30)
    parser.add_argument('--shuffle_prompts', action='store_true')
    
    parser.add_argument('--dim', type=int, default=1536)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--num_blocks', type=int, default=6)
    parser.add_argument('--split_dims', type=str, default='1152,192,192')
    parser.add_argument('--dropout', type=float, default=0.1)
    
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--z_loss_weight', type=float, default=0.1)
    parser.add_argument('--intervals', type=str, default='6,7,8,9,10')
    parser.add_argument('--random_interval', action='store_true')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--grad_accum_steps', type=int, default=5)
    parser.add_argument('--early_stop_patience', type=int, default=10,
                        help='Early stop when BOTH stages have no improvement for N epochs')
    
    parser.add_argument('--eval_prompts_file', type=str, default='prompts/eval.txt')
    parser.add_argument('--eval_interval', type=int, default=0)
    parser.add_argument('--model_path', type=str, default="Qwen/Qwen-Image")
    
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--log_interval', type=int, default=1)
    parser.add_argument('--save_interval', type=int, default=5)
    parser.add_argument('--wandb_project', type=str, default='linca_v4_two_stage')
    parser.add_argument('--wandb_key', type=str, default=None)
    parser.add_argument('--no_wandb', action='store_true')
    
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    setup_training_optimizations()
    intervals = [int(x) for x in args.intervals.split(',')]
    
    def parse_indices(s):
        if '-' in s:
            start, end = map(int, s.split('-'))
            return list(range(start, end))
        return [int(x) for x in s.split(',')]
    
    all_train_indices = parse_indices(args.train_prompts)
    val_source = parse_indices(args.val_sample_range) if args.val_sample_range else all_train_indices
    num_val = min(args.num_val_prompts, len(val_source))
    total = len(val_source)
    val_positions = [int(i * total / num_val) for i in range(num_val)]
    val_indices = sorted([val_source[p] for p in val_positions])
    val_set = set(val_indices)
    train_pool = [idx for idx in all_train_indices if idx not in val_set]
    
    print(f"\n{'='*60}")
    print(f"Two-stage training: Stage1=0-24, Stage2=25-49")
    print(f"Early stopping B: stop only after both stages stall for {args.early_stop_patience} epochs")
    print(f"数据划分: 训练 {len(train_pool)}, 验证 {len(val_indices)}")
    print(f"{'='*60}\n")
    
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    split_dims = [int(x) for x in args.split_dims.split(',')]
    
    predictor_stage1 = LearnedDecompositionPredictor(
        dim=args.dim, num_blocks=args.num_blocks, hidden_dim=args.hidden_dim,
        split_dims=split_dims, dropout=args.dropout,
    ).to(device)
    predictor_stage2 = LearnedDecompositionPredictor(
        dim=args.dim, num_blocks=args.num_blocks, hidden_dim=args.hidden_dim,
        split_dims=split_dims, dropout=args.dropout,
    ).to(device)
    
    num_params = sum(p.numel() for p in predictor_stage1.parameters())
    print(f"Model parameters per stage: {num_params:,} ({num_params/1e6:.1f}M)")
    
    optimizer1 = optim.AdamW(predictor_stage1.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer2 = optim.AdamW(predictor_stage2.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler1 = optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=args.epochs, eta_min=args.lr * 0.01)
    scheduler2 = optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=args.epochs, eta_min=args.lr * 0.01)
    
    scaler = torch.amp.GradScaler('cuda') if args.amp else None
    if args.amp:
        print("Using AMP")
    
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(
            predictor_stage1, predictor_stage2,
            optimizer1, optimizer2,
            args.resume, device
        )
        print(f"Resumed from epoch {start_epoch}")
    
    val_loader = create_dataloader(
        data_dir=args.data_dir, batch_size=args.batch_size, intervals=intervals,
        num_workers=args.num_workers, shuffle=False, prompt_indices=val_indices,
        subsample_seq=None, random_interval=args.random_interval,
    )
    
    if args.exp_name is None:
        args.exp_name = "two_stage_v4"
    output_dir = os.path.join(args.output_dir, args.wandb_project, args.exp_name)
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # 保存初始化权重（epoch 0）
    if start_epoch == 0:
        save_predictor_snapshot(
            predictor_stage1,
            predictor_stage2,
            output_dir,
            epoch=0,
        )
    
    use_wandb = WANDB_AVAILABLE and not args.no_wandb
    if use_wandb:
        if args.wandb_key:
            wandb.login(key=args.wandb_key)
        elif os.environ.get('WANDB_API_KEY'):
            wandb.login(key=os.environ.get('WANDB_API_KEY'))
        wandb.init(project=args.wandb_project, name=args.exp_name, config=vars(args))
    
    best_val_s1 = float('inf')
    best_val_s2 = float('inf')
    patience_s1 = 0
    patience_s2 = 0
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*60}")
        
        if args.prompts_per_epoch > 0 and args.prompts_per_epoch < len(train_pool):
            epoch_train_indices = random.sample(train_pool, args.prompts_per_epoch)
        else:
            epoch_train_indices = train_pool.copy()
            if args.shuffle_prompts:
                random.shuffle(epoch_train_indices)
        
        train_loader = create_dataloader(
            data_dir=args.data_dir, batch_size=args.batch_size, intervals=intervals,
            num_workers=args.num_workers, shuffle=True, prompt_indices=epoch_train_indices,
            subsample_seq=None, random_interval=args.random_interval,
        )
        
        train_losses = train_epoch(
            predictor_stage1, predictor_stage2, train_loader,
            optimizer1, optimizer2, device, epoch, args.log_interval, use_wandb,
            args.grad_accum_steps, scaler=scaler, use_amp=args.amp,
            z_loss_weight=args.z_loss_weight,
        )
        
        del train_loader
        aggressive_memory_cleanup()
        
        val_losses = validate(
            predictor_stage1, predictor_stage2, val_loader, device,
            z_loss_weight=args.z_loss_weight,
        )
        
        scheduler1.step()
        scheduler2.step()
        
        print(f"\n--- Epoch {epoch + 1} Summary ---")
        print(f"Stage1 Train: {train_losses['total_loss_stage1']:.6f}, Val: {val_losses['val_total_loss_stage1']:.6f}")
        print(f"Stage2 Train: {train_losses['total_loss_stage2']:.6f}, Val: {val_losses['val_total_loss_stage2']:.6f}")
        
        improved_s1 = val_losses['val_total_loss_stage1'] < best_val_s1
        improved_s2 = val_losses['val_total_loss_stage2'] < best_val_s2
        
        if improved_s1:
            best_val_s1 = val_losses['val_total_loss_stage1']
            patience_s1 = 0
        else:
            patience_s1 += 1
        
        if improved_s2:
            best_val_s2 = val_losses['val_total_loss_stage2']
            patience_s2 = 0
        else:
            patience_s2 += 1
        
        if improved_s1 or improved_s2:
            save_checkpoint(
                predictor_stage1, predictor_stage2,
                optimizer1, optimizer2,
                epoch + 1, best_val_s1, best_val_s2,
                os.path.join(output_dir, 'best_model.pt')
            )
            predictor_stage1.save_pretrained(os.path.join(output_dir, 'best_predictor_stage1.pt'))
            predictor_stage2.save_pretrained(os.path.join(output_dir, 'best_predictor_stage2.pt'))
        
        print(f"Early Stop: s1_patience={patience_s1}/{args.early_stop_patience}, s2_patience={patience_s2}/{args.early_stop_patience}")
        
        if use_wandb:
            log_d = {
                'epoch': epoch + 1,
                'train/epoch_total_loss_stage1': train_losses['total_loss_stage1'],
                'train/epoch_total_loss_stage2': train_losses['total_loss_stage2'],
                'val/total_loss_stage1': val_losses['val_total_loss_stage1'],
                'val/total_loss_stage2': val_losses['val_total_loss_stage2'],
                'val/best_stage1': best_val_s1,
                'val/best_stage2': best_val_s2,
                'patience_s1': patience_s1,
                'patience_s2': patience_s2,
                'lr': scheduler1.get_last_lr()[0],
            }
            wandb.log(log_d)
        
        if args.early_stop_patience > 0 and patience_s1 >= args.early_stop_patience and patience_s2 >= args.early_stop_patience:
            print(f"\n>>> Early stopping triggered! (Both stages no improvement for {args.early_stop_patience} epochs)")
            break
        
        if (epoch + 1) % args.save_interval == 0:
            save_checkpoint(
                predictor_stage1, predictor_stage2,
                optimizer1, optimizer2,
                epoch + 1,
                train_losses['total_loss_stage1'], train_losses['total_loss_stage2'],
                os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pt')
            )

        # 每个 epoch 结束都保存两个 predictor 的单独权重
        save_predictor_snapshot(
            predictor_stage1,
            predictor_stage2,
            output_dir,
            epoch=epoch + 1,
        )
        
        if args.eval_interval > 0 and (epoch + 1) % args.eval_interval == 0:
            try:
                images = eval_generate_images(
                    predictor_stage1=predictor_stage1,
                    predictor_stage2=predictor_stage2,
                    eval_prompts_file=args.eval_prompts_file,
                    output_dir=output_dir,
                    model_path=args.model_path,
                    interval=6,
                    device=args.device,
                    epoch=epoch + 1,
                    base_seed=args.seed,
                )
                if use_wandb and images:
                    wandb.log({f'eval/images_epoch_{epoch+1}': [wandb.Image(img, caption=f"eval_{i}") for i, img in enumerate(images)]})
                del images
                aggressive_memory_cleanup()
            except Exception as e:
                print(f"Warning: Eval failed: {e}")
                import traceback
                traceback.print_exc()
    
    save_checkpoint(
        predictor_stage1, predictor_stage2,
        optimizer1, optimizer2,
        args.epochs,
        train_losses['total_loss_stage1'], train_losses['total_loss_stage2'],
        os.path.join(output_dir, 'final_model.pt')
    )
    predictor_stage1.save_pretrained(os.path.join(output_dir, 'predictor_stage1.pt'))
    predictor_stage2.save_pretrained(os.path.join(output_dir, 'predictor_stage2.pt'))
    
    aggressive_memory_cleanup()
    if use_wandb:
        wandb.finish()
    
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Best val stage1: {best_val_s1:.6f}, stage2: {best_val_s2:.6f}")
    print(f"Outputs: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
