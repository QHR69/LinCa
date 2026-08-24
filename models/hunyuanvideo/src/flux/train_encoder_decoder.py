"""
Training script for the invertible decomposition network.

Hybrid Glow + RevNet blocks, configurable partitions, fixed per-partition prediction orders.

Notes:
- no gating, which keeps the model simple
- configurable partitions via split_dims
- norm_factor is linear rather than squared
- the GPU memory leak is fixed
- prompts_per_epoch limits how many prompts each epoch samples
- intervals can be sampled at random per training sample
- assorted throughput optimisations
- tighter memory management
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

# No global pipeline cache: it is released after every eval to keep memory flat

# TensorBoard for monitoring
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    SummaryWriter = None  # fallback so type annotations don't fail
    TENSORBOARD_AVAILABLE = False
    print("Warning: tensorboard not available, training will proceed without logging")

# Wandb for monitoring
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Local imports
from flux.modules.invertible_net import (
    InvertibleDecompositionNet,
    LearnedDecompositionPredictor,
    FixedPredictionStrategy,
)
from flux.dataset import MemoryEfficientCacheDataset, collate_fn, create_dataloader


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
        # Let cuDNN pick the fastest algorithms
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # float32 matmul precision, PyTorch 2.0 and later
        if hasattr(torch, 'set_float32_matmul_precision'):
            torch.set_float32_matmul_precision('high')
        
        print("✓ Training optimizations enabled (cudnn.benchmark, high precision matmul)")


def aggressive_memory_cleanup():
    """Aggressive memory cleanup, covering both GPU and host memory."""
    # Several GC passes, to clear reference cycles
    gc.collect()
    gc.collect()
    gc.collect()
    
    # Release GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Return host memory to the OS via libc's malloc_trim, where available
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass  # not available everywhere, so ignore failures


def compute_loss(
    predictor: LearnedDecompositionPredictor,
    target_features: torch.Tensor,   # [batch, seq_len, 3072]
    cache_features: torch.Tensor,    # [batch, max_cache, seq_len, 3072]
    cache_mask: torch.Tensor,        # [batch, max_cache]
    step_distances: torch.Tensor,    # [batch, max_cache]
    intervals: torch.Tensor,         # [batch]
    z_loss_weight: float = 0.1,      # weight on the z0+z1+z2 loss (deprecated, kept for compatibility)
) -> Dict[str, torch.Tensor]:
    """
    Compute the training loss, vectorised across the batch.
    """
    batch_size, max_cache, seq_len, dim = cache_features.shape
    device = target_features.device

    # decompose runs in float32: the 1x1 convolution is a matrix product and float16 loses too much precision
    with torch.amp.autocast('cuda', enabled=False):
        target_f32 = target_features.float()
        cache_flat = cache_features.reshape(batch_size * max_cache, seq_len, dim).float()

        # Decompose the target feature: [batch, seq_len, 3072] -> 3 partitions
        target_z0, target_z1, target_z2 = predictor.decompose(target_f32)

        # Decompose every cached feature: [batch, max_cache, seq_len, 3072] -> 3 partitions
        c_z0, c_z1, c_z2 = predictor.decompose(cache_flat)
    # Reshape back: [batch, max_cache, seq_len, split_dim]
    c_z0 = c_z0.reshape(batch_size, max_cache, seq_len, -1)
    c_z1 = c_z1.reshape(batch_size, max_cache, seq_len, -1)
    c_z2 = c_z2.reshape(batch_size, max_cache, seq_len, -1)

    strategy = predictor.prediction_strategy

    # z0: zeroth order, reusing the nearest cache point
    pred_z0 = strategy.predict_z0(c_z0[:, 0])
    loss_z0 = F.mse_loss(pred_z0, target_z0)

    # z1: first order
    has_2 = (cache_mask.sum(dim=1) >= 2)  # [batch]
    if has_2.all():
        pred_z1 = strategy.predict_z1(
            c_z1[:, 0], c_z1[:, 1],
            step_distances[:, 0].float(), step_distances[:, 1].float()
        )
    elif has_2.any():
        # Mixed: first order where two points exist, zeroth order otherwise
        pred_z1_1 = strategy.predict_z1(
            c_z1[:, 0], c_z1[:, 1],
            step_distances[:, 0].float(), step_distances[:, 1].float()
        )
        pred_z1_0 = strategy.predict_z0(c_z1[:, 0])
        pred_z1 = torch.where(has_2[:, None, None], pred_z1_1, pred_z1_0)
    else:
        pred_z1 = strategy.predict_z0(c_z1[:, 0])
    loss_z1 = F.mse_loss(pred_z1, target_z1)

    # z2: second order
    has_3 = (cache_mask.sum(dim=1) >= 3)  # [batch]
    if has_3.all():
        pred_z2 = strategy.predict_z2(
            c_z2[:, 0], c_z2[:, 1], c_z2[:, 2],
            step_distances[:, 0].float(), step_distances[:, 1].float(), step_distances[:, 2].float()
        )
    elif has_2.all():
        # With at least two points: second order given three, first order given two
        pred_z2_2 = strategy.predict_z2(
            c_z2[:, 0], c_z2[:, 1], c_z2[:, 2],
            step_distances[:, 0].float(), step_distances[:, 1].float(), step_distances[:, 2].float()
        ) if has_3.any() else c_z2[:, 0]
        pred_z2_1 = strategy.predict_z1(
            c_z2[:, 0], c_z2[:, 1],
            step_distances[:, 0].float(), step_distances[:, 1].float()
        )
        if has_3.any():
            pred_z2 = torch.where(has_3[:, None, None], pred_z2_2, pred_z2_1)
        else:
            pred_z2 = pred_z2_1
    else:
        pred_z2 = strategy.predict_z0(c_z2[:, 0])
    loss_z2 = F.mse_loss(pred_z2, target_z2)

    # Normalise by the nearest distance
    norm_factor = step_distances[:, 0].float().clamp(min=1).mean()

    z0_loss = loss_z0 / norm_factor
    z1_loss = loss_z1 / norm_factor
    z2_loss = loss_z2 / norm_factor

    # Reconstruction loss: compose must be float32, or the 3072-dim matmul overflows in float16
    with torch.amp.autocast('cuda', enabled=False):
        pred_reconstructed = predictor.compose(pred_z0.float(), pred_z1.float(), pred_z2.float())
        loss_recon = F.mse_loss(pred_reconstructed, target_features.float())
    reconstruction_loss = loss_recon / norm_factor

    # Total loss: reconstruction at weight 1 plus the partition terms at 0.1
    total_loss = reconstruction_loss + z_loss_weight * (z0_loss + z1_loss + z2_loss)

    return {
        'total_loss': total_loss,
        'reconstruction_loss': reconstruction_loss,
        'z0_loss': z0_loss,
        'z1_loss': z1_loss,
        'z2_loss': z2_loss,
    }


def train_epoch(
    predictor: LearnedDecompositionPredictor,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    log_interval: int = 10,
    writer: Optional[SummaryWriter] = None,
    grad_accum_steps: int = 1,
    scaler: Optional[torch.amp.GradScaler] = None,
    use_amp: bool = False,
    z_loss_weight: float = 0.1,
    use_wandb: bool = False,
) -> Dict[str, float]:
    """
    Run one training epoch, with gradient accumulation and AMP.

    Fixes the memory leak, calls empty_cache less often, and logs to TensorBoard.
    """
    predictor.train()
    
    total_loss_sum = 0.0
    recon_loss_sum = 0.0
    z0_loss_sum = 0.0
    z1_loss_sum = 0.0
    z2_loss_sum = 0.0
    num_batches = 0

    # Accumulators for gradient accumulation
    accum_loss = 0.0
    accum_recon = 0.0
    accum_z0 = 0.0
    accum_z1 = 0.0
    accum_z2 = 0.0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    optimizer.zero_grad(set_to_none=True)  # set_to_none frees the gradient buffers
    
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        target_features = batch['target_features'].to(device, non_blocking=True)
        cache_features = batch['cache_features'].to(device, non_blocking=True)
        cache_mask = batch['cache_mask'].to(device, non_blocking=True)
        step_distances = batch['step_distances'].to(device, non_blocking=True)
        intervals = batch['intervals'].to(device, non_blocking=True)
        
        # Mixed precision via AMP
        with torch.amp.autocast('cuda', enabled=use_amp):
            losses = compute_loss(
                predictor, target_features, cache_features,
                cache_mask, step_distances, intervals,
                z_loss_weight=z_loss_weight,
            )
            loss = losses['total_loss'] / grad_accum_steps
        
        # Backward, scaled when AMP is on
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Accumulate for reporting; .detach().item() keeps the graph from being retained
        accum_loss += losses['total_loss'].detach().item()
        accum_recon += losses['reconstruction_loss'].detach().item()
        accum_z0 += losses['z0_loss'].detach().item()
        accum_z1 += losses['z1_loss'].detach().item()
        accum_z2 += losses['z2_loss'].detach().item()

        # Step the optimiser every grad_accum_steps batches
        if (batch_idx + 1) % grad_accum_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(predictor.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(predictor.parameters(), max_norm=1.0)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
            # Report the mean loss
            avg_accum_loss = accum_loss / grad_accum_steps
            avg_accum_recon = accum_recon / grad_accum_steps
            
            total_loss_sum += accum_loss
            recon_loss_sum += accum_recon
            z0_loss_sum += accum_z0
            z1_loss_sum += accum_z1
            z2_loss_sum += accum_z2
            num_batches += grad_accum_steps

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{avg_accum_loss:.6f}",
                'recon': f"{avg_accum_recon:.6f}",
            })

            # Log to TensorBoard
            if writer is not None and (batch_idx + 1) % (log_interval * grad_accum_steps) == 0:
                global_step = epoch * len(dataloader) + batch_idx
                writer.add_scalar('train/batch_total_loss', avg_accum_loss, global_step)
                writer.add_scalar('train/batch_recon_loss', avg_accum_recon, global_step)
                writer.add_scalar('train/batch_z0_loss', accum_z0 / grad_accum_steps, global_step)
                writer.add_scalar('train/batch_z1_loss', accum_z1 / grad_accum_steps, global_step)
                writer.add_scalar('train/batch_z2_loss', accum_z2 / grad_accum_steps, global_step)

            # Log to wandb
            if use_wandb and WANDB_AVAILABLE and (batch_idx + 1) % (log_interval * grad_accum_steps) == 0:
                global_step = epoch * len(dataloader) + batch_idx
                wandb.log({
                    'train/batch_total_loss': avg_accum_loss,
                    'train/batch_recon_loss': avg_accum_recon,
                    'train/batch_z0_loss': accum_z0 / grad_accum_steps,
                    'train/batch_z1_loss': accum_z1 / grad_accum_steps,
                    'train/batch_z2_loss': accum_z2 / grad_accum_steps,
                    'train/global_step': global_step,
                }, step=global_step)

            # Reset the accumulators
            accum_loss = 0.0
            accum_recon = 0.0
            accum_z0 = 0.0
            accum_z1 = 0.0
            accum_z2 = 0.0

    # Flush a trailing partial accumulation group
    remaining = len(dataloader) % grad_accum_steps
    if remaining > 0 and accum_loss > 0:
        if scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), max_norm=1.0)
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        
        total_loss_sum += accum_loss
        recon_loss_sum += accum_recon
        z0_loss_sum += accum_z0
        z1_loss_sum += accum_z1
        z2_loss_sum += accum_z2
        num_batches += remaining
    
    # Average losses
    if num_batches == 0:
        num_batches = 1
    
    avg_losses = {
        'total_loss': total_loss_sum / num_batches,
        'reconstruction_loss': recon_loss_sum / num_batches,
        'z0_loss': z0_loss_sum / num_batches,
        'z1_loss': z1_loss_sum / num_batches,
        'z2_loss': z2_loss_sum / num_batches,
    }
    
    return avg_losses


@torch.no_grad()
def validate(
    predictor: LearnedDecompositionPredictor,
    dataloader: DataLoader,
    device: torch.device,
    z_loss_weight: float = 0.1,
) -> Dict[str, float]:
    """Run validation."""
    predictor.eval()

    total_loss_sum = 0.0
    recon_loss_sum = 0.0
    z0_loss_sum = 0.0
    z1_loss_sum = 0.0
    z2_loss_sum = 0.0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Validation"):
        target_features = batch['target_features'].to(device, non_blocking=True)
        cache_features = batch['cache_features'].to(device, non_blocking=True)
        cache_mask = batch['cache_mask'].to(device, non_blocking=True)
        step_distances = batch['step_distances'].to(device, non_blocking=True)
        intervals = batch['intervals'].to(device, non_blocking=True)

        losses = compute_loss(
            predictor, target_features, cache_features,
            cache_mask, step_distances, intervals,
            z_loss_weight=z_loss_weight,
        )

        total_loss_sum += losses['total_loss'].item()
        recon_loss_sum += losses['reconstruction_loss'].item()
        z0_loss_sum += losses['z0_loss'].item()
        z1_loss_sum += losses['z1_loss'].item()
        z2_loss_sum += losses['z2_loss'].item()
        num_batches += 1

        # Cleanup
        del target_features, cache_features, cache_mask, step_distances, intervals
        del losses

    return {
        'val_total_loss': total_loss_sum / max(num_batches, 1),
        'val_recon_loss': recon_loss_sum / max(num_batches, 1),
        'val_z0_loss': z0_loss_sum / max(num_batches, 1),
        'val_z1_loss': z1_loss_sum / max(num_batches, 1),
        'val_z2_loss': z2_loss_sum / max(num_batches, 1),
    }


@torch.no_grad()
def eval_generate_images(*args, **kwargs) -> List[Image.Image]:
    """No-op placeholder.

    Mid-training sample previews are not wired up for this backbone: rendering a
    sample would need the full generation pipeline loaded alongside the trainer.
    Train the predictor here, then sample with the dedicated script.
    """
    print("Skipping mid-training sample generation (not supported for this backbone).")
    return []


def save_checkpoint(
    predictor: LearnedDecompositionPredictor,
    optimizer: optim.Optimizer,
    epoch: int,
    loss: float,
    save_path: str,
):
    """Save a checkpoint."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': predictor.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': {
            'dim': predictor.dim,
            'num_blocks': predictor.num_blocks,
            'hidden_dim': predictor.hidden_dim,
            'split_dims': predictor.split_dims,
            'dropout': predictor.dropout,
        }
    }, save_path)
    print(f"Saved checkpoint to {save_path}")


def load_checkpoint(
    predictor: LearnedDecompositionPredictor,
    optimizer: optim.Optimizer,
    load_path: str,
    device: torch.device,
) -> int:
    """Load a checkpoint and return the epoch it was saved at."""
    checkpoint = torch.load(load_path, map_location=device)
    predictor.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch']


def main():
    parser = argparse.ArgumentParser(description="Train hybrid invertible decomposition network")
    
    # Data
    parser.add_argument('--data_dir', type=str, 
                        default='./cache_data/hunyuan',
                        help='Path to cache data directory')
    parser.add_argument('--train_prompts', type=str, default='0-400',
                        help='Prompt indices for training (e.g., "0-400")')
    parser.add_argument('--val_sample_range', type=str, default='200-400',
                        help='Range to sample validation prompts from')
    parser.add_argument('--num_val_prompts', type=int, default=20,
                        help='Number of validation prompts')
    parser.add_argument('--prompts_per_epoch', type=int, default=30,
                        help='Number of prompts to sample per epoch (0=use all)')
    parser.add_argument('--shuffle_prompts', action='store_true',
                        help='Shuffle prompts each epoch')
    parser.add_argument('--no_val', action='store_true',
                        help='Skip validation, use all train prompts, save best by train loss')

    # Predictor architecture
    parser.add_argument('--dim', type=int, default=3072, help='Feature dimension')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension for RevNet blocks')
    parser.add_argument('--num_blocks', type=int, default=6, help='Number of hybrid blocks')
    parser.add_argument('--split_dims', type=str, default='1024,1024,1024',
                        help='Split dimensions for decomposition (comma-separated)')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    # Training
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, 
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--z_loss_weight', type=float, default=0.1,
                        help='Weight for z0+z1+z2 loss (deprecated, kept for compatibility)')
    parser.add_argument('--intervals', type=str, default='6,7,8,9,10',
                        help='Intervals to train on (comma-separated)')
    parser.add_argument('--random_interval', action='store_true',
                        help='Randomly select interval for each sample')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--subsample_seq', type=int, default=0,
                        help='Subsample seq_len per sample (0=use all). Reduces I/O and memory.')
    parser.add_argument('--grad_accum_steps', type=int, default=5,
                        help='Gradient accumulation steps')
    parser.add_argument('--early_stop_patience', type=int, default=10,
                        help='Early stopping patience (0=disable)')
    
    # Eval during training
    parser.add_argument('--eval_prompts_file', type=str, 
                        default='prompts/eval.txt',
                        help='Path to eval prompts file')
    parser.add_argument('--eval_interval', type=int, default=5,
                        help='Run eval every N epochs (0 to disable)')
    parser.add_argument('--model_path', type=str,
                        default="Qwen/Qwen-Image",
                        help='Path to Qwen-Image model for eval')
    
    # Logging
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Output directory for checkpoints and logs')
    parser.add_argument('--exp_name', type=str, default=None,
                        help='Experiment name (default: auto-generated)')
    parser.add_argument('--log_interval', type=int, default=1,
                        help='Log to wandb every N batches')
    parser.add_argument('--save_interval', type=int, default=5,
                        help='Save checkpoint every N epochs')

    # Wandb
    parser.add_argument('--wandb_project', type=str, default='linca-flux',
                        help='Wandb project name')
    parser.add_argument('--wandb_key', type=str, default=None,
                        help='Wandb API key (or set WANDB_API_KEY env var)')
    parser.add_argument('--no_wandb', action='store_true', help='Disable wandb logging')

    # Resume
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    # Misc
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--amp', action='store_true', help='Use automatic mixed precision')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    
    args = parser.parse_args()
    
    # Apply the throughput settings
    setup_training_optimizations()
    
    # Parse intervals
    intervals = [int(x) for x in args.intervals.split(',')]
    
    # Parse prompt indices
    def parse_indices(s):
        if '-' in s:
            start, end = map(int, s.split('-'))
            return list(range(start, end))
        else:
            return [int(x) for x in s.split(',')]
    
    all_train_indices = parse_indices(args.train_prompts)

    if args.no_val:
        # No-validation mode: every prompt is used for training
        val_indices = []
        train_pool = all_train_indices
    else:
        # Take the validation prompts at regular spacing across the range
        num_val = min(args.num_val_prompts, len(all_train_indices))
        total = len(all_train_indices)
        # Pick num_val positions at even spacing
        val_positions = [int(i * total / num_val) for i in range(num_val)]
        val_indices = sorted([all_train_indices[p] for p in val_positions])
        val_set = set(val_indices)
        # The training pool is everything left after the validation prompts
        train_pool = [idx for idx in all_train_indices if idx not in val_set]

    print(f"\n{'='*60}")
    print(f"Data split:")
    print(f"  prompt range:      {args.train_prompts}")
    if args.no_val:
        print(f"  validation:        skipped (--no_val)")
    else:
        print(f"  validation prompts: {len(val_indices)} (evenly spaced across the range)")
    print(f"  training pool:     {len(train_pool)}")
    print(f"  per epoch:         {args.prompts_per_epoch if args.prompts_per_epoch > 0 else 'all'}")
    print(f"  Intervals: {intervals}")
    print(f"  random interval:   {'yes' if args.random_interval else 'no'}")
    print(f"{'='*60}\n")
    
    # Set seed
    set_seed(args.seed)
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Parse split_dims
    split_dims = [int(x) for x in args.split_dims.split(',')]
    
    # Create the predictor: hybrid invertible blocks, configurable partitions, fixed prediction orders
    predictor = LearnedDecompositionPredictor(
        dim=args.dim,
        num_blocks=args.num_blocks,
        hidden_dim=args.hidden_dim,
        split_dims=split_dims,
        dropout=args.dropout,
    ).to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in predictor.parameters())
    print(f"Model parameters: {num_params:,} ({num_params/1e6:.1f}M)")
    
    # Optimizer
    optimizer = optim.AdamW(
        predictor.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    
    # AMP Scaler
    scaler = torch.amp.GradScaler('cuda') if args.amp else None
    if args.amp:
        print("Using Automatic Mixed Precision (AMP)")
    
    # Resume if specified
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(predictor, optimizer, args.resume, device)
        print(f"Resumed from epoch {start_epoch}")
    
    # The validation DataLoader is fixed, and only built when there is a validation set
    subsample = args.subsample_seq if args.subsample_seq > 0 else None
    val_loader = None
    if not args.no_val:
        val_loader = create_dataloader(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            intervals=intervals,
            num_workers=args.num_workers,
            shuffle=False,
            prompt_indices=val_indices,
            subsample_seq=subsample,
            random_interval=args.random_interval,
        )
        print(f"Validation dataset: {len(val_indices)} prompts, {len(val_loader.dataset)} samples")
    else:
        print("Validation: skipped (--no_val)")
    
    # Experiment name
    if args.exp_name is None:
        args.exp_name = f"hybrid_without_gate"

    # Output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"{args.exp_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n📁 Output directory: {output_dir}")

    # Save config
    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Copy run_train.sh into the output directory, to record the configuration used
    import shutil
    script_dir = os.path.dirname(os.path.abspath(__file__))
    run_train_candidates = [
        os.path.join(script_dir, '..', '..', 'run_train.sh'),
        os.path.join(os.getcwd(), 'run_train.sh'),
    ]
    for run_train_path in run_train_candidates:
        run_train_path = os.path.normpath(run_train_path)
        if os.path.exists(run_train_path):
            shutil.copy2(run_train_path, os.path.join(output_dir, 'run_train.sh'))
            print(f"📄 Copied run_train.sh to {output_dir}")
            break

    # Initialize TensorBoard
    writer = None
    if TENSORBOARD_AVAILABLE:
        tensorboard_dir = os.path.join(output_dir, 'tensorboard')
        writer = SummaryWriter(log_dir=tensorboard_dir)
        print(f"📊 TensorBoard logs: {tensorboard_dir}")
        print(f"   Run: tensorboard --logdir={tensorboard_dir}")
    else:
        print("⚠ TensorBoard not available, skipping logging")

    # Initialize wandb
    use_wandb = WANDB_AVAILABLE and not args.no_wandb
    if use_wandb:
        if args.wandb_key:
            wandb.login(key=args.wandb_key)
        elif os.environ.get('WANDB_API_KEY'):
            wandb.login(key=os.environ.get('WANDB_API_KEY'))

        wandb.init(
            project=args.wandb_project,
            name=f"{args.exp_name}_{timestamp}",
            config=vars(args),
        )
        wandb.watch(predictor, log='all', log_freq=100)
        print(f"📊 Wandb project: {args.wandb_project}")
    else:
        if not WANDB_AVAILABLE:
            print("⚠ wandb not installed, skipping wandb logging")
        else:
            print("⚠ wandb disabled (--no_wandb)")

    # Training loop
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*60}")

        # === Resample the training prompts each epoch ===
        if args.prompts_per_epoch > 0 and args.prompts_per_epoch < len(train_pool):
            epoch_train_indices = random.sample(train_pool, args.prompts_per_epoch)
        else:
            epoch_train_indices = train_pool.copy()
            if args.shuffle_prompts:
                random.shuffle(epoch_train_indices)

        print(f"Training on {len(epoch_train_indices)} prompts this epoch")

        # Build this epoch's training DataLoader
        train_loader = create_dataloader(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            intervals=intervals,
            num_workers=args.num_workers,
            shuffle=True,
            prompt_indices=epoch_train_indices,
            subsample_seq=subsample,
            random_interval=args.random_interval,
        )
        print(f"Training dataset: {len(train_loader.dataset)} samples")

        # Train
        train_losses = train_epoch(
            predictor, train_loader, optimizer, device,
            epoch, args.log_interval, writer, args.grad_accum_steps,
            scaler=scaler, use_amp=args.amp,
            z_loss_weight=args.z_loss_weight,
            use_wandb=use_wandb,
        )

        # Release this epoch's dataloader
        del train_loader

        # Validate, when there is a validation set
        val_losses = None
        if val_loader is not None:
            val_losses = validate(predictor, val_loader, device,
                                  z_loss_weight=args.z_loss_weight)

        # Update scheduler
        scheduler.step()

        # Log
        print(f"\n--- Epoch {epoch + 1} Summary ---")
        print(f"Train Loss: {train_losses['total_loss']:.6f}")
        print(f"  Reconstruction: {train_losses['reconstruction_loss']:.6f}")
        print(f"  Z0: {train_losses['z0_loss']:.6f}, Z1: {train_losses['z1_loss']:.6f}, Z2: {train_losses['z2_loss']:.6f}")
        if val_losses is not None:
            print(f"Val Loss: {val_losses['val_total_loss']:.6f}")
            print(f"  Reconstruction: {val_losses['val_recon_loss']:.6f}")
        print(f"LR: {scheduler.get_last_lr()[0]:.6f}")

        # Report GPU memory use
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

        # Log to TensorBoard
        if writer is not None:
            writer.add_scalar('train/epoch_total_loss', train_losses['total_loss'], epoch + 1)
            writer.add_scalar('train/epoch_recon_loss', train_losses['reconstruction_loss'], epoch + 1)
            writer.add_scalar('train/epoch_z0_loss', train_losses['z0_loss'], epoch + 1)
            writer.add_scalar('train/epoch_z1_loss', train_losses['z1_loss'], epoch + 1)
            writer.add_scalar('train/epoch_z2_loss', train_losses['z2_loss'], epoch + 1)
            if val_losses is not None:
                writer.add_scalar('val/total_loss', val_losses['val_total_loss'], epoch + 1)
                writer.add_scalar('val/recon_loss', val_losses['val_recon_loss'], epoch + 1)
                writer.add_scalar('val/z0_loss', val_losses['val_z0_loss'], epoch + 1)
                writer.add_scalar('val/z1_loss', val_losses['val_z1_loss'], epoch + 1)
                writer.add_scalar('val/z2_loss', val_losses['val_z2_loss'], epoch + 1)
            writer.add_scalar('lr', scheduler.get_last_lr()[0], epoch + 1)
            if torch.cuda.is_available():
                writer.add_scalar('gpu_memory_gb', torch.cuda.memory_allocated() / 1024**3, epoch + 1)

        # Log to wandb
        if use_wandb:
            log_dict = {
                'epoch': epoch + 1,
                'train/epoch_total_loss': train_losses['total_loss'],
                'train/epoch_recon_loss': train_losses['reconstruction_loss'],
                'train/epoch_z0_loss': train_losses['z0_loss'],
                'train/epoch_z1_loss': train_losses['z1_loss'],
                'train/epoch_z2_loss': train_losses['z2_loss'],
                'lr': scheduler.get_last_lr()[0],
            }
            if val_losses is not None:
                log_dict.update({
                    'val/total_loss': val_losses['val_total_loss'],
                    'val/recon_loss': val_losses['val_recon_loss'],
                    'val/z0_loss': val_losses['val_z0_loss'],
                    'val/z1_loss': val_losses['val_z1_loss'],
                    'val/z2_loss': val_losses['val_z2_loss'],
                })
            if torch.cuda.is_available():
                log_dict['gpu_memory_gb'] = torch.cuda.memory_allocated() / 1024**3
            wandb.log(log_dict)

        # Pick the loss that drives early stopping and best-model tracking
        current_loss = val_losses['val_total_loss'] if val_losses is not None else train_losses['total_loss']

        # Early stopping check
        if current_loss < best_loss:
            best_loss = current_loss
            patience_counter = 0

            # Save best model
            save_checkpoint(
                predictor, optimizer, epoch + 1, best_loss,
                os.path.join(output_dir, 'best_model.pt')
            )
            predictor.save_pretrained(os.path.join(output_dir, 'best_predictor.pt'))
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epochs (patience: {args.early_stop_patience})")

            if args.early_stop_patience > 0 and patience_counter >= args.early_stop_patience:
                print(f"\n>>> Early stopping triggered!")
                break
        
        # Save periodic checkpoint
        if (epoch + 1) % args.save_interval == 0:
            save_checkpoint(
                predictor, optimizer, epoch + 1, train_losses['total_loss'],
                os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pt')
            )
        
        # Eval: generate images (disabled during training)
        if args.eval_interval > 0 and (epoch + 1) % args.eval_interval == 0:
            try:
                images = eval_generate_images(
                    predictor=predictor,
                    eval_prompts_file=args.eval_prompts_file,
                    output_dir=output_dir,
                    model_path=args.model_path,
                    interval=6,
                    device=args.device,
                    epoch=epoch + 1,
                    base_seed=args.seed,
                )

                # Cleanup
                del images
                aggressive_memory_cleanup()

            except Exception as e:
                print(f"Warning: Eval failed with error: {e}")
                import traceback
                traceback.print_exc()
    
    # Save final model
    save_checkpoint(
        predictor, optimizer, args.epochs, train_losses['total_loss'],
        os.path.join(output_dir, 'final_model.pt')
    )
    predictor.save_pretrained(os.path.join(output_dir, 'predictor.pt'))
    
    # Final cleanup
    aggressive_memory_cleanup()

    # Close TensorBoard writer
    if writer is not None:
        writer.close()
        print(f"\n📊 TensorBoard logs saved")

    # Close wandb
    if use_wandb:
        wandb.finish()
        print(f"📊 Wandb run finished")

    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Best {'train' if args.no_val else 'validation'} loss: {best_loss:.6f}")
    print(f"Outputs saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
