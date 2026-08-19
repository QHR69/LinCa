"""
Multi-GPU (DDP) training for qwen_edit LinCA two-stage invertible decomposition network

与 train.py 逻辑完全一致，仅增加 DDP 以加速训练。
Usage: torchrun --nproc_per_node=N train_ddp.py [args...]
"""

import os
import sys
import argparse
import json
import random
import gc
import ctypes
from datetime import timedelta
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image

_SCRIPT_DIR = Path(__file__).resolve().parent
_LINCA_ROOT = _SCRIPT_DIR.parent.parent.parent
_PIPELINE_BASE = _LINCA_ROOT / "freqca_qwen"  # pipeline dependency under the LinCA root
if str(_PIPELINE_BASE) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_BASE))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from invertible_net import LearnedDecompositionPredictor, FixedPredictionStrategy
from dataset import MemoryEfficientCacheDataset, collate_fn
from data_splits import build_splits, save_splits

# 复用 train.py 中的函数
from train import (
    set_seed,
    setup_training_optimizations,
    aggressive_memory_cleanup,
    compute_loss,
    train_epoch,
    _thorough_cleanup_pipeline,
    _load_single_display_item,
    eval_generate_images_edit,
    STAGE1_MAX_STEP,
    STAGE2_MIN_STEP,
)


def _get_raw_model(model):
    """获取 DDP 包装下的原始模型"""
    return model.module if hasattr(model, 'module') else model


class _DDPPredictorWrapper:
    """包装 DDP 模型，暴露 decompose/compose 供 compute_loss 使用（梯度仍流向 DDP 参数）"""

    def __init__(self, ddp_model):
        self._ddp = ddp_model

    @property
    def prediction_strategy(self):
        return self._ddp.module.prediction_strategy

    def decompose(self, x):
        return self._ddp.module.decompose(x)

    def compose(self, z0, z1, z2):
        return self._ddp.module.compose(z0, z1, z2)

    def parameters(self):
        return self._ddp.parameters()

    def train(self, mode=True):
        self._ddp.train(mode)
        return self


def save_checkpoint_ddp(
    predictor_stage1,
    predictor_stage2,
    optimizer1: optim.Optimizer,
    optimizer2: optim.Optimizer,
    epoch: int,
    loss_s1: float,
    loss_s2: float,
    save_path: str,
):
    """保存 checkpoint，支持 DDP"""
    raw_s1 = _get_raw_model(predictor_stage1)
    raw_s2 = _get_raw_model(predictor_stage2)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'predictor_stage1': raw_s1.state_dict(),
        'predictor_stage2': raw_s2.state_dict(),
        'optimizer1': optimizer1.state_dict(),
        'optimizer2': optimizer2.state_dict(),
        'loss_stage1': loss_s1,
        'loss_stage2': loss_s2,
        'config': {
            'dim': raw_s1.dim,
            'num_blocks': raw_s1.num_blocks,
            'hidden_dim': raw_s1.hidden_dim,
            'split_dims': raw_s1.split_dims,
            'dropout': raw_s1.dropout,
        }
    }, save_path)
    if dist.get_rank() == 0:
        print(f"Saved checkpoint to {save_path}")


def load_checkpoint_ddp(
    predictor_stage1,
    predictor_stage2,
    optimizer1: optim.Optimizer,
    optimizer2: optim.Optimizer,
    load_path: str,
    device: torch.device,
) -> int:
    """加载 checkpoint，支持 DDP"""
    checkpoint = torch.load(load_path, map_location=device)
    raw_s1 = _get_raw_model(predictor_stage1)
    raw_s2 = _get_raw_model(predictor_stage2)
    raw_s1.load_state_dict(checkpoint['predictor_stage1'])
    raw_s2.load_state_dict(checkpoint['predictor_stage2'])
    optimizer1.load_state_dict(checkpoint['optimizer1'])
    optimizer2.load_state_dict(checkpoint['optimizer2'])
    return checkpoint['epoch']


def validate_ddp(
    predictor_stage1,
    predictor_stage2,
    dataloader: DataLoader,
    device: torch.device,
    z_loss_weight: float = 0.1,
) -> Dict[str, float]:
    """DDP 验证：各 rank 计算本地 loss 后 all-reduce 得到全局平均"""
    predictor_stage1.eval()
    predictor_stage2.eval()

    total_s1 = {'loss': 0.0, 'recon': 0.0, 'z0': 0.0, 'z1': 0.0, 'z2': 0.0}
    total_s2 = {'loss': 0.0, 'recon': 0.0, 'z0': 0.0, 'z1': 0.0, 'z2': 0.0}
    count_s1 = 0
    count_s2 = 0

    for batch in tqdm(dataloader, desc="Validation", disable=(dist.get_rank() != 0)):
        target_features = batch['target_features'].to(device, non_blocking=True)
        cache_features = batch['cache_features'].to(device, non_blocking=True)
        cache_mask = batch['cache_mask'].to(device, non_blocking=True)
        step_distances = batch['step_distances'].to(device, non_blocking=True)
        intervals = batch['intervals'].to(device, non_blocking=True)
        target_steps = batch['target_steps'].to(device, non_blocking=True)

        stage1_mask = (target_steps <= STAGE1_MAX_STEP)
        stage2_mask = (target_steps >= STAGE2_MIN_STEP)

        if stage1_mask.any():
            wrap_s1 = _DDPPredictorWrapper(predictor_stage1)
            losses_s1 = compute_loss(
                wrap_s1,
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
            wrap_s2 = _DDPPredictorWrapper(predictor_stage2)
            losses_s2 = compute_loss(
                wrap_s2,
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

    # All-reduce 得到全局 sum 和 count
    buf = torch.tensor([
        total_s1['loss'], total_s1['recon'], total_s1['z0'], total_s1['z1'], total_s1['z2'],
        total_s2['loss'], total_s2['recon'], total_s2['z0'], total_s2['z1'], total_s2['z2'],
        float(count_s1), float(count_s2),
    ], device=device, dtype=torch.float64)
    dist.all_reduce(buf, op=dist.ReduceOp.SUM)

    n1 = max(int(buf[10].item()), 1)
    n2 = max(int(buf[11].item()), 1)
    return {
        'val_total_loss_stage1': buf[0].item() / n1,
        'val_recon_loss_stage1': buf[1].item() / n1,
        'val_z0_loss_stage1': buf[2].item() / n1,
        'val_z1_loss_stage1': buf[3].item() / n1,
        'val_z2_loss_stage1': buf[4].item() / n1,
        'val_total_loss_stage2': buf[5].item() / n2,
        'val_recon_loss_stage2': buf[6].item() / n2,
        'val_z0_loss_stage2': buf[7].item() / n2,
        'val_z1_loss_stage2': buf[8].item() / n2,
        'val_z2_loss_stage2': buf[9].item() / n2,
    }


def create_dataloader_ddp(
    data_dir: str,
    batch_size: int,
    intervals: List[int],
    sample_indices: List[int],
    num_workers: int,
    random_interval: bool,
    shuffle: bool,
    seed: int,
    epoch: int = 0,
) -> DataLoader:
    """创建带 DistributedSampler 的 DataLoader"""
    dataset = MemoryEfficientCacheDataset(
        data_dir=data_dir,
        intervals=intervals,
        sample_indices=sample_indices,
        subsample_seq=None,
        random_interval=random_interval,
    )
    sampler = DistributedSampler(dataset, shuffle=shuffle, seed=seed)
    sampler.set_epoch(epoch)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=False,
    )


def main():
    parser = argparse.ArgumentParser(description="Train qwen_edit two-stage LinCA (DDP multi-GPU)")

    parser.add_argument('--cache_data_dir', type=str, default='./cache_data/qwen_edit')
    parser.add_argument('--dataset_path', type=str, default='./data/gedit_bench')
    parser.add_argument('--prompts_per_epoch', type=int, default=40)
    parser.add_argument('--shuffle_prompts', action='store_true')

    parser.add_argument('--dim', type=int, default=3072)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--num_blocks', type=int, default=6)
    parser.add_argument('--split_dims', type=str, default='1024,1024,1024')
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
    parser.add_argument('--early_stop_patience', type=int, default=10)

    parser.add_argument('--eval_interval', type=int, default=1)
    parser.add_argument('--model_path', type=str, default="Qwen/Qwen-Image-Edit")

    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--log_interval', type=int, default=1)
    parser.add_argument('--save_interval', type=int, default=5)
    parser.add_argument('--wandb_project', type=str, default='qwen_edit_linca_two_stage')
    parser.add_argument('--wandb_key', type=str, default=None)
    parser.add_argument('--no_wandb', action='store_true')

    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--amp', action='store_true')

    args = parser.parse_args()

    # eval 仅 rank0 跑（生成 11 张图约 10–15 分钟），其他 rank 在 barrier 处等待；NCCL 默认 10 分钟会超时，必须设长
    _pg_timeout = timedelta(minutes=120)
    dist.init_process_group("nccl", timeout=_pg_timeout)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(rank)

    if rank == 0:
        setup_training_optimizations()
        print(f">>> DDP training on {world_size} GPUs (process group timeout: {_pg_timeout})")

    intervals = [int(x) for x in args.intervals.split(',')]
    val_indices, display_indices, train_indices = build_splits(args.cache_data_dir)

    if args.exp_name is None:
        args.exp_name = "qwen_edit_two_stage_v4_ddp"
    output_dir = os.path.join(args.output_dir, args.wandb_project, args.exp_name)
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        save_splits(output_dir, val_indices, display_indices, train_indices)
    dist.barrier()

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"两阶段训练 (DDP): Stage1=0-24, Stage2=25-49")
        print(f"Early stopping B: stop only after both stages stall for {args.early_stop_patience} epochs")
        print(f"数据划分: 训练 {len(train_indices)}, 验证 {len(val_indices)}, 展示 {len(display_indices)}")
        print(f"{'='*60}\n")

    set_seed(args.seed)
    split_dims = [int(x) for x in args.split_dims.split(',')]

    predictor_stage1 = LearnedDecompositionPredictor(
        dim=args.dim, num_blocks=args.num_blocks, hidden_dim=args.hidden_dim,
        split_dims=split_dims, dropout=args.dropout,
    ).to(device)
    predictor_stage2 = LearnedDecompositionPredictor(
        dim=args.dim, num_blocks=args.num_blocks, hidden_dim=args.hidden_dim,
        split_dims=split_dims, dropout=args.dropout,
    ).to(device)

    predictor_stage1 = DDP(predictor_stage1, device_ids=[rank])
    predictor_stage2 = DDP(predictor_stage2, device_ids=[rank])

    if rank == 0:
        num_params = sum(p.numel() for p in _get_raw_model(predictor_stage1).parameters())
        print(f"Model parameters per stage: {num_params:,} ({num_params/1e6:.1f}M)")

    # 学习率按 world_size 线性缩放，保持与单卡等效
    lr_scaled = args.lr * world_size
    optimizer1 = optim.AdamW(predictor_stage1.parameters(), lr=lr_scaled, weight_decay=args.weight_decay)
    optimizer2 = optim.AdamW(predictor_stage2.parameters(), lr=lr_scaled, weight_decay=args.weight_decay)
    scheduler1 = optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=args.epochs, eta_min=lr_scaled * 0.01)
    scheduler2 = optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=args.epochs, eta_min=lr_scaled * 0.01)

    scaler = torch.amp.GradScaler('cuda') if args.amp else None
    if rank == 0 and args.amp:
        print("Using AMP")

    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint_ddp(
            predictor_stage1, predictor_stage2,
            optimizer1, optimizer2,
            args.resume, device
        )
        if rank == 0:
            print(f"Resumed from epoch {start_epoch}")

    if rank == 0:
        with open(os.path.join(output_dir, 'config.json'), 'w') as f:
            cfg = vars(args).copy()
            cfg['lr_scaled'] = lr_scaled
            cfg['world_size'] = world_size
            json.dump(cfg, f, indent=2)

    use_wandb = (rank == 0) and WANDB_AVAILABLE and not args.no_wandb
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
        set_seed(args.seed + epoch)
        if args.prompts_per_epoch > 0 and args.prompts_per_epoch < len(train_indices):
            epoch_train_indices = random.sample(train_indices, args.prompts_per_epoch)
        else:
            epoch_train_indices = train_indices.copy()
            if args.shuffle_prompts:
                random.shuffle(epoch_train_indices)

        train_loader = create_dataloader_ddp(
            data_dir=args.cache_data_dir,
            batch_size=args.batch_size,
            intervals=intervals,
            sample_indices=epoch_train_indices,
            num_workers=args.num_workers,
            random_interval=args.random_interval,
            shuffle=True,
            seed=args.seed + epoch,
            epoch=epoch,
        )

        val_loader = create_dataloader_ddp(
            data_dir=args.cache_data_dir,
            batch_size=args.batch_size,
            intervals=intervals,
            sample_indices=val_indices,
            num_workers=args.num_workers,
            random_interval=args.random_interval,
            shuffle=False,
            seed=args.seed,
            epoch=0,
        )

        if rank == 0:
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{args.epochs}")
            print(f"{'='*60}")

        wrap_s1 = _DDPPredictorWrapper(predictor_stage1)
        wrap_s2 = _DDPPredictorWrapper(predictor_stage2)
        train_losses = train_epoch(
            wrap_s1, wrap_s2, train_loader,
            optimizer1, optimizer2, device, epoch, args.log_interval, use_wandb,
            args.grad_accum_steps, scaler=scaler, use_amp=args.amp,
            z_loss_weight=args.z_loss_weight,
        )

        del train_loader
        aggressive_memory_cleanup()

        val_losses = validate_ddp(
            predictor_stage1, predictor_stage2, val_loader, device,
            z_loss_weight=args.z_loss_weight,
        )

        scheduler1.step()
        scheduler2.step()

        if rank == 0:
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
            save_checkpoint_ddp(
                predictor_stage1, predictor_stage2,
                optimizer1, optimizer2,
                epoch + 1, best_val_s1, best_val_s2,
                os.path.join(output_dir, 'best_model.pt')
            )
            if rank == 0:
                raw_s1 = _get_raw_model(predictor_stage1)
                raw_s2 = _get_raw_model(predictor_stage2)
                raw_s1.save_pretrained(os.path.join(output_dir, 'best_predictor_stage1.pt'))
                raw_s2.save_pretrained(os.path.join(output_dir, 'best_predictor_stage2.pt'))

        if rank == 0:
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
            if rank == 0:
                print(f"\n>>> Early stopping triggered!")
            break

        if (epoch + 1) % args.save_interval == 0:
            save_checkpoint_ddp(
                predictor_stage1, predictor_stage2,
                optimizer1, optimizer2,
                epoch + 1,
                train_losses['total_loss_stage1'], train_losses['total_loss_stage2'],
                os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pt')
            )

        if args.eval_interval > 0 and (epoch + 1) % args.eval_interval == 0 and rank == 0:
            try:
                raw_s1 = _get_raw_model(predictor_stage1)
                raw_s2 = _get_raw_model(predictor_stage2)
                display_output_dir = os.path.join(output_dir, f'eval_epoch_{epoch+1}')
                images = eval_generate_images_edit(
                    predictor_stage1=raw_s1,
                    predictor_stage2=raw_s2,
                    display_sample_indices=display_indices,
                    display_output_dir=display_output_dir,
                    model_path=args.model_path,
                    dataset_path=args.dataset_path,
                    cache_data_dir=args.cache_data_dir,
                    interval=6,
                    device=str(device),
                    epoch=epoch + 1,
                    base_seed=args.seed,
                )
                if use_wandb and images:
                    wandb.log({
                        f'eval/images_epoch_{epoch+1}': [
                            wandb.Image(img, caption=f"eval_{i}") for i, img in enumerate(images)
                        ]
                    })
                del images
                aggressive_memory_cleanup()
            except Exception as e:
                print(f"Warning: Eval failed: {e}")
                import traceback
                traceback.print_exc()

        dist.barrier()

    if rank == 0:
        raw_s1 = _get_raw_model(predictor_stage1)
        raw_s2 = _get_raw_model(predictor_stage2)
        save_checkpoint_ddp(
            predictor_stage1, predictor_stage2,
            optimizer1, optimizer2,
            args.epochs,
            train_losses['total_loss_stage1'], train_losses['total_loss_stage2'],
            os.path.join(output_dir, 'final_model.pt')
        )
        raw_s1.save_pretrained(os.path.join(output_dir, 'predictor_stage1.pt'))
        raw_s2.save_pretrained(os.path.join(output_dir, 'predictor_stage2.pt'))

    aggressive_memory_cleanup()
    if use_wandb:
        wandb.finish()

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Training completed!")
        print(f"Best val stage1: {best_val_s1:.6f}, stage2: {best_val_s2:.6f}")
        print(f"Outputs: {output_dir}")
        print(f"{'='*60}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
