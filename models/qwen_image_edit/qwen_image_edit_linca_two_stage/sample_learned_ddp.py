"""
Multi-GPU inference for qwen_edit 1212 full with two-stage learned cache.

特性:
  - 断点续跑：若输出文件已存在则自动跳过，支持中断后继续
  - 计时：统计纯推理时间（不含模型加载），汇总后输出平均单张耗时

Usage:
    # 单卡
    CUDA_VISIBLE_DEVICES=0 python sample_learned_ddp.py \
        --checkpoint_dir <ckpt.pt> --dataset_path <path> --output_dir <out>

    # 多卡
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29601 \
        sample_learned_ddp.py \
        --checkpoint_dir ./checkpoints/qwen-image-edit/\
qwen_image_edit_linca_two_stage/outputs/qwen_edit_linca_two_stage6512_b16_40_10_8/\
qwen_edit_linca_two_stage_6512_b16_40_10_8/\
qwen_edit_two_stage_b6_h512_ddp6512_b16_40_10_8/checkpoint_epoch_50.pt \
        --dataset_path ./data/gedit_bench \
        --output_dir samples/qwen_edit_1212_full_6512_ep50_inter10 \
        --seed 0 --interval 10
"""

import os
import sys
import time
import importlib.util
import torch
import torch.distributed as dist
from pathlib import Path
from PIL import Image
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
_LINCA_ROOT = _SCRIPT_DIR.parent.parent.parent
_PIPELINE_BASE = _LINCA_ROOT / "freqca_qwen"  # pipeline dependency under the LinCA root
if str(_PIPELINE_BASE) not in sys.path:
    sys.path.insert(1, str(_PIPELINE_BASE))

from pipeline import QwenImageEditPipeline

# 清除 pipeline 写入的 cache_functions，改用本地版本
_to_remove = [k for k in list(sys.modules) if k == "cache_functions" or k.startswith("cache_functions.")]
for _k in _to_remove:
    del sys.modules[_k]
_local_cf = _SCRIPT_DIR / "cache_functions" / "__init__.py"
_spec = importlib.util.spec_from_file_location(
    "cache_functions", _local_cf,
    submodule_search_locations=[str(_SCRIPT_DIR / "cache_functions")]
)
_cf_mod = importlib.util.module_from_spec(_spec)
sys.modules["cache_functions"] = _cf_mod
_spec.loader.exec_module(_cf_mod)
cache_init               = _cf_mod.cache_init
pipeline_with_learned_cache = _cf_mod.pipeline_with_learned_cache
set_predictor_two_stage  = _cf_mod.set_predictor_two_stage

from invertible_net import LearnedDecompositionPredictor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_two_stage_predictors(path_s1: str, path_s2: str, device: str = 'cuda'):
    if path_s1 == path_s2 and os.path.exists(path_s1):
        ckpt = _torch_load(path_s1, device)
        if isinstance(ckpt, dict) and 'config' in ckpt and 'predictor_stage1' in ckpt and 'predictor_stage2' in ckpt:
            cfg = ckpt['config']
            def _mk():
                return LearnedDecompositionPredictor(
                    dim=cfg.get('dim', 3072), num_blocks=cfg.get('num_blocks', 6),
                    hidden_dim=cfg.get('hidden_dim', 512),
                    split_dims=cfg.get('split_dims', [1024, 1024, 1024]),
                    dropout=cfg.get('dropout', 0.1),
                )
            p1, p2 = _mk(), _mk()
            p1.load_state_dict(ckpt['predictor_stage1'], strict=True)
            p2.load_state_dict(ckpt['predictor_stage2'], strict=True)
            return p1.to(device), p2.to(device)
    p1 = LearnedDecompositionPredictor.from_pretrained(path_s1, device=device)
    p2 = LearnedDecompositionPredictor.from_pretrained(path_s2, device=device)
    return p1, p2


def _build_dataset(base_path: Path):
    import csv
    with open(base_path / "logs" / "id_mapping.csv", "r", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("task_type") != "task_type"]
    items = []
    for i, row in enumerate(rows):
        new_id    = row["new_id"].strip()
        task_type = row["task_type"].strip()
        language  = row["language"].strip()
        img_path  = base_path / "images" / task_type / language / f"{new_id}.png"
        if not img_path.exists():
            img_path = base_path / "images" / task_type / language / f"{new_id}.jpg"
        prompt_path = base_path / "prompts" / task_type / language / f"{new_id}.txt"
        if not img_path.exists() or not prompt_path.exists():
            continue
        instruction = prompt_path.read_text(encoding="utf-8").strip()
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size  # 原始尺寸，保存后用于输出 resize 回原图
        items.append({
            "idx": i,               # 全局行号，用于 seed 对齐（与图片多少卡无关）
            "task_type": task_type,
            "instruction": instruction,
            "instruction_language": language,
            "key": new_id,
            "input_image": img,
            "orig_size": (orig_w, orig_h),
        })
    return items


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',     type=str, default=None)
    parser.add_argument('--checkpoint_dir', type=str, default=None)
    parser.add_argument('--dataset_path',   type=str, default='./data/gedit_bench')
    parser.add_argument('--output_dir',     type=str, default='samples/qwen_edit_1212_full')
    parser.add_argument('--seed',           type=int, default=0)
    parser.add_argument('--interval',       type=int, default=7)
    parser.add_argument('--model_path',     type=str, default="Qwen/Qwen-Image-Edit")
    parser.add_argument('--width',          type=int, default=1024)
    parser.add_argument('--height',         type=int, default=1024)
    parser.add_argument('--num_steps',      type=int, default=50)
    parser.add_argument('--guidance_scale', type=float, default=1.0)
    parser.add_argument('--true_cfg_scale', type=float, default=4.0)
    parser.add_argument('--negative_prompt',type=str, default=' ')
    parser.add_argument('--limit',          type=int, default=None)
    parser.add_argument('--z2_forecast_method', type=str, default='lagrange',
                        choices=['lagrange', 'hermite'], help='2nd-order forecast: lagrange (default) or hermite')
    args = parser.parse_args()

    # ---- DDP 初始化 ----
    dist.init_process_group("nccl")
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    device     = f"cuda:{rank}"
    torch.cuda.set_device(rank)

    # ---- 解析 checkpoint 路径 ----
    if args.checkpoint:
        path_s1 = path_s2 = args.checkpoint
    elif args.checkpoint_dir:
        if args.checkpoint_dir.rstrip(os.sep).endswith('.pt') and os.path.isfile(args.checkpoint_dir):
            path_s1 = path_s2 = args.checkpoint_dir
        else:
            path_s1 = os.path.join(args.checkpoint_dir, 'best_predictor_stage1.pt')
            path_s2 = os.path.join(args.checkpoint_dir, 'best_predictor_stage2.pt')
            if not os.path.exists(path_s1):
                path_s1 = os.path.join(args.checkpoint_dir, 'predictor_stage1.pt')
            if not os.path.exists(path_s2):
                path_s2 = os.path.join(args.checkpoint_dir, 'predictor_stage2.pt')
            if not os.path.exists(path_s1) or not os.path.exists(path_s2):
                import glob
                ckpts = sorted(glob.glob(os.path.join(args.checkpoint_dir, 'checkpoint_epoch_*.pt')))
                if ckpts:
                    path_s1 = path_s2 = ckpts[-1]
                else:
                    raise FileNotFoundError(f"No checkpoint in {args.checkpoint_dir}")
    else:
        raise ValueError("Need --checkpoint or --checkpoint_dir")

    # ---- 加载 predictor ----
    if rank == 0:
        print(f">>> Loading predictors ...")
    predictor_stage1, predictor_stage2 = load_two_stage_predictors(path_s1, path_s2, device=device)
    predictor_stage1.register_inverse_weights(); predictor_stage1.eval()
    predictor_stage2.register_inverse_weights(); predictor_stage2.eval()
    set_predictor_two_stage(predictor_stage1, predictor_stage2)

    # ---- 加载 pipeline ----
    if rank == 0:
        print(f">>> Loading pipeline ...")
    pipe = QwenImageEditPipeline.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
    ).to(device)
    pipe = pipeline_with_learned_cache(pipe)

    # ---- 加载数据集 ----
    base_path = Path(args.dataset_path)
    dataset   = _build_dataset(base_path)
    if args.limit:
        dataset = dataset[:args.limit]

    total    = len(dataset)
    per_proc = (total + world_size - 1) // world_size
    start    = rank * per_proc
    end      = min(start + per_proc, total)
    local_dataset = dataset[start:end]

    # ---- 创建Output directory（rank 0 负责） ----
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        for tt in set(d["task_type"] for d in dataset):
            for lang in ["en", "cn"]:
                (Path(args.output_dir) / "fullset" / tt / lang).mkdir(parents=True, exist_ok=True)
    dist.barrier()

    # ---- cache 初始化参数 ----
    cache_kwargs = {
        'num_steps': args.num_steps, 'test_FLOPs': False, 'monitor_gpu_usage': False,
        'interval': args.interval, 'max_order': 2, 'min_order': 0, 'first_enhance': 3,
        'forecast_method': 'hermite', 'decompose_method': 'learned',
        'use_z_cache': False, 'forecast_steps': 5,
        'z2_forecast_method': args.z2_forecast_method,
    }

    # ---- 推理循环 ----
    times     = []   # 每张图的纯推理时间（秒）
    skipped   = 0
    processed = 0

    for i, item in enumerate(tqdm(local_dataset, desc=f"Rank {rank}", disable=(rank != 0))):
        save_path = (
            Path(args.output_dir) / "fullset"
            / item["task_type"] / item["instruction_language"]
            / f"{item['key']}.png"
        )

        # ---- 断点续跑：已存在则跳过 ----
        if save_path.exists():
            skipped += 1
            continue

        seed      = args.seed + item["idx"]
        generator = torch.Generator(device).manual_seed(seed)
        cache_dic, current = cache_init(cache_kwargs)

        img = item["input_image"].resize((args.width, args.height), Image.Resampling.LANCZOS)

        # ---- 计时：仅计算推理时间 ----
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.inference_mode():
            result = pipe(
                image=img,
                prompt=item["instruction"],
                negative_prompt=args.negative_prompt,
                height=args.height,
                width=args.width,
                num_inference_steps=args.num_steps,
                guidance_scale=args.guidance_scale,
                true_cfg_scale=args.true_cfg_scale,
                generator=generator,
                max_sequence_length=512,
                cache_dic=cache_dic,
                current=current,
            )
            image = result.images[0]

        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)

        # ---- 输出 resize 回原图尺寸 ----
        orig_w, orig_h = item.get("orig_size", (args.width, args.height))
        if image.size != (orig_w, orig_h):
            image = image.resize((orig_w, orig_h), Image.Resampling.LANCZOS)

        # ---- 保存 ----
        save_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(save_path, quality=95)
        processed += 1

        del result, generator, cache_dic, current, image
        if (i + 1) % 5 == 0:
            torch.cuda.empty_cache()

    # ---- 跨卡汇总时间 ----
    n_proc = len(times)
    t_proc = sum(times)
    stats  = torch.tensor([t_proc, float(n_proc)], dtype=torch.float64, device=device)
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)

    dist.barrier()
    if rank == 0:
        total_t = stats[0].item()
        total_n = int(stats[1].item())
        print(f"\n>>> Done! total={total}, processed={total_n}, skipped(all ranks)={total - total_n}")
        if total_n > 0:
            print(f">>> 平均单张推理时间: {total_t / total_n:.3f}s  (合计 {total_n} 张，总耗时 {total_t:.1f}s)")
        else:
            print(">>> 所有图片均已存在，无新生成（全部跳过）")

    dist.destroy_process_group()


if __name__ == '__main__':
    main()
