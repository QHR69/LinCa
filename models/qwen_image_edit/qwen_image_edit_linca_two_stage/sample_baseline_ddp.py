"""
Multi-GPU inference using the ORIGINAL Qwen Image Edit model (no cache acceleration).

所有 50 步全部做完整 transformer forward，相当于原始模型推理。
输出格式、seed 规则与 sample_learned_ddp.py 完全一致，可直接做质量对比。

特性:
  - 无缓存：设 first_enhance = num_steps，使所有步均为 full compute
  - 断点续跑：若输出文件已存在则自动跳过
  - 计时：统计纯推理时间（不含模型加载），汇总后输出平均单张耗时
  - seed 完全对齐：seed = base_seed + item["idx"]（与 sample_learned_ddp.py 一致）

Usage:
    # 单卡
    CUDA_VISIBLE_DEVICES=0 python sample_baseline_ddp.py \
        --dataset_path ./data/gedit_bench \
        --output_dir samples/qwen_edit_1212_baseline_50step \
        --seed 0

    # 多卡
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29602 \
        sample_baseline_ddp.py \
        --dataset_path ./data/gedit_bench \
        --output_dir samples/qwen_edit_1212_baseline_50step \
        --seed 0
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

# 清除 pipeline 写入的 cache_functions，改用本地版本（提供 cache_init 等函数）
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
cache_init                  = _cf_mod.cache_init
pipeline_with_learned_cache = _cf_mod.pipeline_with_learned_cache
# 注意：pipeline_with_learned_cache 只替换 transformer.forward 以支持 cache_dic/current 参数，
# 不传 checkpoint_path 所以不加载任何预测器。
# 再结合 first_enhance=num_steps + decompose_method='None'，所有步均全量计算，等同于原始模型。


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
        orig_w, orig_h = img.size
        items.append({
            "idx": i,               # 全局行号，决定 seed（不受分卡影响）
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
    parser.add_argument('--dataset_path',   type=str, default='./data/gedit_bench')
    parser.add_argument('--output_dir',     type=str, default='samples/qwen_edit_1212_baseline_50step')
    parser.add_argument('--seed',           type=int, default=0)
    parser.add_argument('--model_path',     type=str, default="Qwen/Qwen-Image-Edit")
    parser.add_argument('--width',          type=int, default=1024)
    parser.add_argument('--height',         type=int, default=1024)
    parser.add_argument('--num_steps',      type=int, default=50)
    parser.add_argument('--guidance_scale', type=float, default=1.0)
    parser.add_argument('--true_cfg_scale', type=float, default=4.0)
    parser.add_argument('--negative_prompt',type=str, default=' ')
    parser.add_argument('--limit',          type=int, default=None)
    args = parser.parse_args()

    # ---- DDP 初始化 ----
    dist.init_process_group("nccl")
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    device     = f"cuda:{rank}"
    torch.cuda.set_device(rank)

    # ---- 加载 pipeline（原始模型，不加任何 cache wrapper） ----
    if rank == 0:
        print(f">>> Loading original Qwen pipeline (no cache) ...")
    pipe = QwenImageEditPipeline.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
    ).to(device)
    # 替换 transformer.forward 以支持 cache_dic/current 参数（不传 checkpoint，不加载预测器）
    # 配合 first_enhance=num_steps + decompose_method='None'：全部 50 步均为 full compute
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

    # ---- cache_dic 参数：禁用缓存，等效于原始模型 50 步全量推理 ----
    # first_enhance = num_steps 使所有步均满足 step < first_enhance，
    # cal_type 将每步标记为 'full'，transformer 执行完整 forward，无任何跳步。
    cache_kwargs = {
        'num_steps':       args.num_steps,
        'test_FLOPs':      False,
        'monitor_gpu_usage': False,
        'interval':        1,
        'max_order':       0,
        'min_order':       0,
        'first_enhance':   args.num_steps,   # ← 关键：覆盖全部步，全部 type='full'
        'forecast_method': 'hermite',
        'decompose_method':'None',
        'use_z_cache':     False,
        'forecast_steps':  1,
    }

    # ---- 推理循环 ----
    times     = []
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

        # seed = base_seed + 全局行号（与 sample_learned_ddp.py 完全一致）
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
