"""
Generate cache data for qwen_edit LinCA training.
202 samples: indices [0, 6, 12, ..., 1206] from gedit_bench_numbered.
Resize to 1024x1024, seed = base_seed + original_idx.
Cache saved to ./cache_data, images to current dir.

Usage (single GPU):
    python generate_cache_data_edit.py

Usage (multi-GPU):
    torchrun --nproc_per_node=4 generate_cache_data_edit.py --ddp
"""

import os
import sys
import json
import csv
import torch
import argparse
from pathlib import Path
from PIL import Image, ExifTags
from tqdm import tqdm
from dataclasses import dataclass, asdict
from typing import List, Optional
import types
import importlib.util

_script_dir = Path(__file__).resolve().parent
_linca_root = _script_dir.parent.parent.parent  # qwen_edit_data->qwen_edit->Linca->LinCA
sys.path.insert(0, str(_script_dir))

# Load transformer from linca_data (avoid pipeline package conflict)
_tf_spec = importlib.util.spec_from_file_location(
    "transformer_qwenimage_data",
    _linca_root / "Linca" / "qwen_image" / "linca_data" / "pipeline" / "transformer_qwenimage_data.py",
)
_tf_mod = importlib.util.module_from_spec(_tf_spec)
sys.path.insert(0, str(_linca_root / "Linca" / "qwen_image" / "linca_data"))
_tf_spec.loader.exec_module(_tf_mod)
QwenImageTransformer2DModelForData = _tf_mod.QwenImageTransformer2DModelForData

from pipeline.pipeline_qwenimage_edit_data import QwenImageEditPipelineForData


@dataclass
class DataGenConfig:
    dataset_path: str = "./data/gedit_bench"
    cache_output_dir: str = "./cache_data/qwen_edit"
    image_output_dir: str = "."  # current dir for edited images
    negative_prompt: str = " "
    width: int = 1024
    height: int = 1024
    num_steps: int = 50
    guidance_scale: float = 1.0
    true_cfg_scale: float = 4.0
    seed: int = 0
    max_sequence_length: int = 512
    model_path: str = "Qwen/Qwen-Image-Edit"
    sample_interval: int = 6  # indices 0, 6, 12, ..., 1206 -> 202 samples
    limit: Optional[int] = None  # limit samples for testing (e.g. 1)
    only_indices: Optional[List[int]] = None  # 仅生成指定 sample_idx，用于补全缺失


def load_id_mapping_and_build_samples(config: DataGenConfig) -> List[dict]:
    """Load id_mapping.csv, take every 6th (indices 0,6,...,1206), return list of sample dicts."""
    mapping_path = Path(config.dataset_path) / "logs" / "id_mapping.csv"
    if not mapping_path.exists():
        raise FileNotFoundError(f"id_mapping not found: {mapping_path}")

    rows = []
    with open(mapping_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r.get("task_type") == "task_type":
                continue
            rows.append(r)

    # original_idx: 0, 6, 12, ..., 1206
    indices = list(range(0, len(rows), config.sample_interval))
    samples = []
    for i, orig_idx in enumerate(indices):
        if orig_idx >= len(rows):
            break
        row = rows[orig_idx]
        new_id = row["new_id"].strip()
        task_type = row["task_type"].strip()
        language = row["language"].strip()
        img_path = Path(config.dataset_path) / "images" / task_type / language / f"{new_id}.png"
        if not img_path.exists():
            img_path = Path(config.dataset_path) / "images" / task_type / language / f"{new_id}.jpg"
        prompt_path = Path(config.dataset_path) / "prompts" / task_type / language / f"{new_id}.txt"
        if not img_path.exists() or not prompt_path.exists():
            continue
        if config.limit is not None and len(samples) >= config.limit:
            break
        if config.only_indices is not None and i not in config.only_indices:
            continue
        with open(prompt_path, "r", encoding="utf-8") as pf:
            instruction = pf.read().strip()
        samples.append({
            "sample_idx": i,
            "original_idx": orig_idx,
            "new_id": new_id,
            "task_type": task_type,
            "language": language,
            "img_path": str(img_path),
            "instruction": instruction,
        })
    return samples


def rebuild_index_from_dirs(config: DataGenConfig):
    """从已有 sample_* 目录重建 index.json"""
    samples = []
    for d in sorted(Path(config.cache_output_dir).glob("sample_*")):
        if not d.is_dir():
            continue
        meta_path = d / "metadata.json"
        if not meta_path.exists():
            continue
        with open(meta_path, "r", encoding="utf-8") as f:
            m = json.load(f)
        samples.append({
            "sample_idx": m["sample_idx"],
            "original_idx": m["original_idx"],
            "new_id": m["new_id"],
            "dir": d.name,
            "seed": m["seed"],
            "seq_length": m["seq_length"],
        })
    samples.sort(key=lambda x: x["sample_idx"])
    index_data = {
        "config": asdict(config),
        "num_samples": len(samples),
        "num_steps": config.num_steps,
        "feature_dim": 3072,
        "seq_length": 4096,
        "samples": samples,
    }
    index_path = os.path.join(config.cache_output_dir, "index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index_data, f, indent=2, ensure_ascii=False)
    print(f"Rebuilt index: {len(samples)} samples -> {index_path}")


def setup_transformer_for_data(pipe):
    """Patch transformer forward to return intermediate features."""
    pipe.transformer.forward = types.MethodType(
        QwenImageTransformer2DModelForData.forward,
        pipe.transformer,
    )
    return pipe


def main_single(config: DataGenConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    samples = load_id_mapping_and_build_samples(config)
    print(f"Loaded {len(samples)} samples (indices 0,{config.sample_interval},...)")

    pipe = QwenImageEditPipelineForData.from_pretrained(
        config.model_path,
        torch_dtype=torch.bfloat16,
    ).to(device=device)
    pipe = setup_transformer_for_data(pipe)

    os.makedirs(config.cache_output_dir, exist_ok=True)
    os.makedirs(config.image_output_dir, exist_ok=True)

    index_data = {
        "config": asdict(config),
        "num_samples": len(samples),
        "num_steps": config.num_steps,
        "feature_dim": 3072,
        "seq_length": 4096,
        "samples": [],
    }

    for s in tqdm(samples, desc="Generating cache"):
        sample_dir = os.path.join(config.cache_output_dir, f"sample_{s['sample_idx']:04d}")
        cond_dir = os.path.join(sample_dir, "cond")
        uncond_dir = os.path.join(sample_dir, "uncond")
        os.makedirs(cond_dir, exist_ok=True)
        os.makedirs(uncond_dir, exist_ok=True)

        seed = config.seed + s["original_idx"]
        generator = torch.Generator(device).manual_seed(seed)

        img = Image.open(s["img_path"]).convert("RGB")
        img = img.resize((config.width, config.height), Image.Resampling.LANCZOS)

        try:
            cache_data = pipe.generate_and_save_cache(
                image=img,
                prompt=s["instruction"],
                negative_prompt=config.negative_prompt,
                height=config.height,
                width=config.width,
                num_inference_steps=config.num_steps,
                guidance_scale=config.guidance_scale,
                true_cfg_scale=config.true_cfg_scale,
                generator=generator,
                max_sequence_length=config.max_sequence_length,
            )

            for step in range(config.num_steps):
                torch.save(cache_data["cond"][step], os.path.join(cond_dir, f"step_{step:02d}.pt"))
                if len(cache_data["uncond"]) > 0:
                    torch.save(cache_data["uncond"][step], os.path.join(uncond_dir, f"step_{step:02d}.pt"))

            out_img = cache_data["image"]
            img_path_out = os.path.join(config.image_output_dir, f"sample_{s['sample_idx']:04d}.png")
            out_img.save(img_path_out)

            metadata = {
                "sample_idx": s["sample_idx"],
                "original_idx": s["original_idx"],
                "new_id": s["new_id"],
                "task_type": s["task_type"],
                "language": s["language"],
                "instruction": s["instruction"],
                "seed": seed,
                "num_steps": config.num_steps,
                "height": config.height,
                "width": config.width,
                "feature_dim": 3072,
                "seq_length": cache_data["seq_length"],
                "image_path": img_path_out,
            }
            with open(os.path.join(sample_dir, "metadata.json"), "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            index_data["samples"].append({
                "sample_idx": s["sample_idx"],
                "original_idx": s["original_idx"],
                "new_id": s["new_id"],
                "dir": f"sample_{s['sample_idx']:04d}",
                "seed": seed,
                "seq_length": cache_data["seq_length"],
            })

        except Exception as e:
            print(f"Error sample {s['sample_idx']}: {e}")
            continue

        if s["sample_idx"] % 10 == 0:
            torch.cuda.empty_cache()

    index_path = os.path.join(config.cache_output_dir, "index.json")
    if config.only_indices is not None:
        rebuild_index_from_dirs(config)
    else:
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index_data, f, indent=2, ensure_ascii=False)

    print(f"\nDone! Cache: {config.cache_output_dir}, Index: {index_path}")


def main_ddp(config: DataGenConfig):
    import torch.distributed as dist

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = f"cuda:{rank}"
    torch.cuda.set_device(rank)

    if rank == 0:
        print(f"Running on {world_size} GPUs")

    samples = load_id_mapping_and_build_samples(config)
    total = len(samples)
    if rank == 0:
        print(f"Total samples: {total}")

    if total == 0:
        print("No samples to process.")
        dist.destroy_process_group()
        return

    per_proc = (total + world_size - 1) // world_size
    start = rank * per_proc
    end = min(start + per_proc, total)
    local_samples = samples[start:end]

    if rank == 0:
        pipe = QwenImageEditPipelineForData.from_pretrained(
            config.model_path,
            torch_dtype=torch.bfloat16,
        ).to(device=device)
    else:
        pipe = QwenImageEditPipelineForData.from_pretrained(
            config.model_path,
            torch_dtype=torch.bfloat16,
        ).to(device=device)
    pipe = setup_transformer_for_data(pipe)

    if rank == 0:
        os.makedirs(config.cache_output_dir, exist_ok=True)
        os.makedirs(config.image_output_dir, exist_ok=True)
    dist.barrier()

    local_index = []
    iterator = tqdm(local_samples, desc=f"GPU {rank}", disable=(rank != 0))

    for s in iterator:
        sample_dir = os.path.join(config.cache_output_dir, f"sample_{s['sample_idx']:04d}")
        cond_dir = os.path.join(sample_dir, "cond")
        uncond_dir = os.path.join(sample_dir, "uncond")
        os.makedirs(cond_dir, exist_ok=True)
        os.makedirs(uncond_dir, exist_ok=True)

        seed = config.seed + s["original_idx"]
        generator = torch.Generator(device).manual_seed(seed)

        img = Image.open(s["img_path"]).convert("RGB")
        img = img.resize((config.width, config.height), Image.Resampling.LANCZOS)

        try:
            cache_data = pipe.generate_and_save_cache(
                image=img,
                prompt=s["instruction"],
                negative_prompt=config.negative_prompt,
                height=config.height,
                width=config.width,
                num_inference_steps=config.num_steps,
                guidance_scale=config.guidance_scale,
                true_cfg_scale=config.true_cfg_scale,
                generator=generator,
                max_sequence_length=config.max_sequence_length,
            )

            for step in range(config.num_steps):
                torch.save(cache_data["cond"][step], os.path.join(cond_dir, f"step_{step:02d}.pt"))
                if len(cache_data["uncond"]) > 0:
                    torch.save(cache_data["uncond"][step], os.path.join(uncond_dir, f"step_{step:02d}.pt"))

            out_img = cache_data["image"]
            img_path_out = os.path.join(config.image_output_dir, f"sample_{s['sample_idx']:04d}.png")
            out_img.save(img_path_out)

            metadata = {
                "sample_idx": s["sample_idx"],
                "original_idx": s["original_idx"],
                "new_id": s["new_id"],
                "task_type": s["task_type"],
                "language": s["language"],
                "instruction": s["instruction"],
                "seed": seed,
                "num_steps": config.num_steps,
                "height": config.height,
                "width": config.width,
                "feature_dim": 3072,
                "seq_length": cache_data["seq_length"],
                "image_path": img_path_out,
                "generated_by_gpu": rank,
            }
            with open(os.path.join(sample_dir, "metadata.json"), "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            local_index.append({
                "sample_idx": s["sample_idx"],
                "original_idx": s["original_idx"],
                "new_id": s["new_id"],
                "dir": f"sample_{s['sample_idx']:04d}",
                "seed": seed,
                "seq_length": cache_data["seq_length"],
            })

        except Exception as e:
            print(f"[GPU {rank}] Error sample {s['sample_idx']}: {e}")
            continue

        if (s["sample_idx"] - start) % 10 == 0:
            torch.cuda.empty_cache()

    dist.barrier()

    # Each rank saves its index
    rank_index_path = os.path.join(config.cache_output_dir, f"index_rank_{rank}.json")
    with open(rank_index_path, "w", encoding="utf-8") as f:
        json.dump(local_index, f, indent=2, ensure_ascii=False)

    dist.barrier()

    if rank == 0:
        all_data = []
        for r in range(world_size):
            rpath = os.path.join(config.cache_output_dir, f"index_rank_{r}.json")
            if os.path.exists(rpath):
                with open(rpath, "r", encoding="utf-8") as f:
                    all_data.extend(json.load(f))
                os.remove(rpath)
        all_data.sort(key=lambda x: x["sample_idx"])

        index_data = {
            "config": asdict(config),
            "num_samples": len(all_data),
            "num_steps": config.num_steps,
            "feature_dim": 3072,
            "seq_length": 4096,
            "world_size": world_size,
            "samples": all_data,
        }
        index_path = os.path.join(config.cache_output_dir, "index.json")
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index_data, f, indent=2, ensure_ascii=False)
        print(f"\nDone! Cache: {config.cache_output_dir}, Index: {index_path}, Total: {len(all_data)}")

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate qwen_edit cache data (202 samples)")
    parser.add_argument("--dataset_path", type=str, default="./data/gedit_bench")
    parser.add_argument("--cache_output_dir", type=str, default="./cache_data/qwen_edit")
    parser.add_argument("--image_output_dir", type=str, default=".")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--true_cfg_scale", type=float, default=4.0)
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen-Image-Edit")
    parser.add_argument("--ddp", action="store_true", help="Use multi-GPU DDP")
    parser.add_argument("--limit", type=int, default=None, help="Limit samples for testing (e.g. 1)")
    parser.add_argument("--only_indices", type=str, default=None,
                        help="Comma-separated sample_idx to generate only (e.g. 24,25,50)")
    args = parser.parse_args()

    only_indices = None
    if args.only_indices:
        only_indices = [int(x.strip()) for x in args.only_indices.split(",")]

    config = DataGenConfig(
        dataset_path=args.dataset_path,
        cache_output_dir=args.cache_output_dir,
        image_output_dir=args.image_output_dir,
        seed=args.seed,
        num_steps=args.num_steps,
        true_cfg_scale=args.true_cfg_scale,
        model_path=args.model_path,
        limit=args.limit,
        only_indices=only_indices,
    )

    if args.ddp:
        main_ddp(config)
    else:
        main_single(config)
