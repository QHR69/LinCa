"""
Sample script for qwen_edit inference with two-stage learned invertible decomposition network

全量 1212 张推理，seed = base_seed + original_idx (id_mapping 行索引)
输出: fullset/task_type/language/key.png

Usage:
    python sample_learned.py \
        --checkpoint outputs/.../checkpoint_epoch_26.pt \
        --dataset_path ./data/gedit_bench \
        --output_dir samples/qwen_edit_1212_full
"""

import os
import sys
import time
import torch
import argparse
import gc
from pathlib import Path
from PIL import Image, ExifTags
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from pipeline import QwenImageEditPipeline
from cache_functions import cache_init, pipeline_with_learned_cache, set_predictor_two_stage, get_predictor
from invertible_net import LearnedDecompositionPredictor


def _torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_predictor_flexible(checkpoint_path: str, device: str = 'cuda'):
    ckpt = _torch_load(checkpoint_path, device)
    if isinstance(ckpt, dict) and 'config' in ckpt and 'model_state_dict' in ckpt:
        cfg = ckpt['config']
        model = LearnedDecompositionPredictor(
            dim=cfg.get('dim', 3072),
            num_blocks=cfg.get('num_blocks', 6),
            hidden_dim=cfg.get('hidden_dim', 512),
            split_dims=cfg.get('split_dims', [1024, 1024, 1024]),
            dropout=cfg.get('dropout', 0.1),
        )
        model.load_state_dict(ckpt['model_state_dict'], strict=True)
        model = model.to(device)
        return model
    return LearnedDecompositionPredictor.from_pretrained(checkpoint_path, device=device)


def load_two_stage_predictors(path_s1: str, path_s2: str, device: str = 'cuda'):
    if path_s1 == path_s2 and os.path.exists(path_s1):
        ckpt = _torch_load(path_s1, device)
        if isinstance(ckpt, dict) and 'config' in ckpt and 'predictor_stage1' in ckpt and 'predictor_stage2' in ckpt:
            cfg = ckpt['config']
            p1 = LearnedDecompositionPredictor(
                dim=cfg.get('dim', 3072),
                num_blocks=cfg.get('num_blocks', 6),
                hidden_dim=cfg.get('hidden_dim', 512),
                split_dims=cfg.get('split_dims', [1024, 1024, 1024]),
                dropout=cfg.get('dropout', 0.1),
            )
            p2 = LearnedDecompositionPredictor(
                dim=cfg.get('dim', 3072),
                num_blocks=cfg.get('num_blocks', 6),
                hidden_dim=cfg.get('hidden_dim', 512),
                split_dims=cfg.get('split_dims', [1024, 1024, 1024]),
                dropout=cfg.get('dropout', 0.1),
            )
            p1.load_state_dict(ckpt['predictor_stage1'], strict=True)
            p2.load_state_dict(ckpt['predictor_stage2'], strict=True)
            return p1.to(device), p2.to(device)
    return load_predictor_flexible(path_s1, device), load_predictor_flexible(path_s2, device)


def _build_dataset_from_gedit_raw(base_path: Path):
    """从 gedit_bench_numbered 构建 1212 条数据"""
    import csv
    with open(base_path / "logs" / "id_mapping.csv", "r", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("task_type") != "task_type"]
    items = []
    for i, row in enumerate(rows):
        new_id = row["new_id"].strip()
        task_type = row["task_type"].strip()
        language = row["language"].strip()
        img_path = base_path / "images" / task_type / language / f"{new_id}.png"
        if not img_path.exists():
            img_path = base_path / "images" / task_type / language / f"{new_id}.jpg"
        prompt_path = base_path / "prompts" / task_type / language / f"{new_id}.txt"
        if not img_path.exists() or not prompt_path.exists():
            continue
        with open(prompt_path, "r", encoding="utf-8") as pf:
            instruction = pf.read().strip()
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size
        items.append({
            "idx": i,  # original_idx for seed
            "task_type": task_type,
            "instruction": instruction,
            "instruction_language": language,
            "key": new_id,
            "input_image": img,
            "orig_size": (orig_w, orig_h),
        })
    return items


def create_folders(output_dir: str, task_types: list, languages: list):
    base_dir = Path(output_dir) / "fullset"
    for task_type in task_types:
        for lang in languages:
            (base_dir / task_type / lang).mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="qwen_edit 1212 full inference with two-stage learned cache")

    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--checkpoint_stage1', type=str, default=None)
    parser.add_argument('--checkpoint_stage2', type=str, default=None)
    parser.add_argument('--checkpoint_dir', type=str, default=None)

    parser.add_argument('--dataset_path', type=str, default='./data/gedit_bench')
    parser.add_argument('--output_dir', type=str, default='samples/qwen_edit_1212_full')

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--interval', type=int, default=7)

    parser.add_argument('--model_path', type=str, default="Qwen/Qwen-Image-Edit")

    parser.add_argument('--width', type=int, default=1024)
    parser.add_argument('--height', type=int, default=1024)
    parser.add_argument('--num_steps', type=int, default=50)
    parser.add_argument('--guidance_scale', type=float, default=1.0)
    parser.add_argument('--true_cfg_scale', type=float, default=4.0)
    parser.add_argument('--negative_prompt', type=str, default=' ')

    parser.add_argument('--max_order', type=int, default=2)
    parser.add_argument('--min_order', type=int, default=0)
    parser.add_argument('--first_enhance', type=int, default=3)
    parser.add_argument('--forecast_method', type=str, default='hermite')
    parser.add_argument('--forecast_steps', type=int, default=5)
    parser.add_argument('--z2_forecast_method', type=str, default='lagrange',
                        choices=['lagrange', 'hermite'], help='2nd-order forecast: lagrange (default) or hermite')

    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--limit', type=int, default=None, help='Limit samples for testing')
    parser.add_argument('--add_metadata', action='store_true')

    args = parser.parse_args()

    t_start = time.perf_counter()

    if args.checkpoint is not None:
        if not os.path.isfile(args.checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
        path_s1 = path_s2 = args.checkpoint
    elif args.checkpoint_dir is not None:
        dirpath = args.checkpoint_dir
        if not os.path.isdir(dirpath):
            raise FileNotFoundError(f"Checkpoint dir not found: {dirpath}")
        path_s1 = os.path.join(dirpath, 'best_predictor_stage1.pt')
        path_s2 = os.path.join(dirpath, 'best_predictor_stage2.pt')
        if not os.path.exists(path_s1):
            path_s1 = os.path.join(dirpath, 'predictor_stage1.pt')
        if not os.path.exists(path_s2):
            path_s2 = os.path.join(dirpath, 'predictor_stage2.pt')
        if not os.path.exists(path_s1) or not os.path.exists(path_s2):
            import glob
            ckpts = sorted(glob.glob(os.path.join(dirpath, 'checkpoint_epoch_*.pt')))
            if ckpts:
                path_s1 = path_s2 = ckpts[-1]
            else:
                raise FileNotFoundError(f"No valid checkpoint in {dirpath}")
    else:
        path_s1 = args.checkpoint_stage1
        path_s2 = args.checkpoint_stage2

    if path_s1 is None or path_s2 is None:
        raise ValueError("Must provide --checkpoint, --checkpoint_dir, or --checkpoint_stage1 + --checkpoint_stage2")
    if not os.path.exists(path_s1) or not os.path.exists(path_s2):
        raise FileNotFoundError(f"Checkpoint not found: {path_s1} or {path_s2}")

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    print(f"\n>>> Loading two-stage predictors:")
    print(f"    Stage1: {path_s1}")
    print(f"    Stage2: {path_s2}")

    predictor_stage1, predictor_stage2 = load_two_stage_predictors(path_s1, path_s2, device=device)
    predictor_stage1.register_inverse_weights()
    predictor_stage2.register_inverse_weights()
    predictor_stage1.eval()
    predictor_stage2.eval()

    set_predictor_two_stage(predictor_stage1, predictor_stage2)
    if get_predictor() is None:
        raise RuntimeError("Failed to set global predictor!")
    print(f"✓ Two-stage predictors loaded")

    num_params = sum(p.numel() for p in predictor_stage1.parameters())
    print(f"  Parameters per stage: {num_params:,} ({num_params/1e6:.1f}M)")

    print(f"\n>>> Loading Qwen-Image-Edit pipeline from: {args.model_path}")
    pipe = QwenImageEditPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
    ).to(device)
    pipe = pipeline_with_learned_cache(pipe)
    print("✓ Pipeline loaded with two-stage learned cache")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_load_end = time.perf_counter()

    base_path = Path(args.dataset_path)
    if not (base_path / "logs" / "id_mapping.csv").exists():
        raise FileNotFoundError(f"Dataset not found: {args.dataset_path}")

    dataset = _build_dataset_from_gedit_raw(base_path)
    if args.limit is not None:
        dataset = dataset[:args.limit]
    print(f"\n>>> Loaded {len(dataset)} samples from: {args.dataset_path}")

    os.makedirs(args.output_dir, exist_ok=True)
    task_types = list(set(d["task_type"] for d in dataset))
    languages = ["en", "cn"]
    create_folders(args.output_dir, task_types, languages)

    print(f"\n>>> Generating images...")
    print(f"    Seed base: {args.seed}, Interval: {args.interval}")
    print(f"    Output: {args.output_dir}, Size: {args.width}x{args.height}\n")

    cache_kwargs = {
        'num_steps': args.num_steps,
        'test_FLOPs': False,
        'monitor_gpu_usage': False,
        'interval': args.interval,
        'max_order': args.max_order,
        'min_order': args.min_order,
        'first_enhance': args.first_enhance,
        'forecast_method': args.forecast_method,
        'decompose_method': 'learned',
        'use_z_cache': False,
        'forecast_steps': args.forecast_steps,
        'z2_forecast_method': args.z2_forecast_method,
    }

    for i, item in enumerate(tqdm(dataset, desc="Generating")):
        seed = args.seed + item["idx"]  # original_idx = id_mapping row index
        generator = torch.Generator(device).manual_seed(seed)
        cache_dic, current = cache_init(cache_kwargs)

        img = item["input_image"].resize((args.width, args.height), Image.Resampling.LANCZOS)

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

        # 输出 resize 回原图尺寸
        orig_w, orig_h = item.get("orig_size", (args.width, args.height))
        if image.size != (orig_w, orig_h):
            image = image.resize((orig_w, orig_h), Image.Resampling.LANCZOS)

        save_dir = Path(args.output_dir) / "fullset" / item["task_type"] / item["instruction_language"]
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{item['key']}.png"

        exif_data = Image.Exif()
        exif_data[ExifTags.Base.Software] = "AI generated;edit;qwen-learned-two-stage"
        exif_data[ExifTags.Base.Make] = "Qwen"
        exif_data[ExifTags.Base.Model] = "learned-decomposition-two-stage"
        if args.add_metadata:
            exif_data[ExifTags.Base.ImageDescription] = item["instruction"]

        image.save(save_path, exif=exif_data, quality=95, subsampling=0)

        del result, generator, cache_dic, current, image
        if (i + 1) % 5 == 0:
            torch.cuda.empty_cache()

    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\n>>> Done! Generated {len(dataset)} images in: {args.output_dir}")


if __name__ == '__main__':
    main()
