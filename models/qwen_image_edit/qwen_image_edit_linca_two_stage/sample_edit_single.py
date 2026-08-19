"""
普适单张/批量图片编辑推理脚本（LinCA 两阶段 learned cache）

不依赖 gedit_bench_numbered 数据集，可对任意图片+prompt 进行编辑。
输出会 resize 回原图尺寸。

Usage:
    # 单张
    python sample_edit_single.py --checkpoint checkpoints/checkpoint_epoch_50.pt \\
        --image demo_pairs/images/img_1.png --prompt "将背景改为城市街道" \\
        --output output.png

    # 批量（从 pairs.csv）
    python sample_edit_single.py --checkpoint checkpoints/checkpoint_epoch_50.pt \\
        --pairs_file demo_pairs/pairs.csv --output_dir samples/demo_edit
"""

import os
import sys
import csv
import torch
import argparse
from pathlib import Path
from PIL import Image

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
_LINCA_ROOT = _SCRIPT_DIR.parent.parent.parent
_PIPELINE_BASE = _LINCA_ROOT / "freqca_qwen"
if str(_PIPELINE_BASE) not in sys.path:
    sys.path.insert(1, str(_PIPELINE_BASE))

_to_remove = [k for k in list(sys.modules) if k == "cache_functions" or k.startswith("cache_functions.")]
for _k in _to_remove:
    del sys.modules[_k]
_local_cf = _SCRIPT_DIR / "cache_functions" / "__init__.py"
_spec = __import__("importlib").util.spec_from_file_location(
    "cache_functions", _local_cf,
    submodule_search_locations=[str(_SCRIPT_DIR / "cache_functions")]
)
_cf_mod = __import__("importlib").util.module_from_spec(_spec)
sys.modules["cache_functions"] = _cf_mod
_spec.loader.exec_module(_cf_mod)
cache_init = _cf_mod.cache_init
pipeline_with_learned_cache = _cf_mod.pipeline_with_learned_cache
set_predictor_two_stage = _cf_mod.set_predictor_two_stage

from pipeline import QwenImageEditPipeline
from invertible_net import LearnedDecompositionPredictor


def _torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_two_stage(checkpoint_path: str, device: str):
    ckpt = _torch_load(checkpoint_path, device)
    if not (isinstance(ckpt, dict) and "predictor_stage1" in ckpt and "predictor_stage2" in ckpt):
        raise ValueError(f"Checkpoint must contain predictor_stage1/2: {checkpoint_path}")
    cfg = ckpt["config"]
    def _mk():
        return LearnedDecompositionPredictor(
            dim=cfg.get("dim", 3072), num_blocks=cfg.get("num_blocks", 6),
            hidden_dim=cfg.get("hidden_dim", 512),
            split_dims=cfg.get("split_dims", [1024, 1024, 1024]),
            dropout=cfg.get("dropout", 0.1),
        )
    p1, p2 = _mk(), _mk()
    p1.load_state_dict(ckpt["predictor_stage1"], strict=True)
    p2.load_state_dict(ckpt["predictor_stage2"], strict=True)
    p1.register_inverse_weights()
    p2.register_inverse_weights()
    p1.eval()
    p2.eval()
    return p1.to(device), p2.to(device)


def load_pairs(pairs_file: str, images_dir: str):
    """加载 pairs.csv: image,prompt"""
    items = []
    images_dir = Path(images_dir)
    with open(pairs_file, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            img = row.get("image", "").strip()
            prompt = row.get("prompt", "").strip()
            if not img or not prompt:
                continue
            ip = images_dir / img if not Path(img).is_absolute() else Path(img)
            if not ip.exists():
                continue
            items.append({"image_path": str(ip), "prompt": prompt})
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--image", type=str, default=None, help="单张图片路径")
    ap.add_argument("--prompt", type=str, default=None, help="单张时的编辑指令")
    ap.add_argument("--pairs_file", type=str, default=None, help="批量: pairs.csv")
    ap.add_argument("--images_dir", type=str, default=None, help="pairs 中 image 列相对于此目录")
    ap.add_argument("--output", type=str, default="output.png", help="单张输出路径")
    ap.add_argument("--output_dir", type=str, default="samples/demo_edit", help="批量Output directory")
    ap.add_argument("--limit", type=int, default=None, help="批量时最多处理条数（用于快速验证）")
    ap.add_argument("--z2_forecast_method", type=str, default="lagrange",
                    choices=["lagrange", "hermite"], help="2nd-order forecast: lagrange (default) or hermite")
    ap.add_argument("--interval", type=int, default=7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model_path", type=str, default="Qwen/Qwen-Image-Edit")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--num_steps", type=int, default=50)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    p1, p2 = load_two_stage(args.checkpoint, device)
    set_predictor_two_stage(p1, p2)

    pipe = QwenImageEditPipeline.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
    ).to(device)
    pipe = pipeline_with_learned_cache(pipe)

    cache_kwargs = {
        "num_steps": args.num_steps, "test_FLOPs": False, "monitor_gpu_usage": False,
        "interval": args.interval, "max_order": 2, "min_order": 0, "first_enhance": 3,
        "forecast_method": "hermite", "decompose_method": "learned",
        "use_z_cache": False, "forecast_steps": 5,
        "z2_forecast_method": args.z2_forecast_method,
    }

    if args.image and args.prompt:
        # 单张
        img = Image.open(args.image).convert("RGB")
        orig_w, orig_h = img.size
        img_resized = img.resize((args.width, args.height), Image.Resampling.LANCZOS)
        gen = torch.Generator(device).manual_seed(args.seed)
        cache_dic, current = cache_init(cache_kwargs)
        with torch.inference_mode():
            result = pipe(
                image=img_resized,
                prompt=args.prompt,
                negative_prompt=" ",
                height=args.height,
                width=args.width,
                num_inference_steps=args.num_steps,
                guidance_scale=1.0,
                true_cfg_scale=4.0,
                generator=gen,
                max_sequence_length=512,
                cache_dic=cache_dic,
                current=current,
            )
        out = result.images[0]
        if out.size != (orig_w, orig_h):
            out = out.resize((orig_w, orig_h), Image.Resampling.LANCZOS)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        out.save(args.output, quality=95)
        print(f"Saved: {args.output} (resized to {orig_w}x{orig_h})")

    elif args.pairs_file:
        # 批量
        images_dir = args.images_dir or str(Path(args.pairs_file).parent / "images")
        pairs = load_pairs(args.pairs_file, images_dir)
        if args.limit is not None:
            pairs = pairs[:args.limit]
        if not pairs:
            print("No valid pairs")
            return
        os.makedirs(args.output_dir, exist_ok=True)
        for i, p in enumerate(pairs):
            img = Image.open(p["image_path"]).convert("RGB")
            orig_w, orig_h = img.size
            img_resized = img.resize((args.width, args.height), Image.Resampling.LANCZOS)
            gen = torch.Generator(device).manual_seed(args.seed + i)
            cache_dic, current = cache_init(cache_kwargs)
            with torch.inference_mode():
                result = pipe(
                    image=img_resized,
                    prompt=p["prompt"],
                    negative_prompt=" ",
                    height=args.height,
                    width=args.width,
                    num_inference_steps=args.num_steps,
                    guidance_scale=1.0,
                    true_cfg_scale=4.0,
                    generator=gen,
                    max_sequence_length=512,
                    cache_dic=cache_dic,
                    current=current,
                )
            out = result.images[0]
            if out.size != (orig_w, orig_h):
                out = out.resize((orig_w, orig_h), Image.Resampling.LANCZOS)
            stem = Path(p["image_path"]).stem
            save_path = Path(args.output_dir) / f"{stem}.png"
            out.save(save_path, quality=95)
            print(f"  {i+1}/{len(pairs)}: {save_path} ({orig_w}x{orig_h})")
        print(f"Done. {len(pairs)} images -> {args.output_dir}")

    else:
        print("Use --image + --prompt for single, or --pairs_file for batch")
        return


if __name__ == "__main__":
    main()
