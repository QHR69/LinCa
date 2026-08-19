"""
Sample script for inference with two-stage learned invertible decomposition network

使用训练好的两阶段 learned 权重进行推理

用法 1 - 单文件 checkpoint（推荐，checkpoint_epoch_*.pt 含 stage1+stage2）:
    python sample_learned.py \
        --checkpoint outputs/.../checkpoint_epoch_26.pt \
        --prompt_file prompts/DrawBench200.txt \
        --output_dir samples/output

用法 2 - 指定目录（自动查找 best_/predictor_stage1.pt 或 checkpoint_epoch_*.pt）:
    python sample_learned.py \
        --checkpoint_dir outputs/linca_v4_two_stage/two_stage_v4 \
        ...

用法 3 - 分别指定 stage1/stage2:
    python sample_learned.py \
        --checkpoint_stage1 .../best_predictor_stage1.pt \
        --checkpoint_stage2 .../best_predictor_stage2.pt \
        ...
"""

import os
import time
import torch
import argparse
import gc
from PIL import Image, ExifTags
from tqdm import tqdm

from pipeline.pipeline_qwenimage import QwenImagePipeline
from cache_functions import cache_init, pipeline_with_learned_cache, set_predictor_two_stage, get_predictor
from invertible_net import LearnedDecompositionPredictor


def _torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_predictor_flexible(checkpoint_path: str, device: str = 'cuda'):
    """
    灵活加载 predictor：支持训练 checkpoint (config+model_state_dict) 或 from_pretrained 格式。
    two_stage 版本使用 split_dims。
    """
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
    """
    加载两阶段 predictor。若 path_s1 与 path_s2 指向同一 checkpoint 文件且该文件包含
    predictor_stage1 和 predictor_stage2，则从该文件加载；否则分别加载两个文件。
    """
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


def read_prompts(prompt_file: str):
    """读取 prompt 文件"""
    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts


def collect_missing_images(prompts, num_images_per_prompt, output_dir):
    """收集尚未生成的图片 (img_idx, prompt_idx, prompt)，支持断点续采"""
    missing = []
    for img_idx in range(num_images_per_prompt):
        for i, prompt in enumerate(prompts):
            filename = f"img_{i:04d}_{img_idx}.jpg" if num_images_per_prompt > 1 else f"img_{i:04d}.jpg"
            output_path = os.path.join(output_dir, filename)
            if not os.path.exists(output_path):
                missing.append((img_idx, i, prompt))
    return missing


def main():
    parser = argparse.ArgumentParser(description="Generate images using two-stage learned decomposition (v4)")
    
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Single checkpoint file (checkpoint_epoch_*.pt) containing both stage1+stage2; 等同 --checkpoint_stage1/2 指向同一文件')
    parser.add_argument('--checkpoint_stage1', type=str, default=None,
                        help='Path to stage1 predictor (e.g., best_predictor_stage1.pt)')
    parser.add_argument('--checkpoint_stage2', type=str, default=None,
                        help='Path to stage2 predictor (e.g., best_predictor_stage2.pt)')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Directory containing best_predictor_stage1.pt or predictor_stage1.pt (或 checkpoint_epoch_*.pt)')
    
    parser.add_argument('--prompt_file', type=str, default='prompts/DrawBench200.txt')
    parser.add_argument('--output_dir', type=str, default='samples/two_stage_output')
    parser.add_argument('--limit', type=int, default=None, help='最多生成的 prompt 数量（用于快速验证）')
    
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--interval', type=int, default=7)
    
    parser.add_argument('--model_path', type=str, default="Qwen/Qwen-Image")
    
    parser.add_argument('--width', type=int, default=1328)
    parser.add_argument('--height', type=int, default=1328)
    parser.add_argument('--num_steps', type=int, default=50)
    parser.add_argument('--guidance_scale', type=float, default=1.0)
    parser.add_argument('--true_cfg_scale', type=float, default=4.0)
    parser.add_argument('--negative_prompt', type=str, default=' ')
    parser.add_argument('--num_images_per_prompt', type=int, default=1)
    
    parser.add_argument('--max_order', type=int, default=2)
    parser.add_argument('--min_order', type=int, default=0)
    parser.add_argument('--first_enhance', type=int, default=3)
    parser.add_argument('--forecast_method', type=str, default='hermite')
    parser.add_argument('--forecast_steps', type=int, default=5)
    parser.add_argument('--z2_forecast_method', type=str, default='lagrange',
                        choices=['lagrange', 'hermite'], help='2nd-order forecast: lagrange (default) or hermite')
    
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--add_metadata', action='store_true')
    parser.add_argument('--timing_file', type=str, default=None, help='Append timing result (exp_name, interval, load_sec, avg_inference_sec, num_timed)')
    parser.add_argument('--exp_name', type=str, default=None, help='Experiment name for timing log')
    parser.add_argument('--num_timed_images', type=int, default=10, help='Number of images to time for avg inference')
    
    args = parser.parse_args()
    
    t_start = time.perf_counter()
    # 1. --checkpoint: 单文件（checkpoint_epoch_*.pt），同时用作 stage1 和 stage2
    if args.checkpoint is not None:
        if not os.path.isfile(args.checkpoint):
            raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")
        path_s1 = path_s2 = args.checkpoint
    # 2. --checkpoint_dir: 目录，查找 best_/predictor_stage1.pt 或 checkpoint_epoch_*.pt
    elif args.checkpoint_dir is not None:
        dirpath = args.checkpoint_dir
        if not os.path.isdir(dirpath):
            raise FileNotFoundError(f"Checkpoint directory not found: {dirpath}")
        path_s1 = os.path.join(dirpath, 'best_predictor_stage1.pt')
        path_s2 = os.path.join(dirpath, 'best_predictor_stage2.pt')
        if not os.path.exists(path_s1):
            path_s1 = os.path.join(dirpath, 'predictor_stage1.pt')
        if not os.path.exists(path_s2):
            path_s2 = os.path.join(dirpath, 'predictor_stage2.pt')
        # 若只有 checkpoint_epoch_*.pt，用最新的
        if not os.path.exists(path_s1) or not os.path.exists(path_s2):
            import glob
            ckpts = sorted(glob.glob(os.path.join(dirpath, 'checkpoint_epoch_*.pt')))
            if ckpts:
                path_s1 = path_s2 = ckpts[-1]
            else:
                raise FileNotFoundError(
                    f"No valid checkpoint in {dirpath}. Need best_predictor_stage1.pt+stage2.pt or predictor_stage1.pt+stage2.pt or checkpoint_epoch_*.pt"
                )
    # 3. --checkpoint_stage1 + --checkpoint_stage2
    else:
        path_s1 = args.checkpoint_stage1
        path_s2 = args.checkpoint_stage2
    
    if path_s1 is None or path_s2 is None:
        raise ValueError(
            "Must provide one of: --checkpoint <file>, --checkpoint_dir <dir>, "
            "or --checkpoint_stage1 + --checkpoint_stage2"
        )
    if not os.path.exists(path_s1):
        raise FileNotFoundError(f"Stage1 checkpoint not found: {path_s1}")
    if not os.path.exists(path_s2):
        raise FileNotFoundError(f"Stage2 checkpoint not found: {path_s2}")
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    print(f"\n>>> Loading two-stage predictors:")
    print(f"    Stage1: {path_s1}")
    print(f"    Stage2: {path_s2}")
    
    predictor_stage1, predictor_stage2 = load_two_stage_predictors(path_s1, path_s2, device=device)
    # 加载后预计算并缓存逆向权重，推理时不再调用 torch.linalg.inv
    predictor_stage1.register_inverse_weights()
    predictor_stage2.register_inverse_weights()
    predictor_stage1.eval()
    predictor_stage2.eval()
    
    set_predictor_two_stage(predictor_stage1, predictor_stage2)
    
    if get_predictor() is None:
        raise RuntimeError("Failed to set global predictor!")
    print(f"✓ Two-stage predictors loaded, device: {next(predictor_stage1.parameters()).device}")
    
    num_params = sum(p.numel() for p in predictor_stage1.parameters())
    print(f"  Parameters per stage: {num_params:,} ({num_params/1e6:.1f}M)")
    
    print(f"\n>>> Loading Qwen-Image pipeline from: {args.model_path}")
    pipe = QwenImagePipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
    ).to(device)
    pipe = pipeline_with_learned_cache(pipe)
    print("✓ Pipeline loaded with two-stage learned cache")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_load_end = time.perf_counter()
    
    prompts = read_prompts(args.prompt_file)
    if args.limit is not None:
        prompts = prompts[:args.limit]
        print(f"\n>>> Limited to {len(prompts)} prompts (--limit {args.limit})")
    print(f"\n>>> Loaded {len(prompts)} prompts from: {args.prompt_file}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\n>>> Generating images...")
    print(f"    Seed: {args.seed}, Interval: {args.interval}")
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
    
    total_images = len(prompts) * args.num_images_per_prompt
    missing = collect_missing_images(prompts, args.num_images_per_prompt, args.output_dir)
    if len(missing) < total_images:
        print(f">>> Resuming: {len(missing)} images to generate, {total_images - len(missing)} already exist")
    if not missing:
        print(f">>> All {total_images} images already exist, nothing to generate")
    num_timed = min(args.num_timed_images, len(missing)) if args.timing_file else 0
    inference_times = []
    
    for idx, (img_idx, i, prompt) in enumerate(tqdm(missing, desc="Generating")):
        seed = args.seed + i + img_idx * len(prompts)
        generator = torch.Generator(device).manual_seed(seed)
        cache_dic, current = cache_init(cache_kwargs)
        
        do_time = args.timing_file and idx < num_timed
        if do_time and torch.cuda.is_available():
            torch.cuda.synchronize()
        t_infer_start = time.perf_counter() if do_time else None
        
        with torch.inference_mode():
            result = pipe(
                prompt=prompt,
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
        
        if do_time:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            inference_times.append(time.perf_counter() - t_infer_start)
        
        filename = f"img_{i:04d}_{img_idx}.jpg" if args.num_images_per_prompt > 1 else f"img_{i:04d}.jpg"
        output_path = os.path.join(args.output_dir, filename)
        
        exif_data = Image.Exif()
        exif_data[ExifTags.Base.Software] = "AI generated;t2i;qwen-learned-v4-two-stage"
        exif_data[ExifTags.Base.Make] = "Qwen"
        exif_data[ExifTags.Base.Model] = "learned-decomposition-v4-two-stage"
        if args.add_metadata:
            exif_data[ExifTags.Base.ImageDescription] = prompt
        
        image.save(output_path, exif=exif_data, quality=95, subsampling=0)
        
        del result, generator, cache_dic, current, image
        if (idx + 1) % 5 == 0:
            torch.cuda.empty_cache()
    
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    
    if args.timing_file and inference_times:
        load_sec = t_load_end - t_start
        avg_inference_sec = sum(inference_times) / len(inference_times)
        exp_name = args.exp_name or os.path.basename(args.output_dir.rstrip('/'))
        os.makedirs(os.path.dirname(args.timing_file) or '.', exist_ok=True)
        with open(args.timing_file, 'a') as f:
            f.write(f"{exp_name}\t{args.interval}\t{load_sec:.2f}\t{avg_inference_sec:.2f}\t{len(inference_times)}\n")
        print(f"\n[TIMING] load_sec={load_sec:.2f} (excluded) | avg_inference_sec={avg_inference_sec:.2f} over {len(inference_times)} images -> {args.timing_file}")
    
    print(f"\n>>> Done! Generated {total_images} images in: {args.output_dir}")


if __name__ == '__main__':
    main()
