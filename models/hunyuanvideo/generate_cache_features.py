"""
Generate cached features for offline training（HunyuanVideo版本，单pass，和Flux一致）

HunyuanVideo是guidance-distilled模型，推理时只做1次forward（embedded guidance）
不需要传统CFG的cond/uncond两次forward

Data format:
    cache_data/
    ├── prompt_0000/
    │   ├── step_00.pt  # [seq_len, 3072]
    │   ├── step_01.pt
    │   └── ...
    └── prompt_0001/
        └── ...

使用方法 (VBench JSON格式):
    python generate_cache_features.py \
        --vbench-json-path eval/VBench_full_info.json \
        --output_dir cache_data \
        --index-start 0 \
        --index-end 945 \
        --video-size 480 640 \
        --video-length 65 \
        --infer-steps 50 \
        --flow-reverse \
        --seed 42 \
        ... (remaining flags match sample_video.py)

使用方法 (文本文件格式):
    python generate_cache_features.py \
        --prompt_file prompts/train.txt \
        --output_dir cache_data \
        --start_idx 0 \
        --end_idx 100 \
        --video-size 480 640 \
        --video-length 65 \
        --infer-steps 50 \
        --flow-reverse \
        --seed 42 \
        ... (remaining flags match sample_video.py)
"""

import os
import sys
import json
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
from loguru import logger

# Add src/ to path so we can import both hyvideo and flux
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from hyvideo.config import parse_args
from hyvideo.inference import HunyuanVideoSampler
from hyvideo.constants import PRECISION_TO_TYPE
from hyvideo.diffusion.pipelines import HunyuanVideoPipeline
from hyvideo.diffusion.schedulers import FlowMatchDiscreteScheduler
from hyvideo.utils.data_utils import align_to

from flux.modules.cache_functions.cache_init import cache_init
from flux.modules.cache_functions.cal_type import cal_type

from diffusers.utils.torch_utils import randn_tensor


def read_prompts(prompt_file: str):
    """Read the prompt file（文本格式，每行一个prompt）"""
    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts


def read_prompts_from_json(json_path: str):
    """读取VBench JSON格式的prompt文件"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    prompts = [item['prompt_en'] for item in data]
    return prompts


def generate_features_for_prompt(
    sampler,
    prompt,
    prompt_idx,
    output_dir,
    num_steps=50,
    seed=42,
):
    """为单个 prompt 生成并保存每步的特征（单pass，和Flux一致）"""

    args = sampler.args
    device = torch.device(sampler.device) if isinstance(sampler.device, str) else sampler.device
    pipeline = sampler.pipeline

    height = args.video_size[0]
    width = args.video_size[1]
    video_length = args.video_length

    target_height = align_to(height, 16)
    target_width = align_to(width, 16)

    # ---- Text Encoding (cond only, no CFG) ----
    prompt_list = [prompt.strip()]

    target_dtype = PRECISION_TO_TYPE[args.precision]

    (
        prompt_embeds,
        _,
        prompt_mask,
        _,
    ) = pipeline.encode_prompt(
        prompt_list,
        device,
        num_videos_per_prompt=1,
        do_classifier_free_guidance=False,
        data_type="video" if video_length > 1 else "image",
    )

    if pipeline.text_encoder_2 is not None:
        (
            prompt_embeds_2,
            _,
            prompt_mask_2,
            _,
        ) = pipeline.encode_prompt(
            prompt_list,
            device,
            num_videos_per_prompt=1,
            do_classifier_free_guidance=False,
            text_encoder=pipeline.text_encoder_2,
            data_type="video" if video_length > 1 else "image",
        )
    else:
        prompt_embeds_2 = None

    # ---- Scheduler ----
    scheduler = FlowMatchDiscreteScheduler(
        shift=args.flow_shift,
        reverse=args.flow_reverse,
        solver=args.flow_solver,
    )
    scheduler.set_timesteps(num_steps, device=device)
    timesteps = scheduler.timesteps

    # ---- Latents ----
    vae_ver = args.vae
    vl = video_length
    if "884" in vae_ver:
        vl = (video_length - 1) // 4 + 1
    elif "888" in vae_ver:
        vl = (video_length - 1) // 8 + 1

    num_channels_latents = pipeline.transformer.config.in_channels
    # Use same seed for all prompts (matching VBench: seed=42 for first video of each prompt)
    generator = torch.Generator(device).manual_seed(seed)
    latents = randn_tensor(
        (1, num_channels_latents, vl, target_height // pipeline.vae_scale_factor, target_width // pipeline.vae_scale_factor),
        generator=generator,
        device=device,
        dtype=prompt_embeds.dtype,
    )
    if hasattr(scheduler, "init_noise_sigma"):
        latents = latents * scheduler.init_noise_sigma

    # ---- RoPE ----
    freqs_cos, freqs_sin = sampler.get_rotary_pos_embed(video_length, target_height, target_width)

    # ---- Embedded guidance ----
    embedded_guidance_scale = getattr(args, 'embedded_cfg_scale', None)

    # ---- Cache init (all steps = full, collect all features) ----
    cache_dic, current = cache_init(
        num_steps=num_steps,
        test_FLOPs=False,
        monitor_gpu_usage=False,
        interval=1,
        max_order=0,
        min_order=0,
        first_enhance=num_steps,  # all steps are full
        forecast_method='hermite',
        decompose_method='None',
        use_z_cache=False,
        forecast_steps=1,
    )

    autocast_enabled = (target_dtype != torch.float32) and not args.disable_autocast

    # ---- Create output directory ----
    prompt_dir = output_dir / f"prompt_{prompt_idx:04d}"
    prompt_dir.mkdir(parents=True, exist_ok=True)

    # ---- Denoising Loop (single pass, like Flux) ----
    with torch.no_grad():
        for i, t in enumerate(timesteps):
            current['step'] = i
            # cal_type manages current['type'] and current['activated_steps']
            # With first_enhance=num_steps, all steps are 'full'
            cal_type(cache_dic, current)

            latent_model_input = scheduler.scale_model_input(latents, t)

            t_expand = t.repeat(latent_model_input.shape[0])
            guidance_expand = (
                torch.tensor(
                    [embedded_guidance_scale] * latent_model_input.shape[0],
                    dtype=torch.float32,
                    device=device,
                ).to(target_dtype) * 1000.0
                if embedded_guidance_scale is not None
                else None
            )

            with torch.autocast(device_type="cuda", dtype=target_dtype, enabled=autocast_enabled):
                noise_pred = pipeline.transformer(
                    latent_model_input,
                    t_expand,
                    text_states=prompt_embeds,
                    text_mask=prompt_mask,
                    text_states_2=prompt_embeds_2,
                    freqs_cos=freqs_cos,
                    freqs_sin=freqs_sin,
                    guidance=guidance_expand,
                    return_dict=True,
                    cache_dic=cache_dic,
                    current=current,
                )["x"]

            # Scheduler step
            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            # ---- Extract and save features ----
            # Model sets current['module'] = 'denoise'
            # For decompose_method='None': cache[-1]['denoise'] = {0: feature, 1: deriv, ...}
            if -1 in cache_dic['cache'] and 'denoise' in cache_dic['cache'][-1]:
                denoise_cache = cache_dic['cache'][-1]['denoise']
                if isinstance(denoise_cache, dict) and 0 in denoise_cache:
                    feature = denoise_cache[0]
                    if feature.dim() == 3 and feature.shape[0] == 1:
                        feature = feature.squeeze(0)
                    torch.save(feature.cpu(), prompt_dir / f"step_{i:02d}.pt")

    # Cleanup
    del latents, cache_dic, current
    torch.cuda.empty_cache()


def main():
    # First, extract our custom args that HunyuanVideo's parse_args doesn't know about
    # Must do this BEFORE calling parse_args() which uses parser.parse_args() (strict)
    extra_parser = argparse.ArgumentParser(add_help=False)
    extra_parser.add_argument('--output_dir', type=str, required=True)
    extra_parser.add_argument('--prompt_file', type=str, default=None)
    extra_parser.add_argument('--start_idx', type=int, default=0)
    extra_parser.add_argument('--end_idx', type=int, default=None)
    extra_args, remaining_argv = extra_parser.parse_known_args()

    # Replace sys.argv so HunyuanVideo's parse_args only sees its own args
    original_argv = sys.argv
    sys.argv = [sys.argv[0]] + remaining_argv

    # Now parse HunyuanVideo standard args
    args = parse_args()

    # Restore sys.argv
    sys.argv = original_argv

    # Get params from both sources
    vbench_json_path = getattr(args, 'vbench_json_path', None)
    prompt_file = extra_args.prompt_file
    output_dir_str = extra_args.output_dir
    index_start = int(getattr(args, 'index_start', 0))
    index_end = int(getattr(args, 'index_end', 0))
    start_idx = extra_args.start_idx
    end_idx = extra_args.end_idx

    output_dir = Path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read prompts: prefer VBench JSON, fallback to text file
    if vbench_json_path is not None:
        prompts = read_prompts_from_json(vbench_json_path)
        logger.info(f"Read {len(prompts)} prompts from VBench JSON: {vbench_json_path}")
        # Use index_start/index_end (inclusive) for VBench JSON
        if index_end <= 0 or index_end >= len(prompts):
            index_end = len(prompts) - 1
        prompts_subset = prompts[index_start:index_end + 1]
        start_offset = index_start
        logger.info(f"Processing VBench prompt range: [{index_start}, {index_end}] ({len(prompts_subset)} prompts)")
    elif prompt_file is not None:
        prompts = read_prompts(prompt_file)
        logger.info(f"Read {len(prompts)} prompts from text file: {prompt_file}")
        if end_idx is not None:
            prompts_subset = prompts[start_idx:end_idx]
        else:
            prompts_subset = prompts[start_idx:]
        start_offset = start_idx
        logger.info(f"Processing prompt range: {start_idx} - {start_idx + len(prompts_subset)}")
    else:
        raise ValueError("Must provide either --vbench-json-path or --prompt_file")

    # Load model
    models_root_path = Path(args.model_base)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")

    logger.info("Loading HunyuanVideo model...")
    sampler = HunyuanVideoSampler.from_pretrained(models_root_path, args=args)
    args = sampler.args
    logger.info("Model loaded")

    num_steps = args.infer_steps

    logger.info(f"Starting feature generation (single pass, like Flux)")
    logger.info(f"  Prompts: {len(prompts_subset)}")
    logger.info(f"  Steps: {num_steps}")
    logger.info(f"  Seed: {getattr(args, 'seed', 0)}")
    logger.info(f"  Video size: {args.video_size}")
    logger.info(f"  Video length: {args.video_length}")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"  Format: prompt_XXXX/step_XX.pt")

    skipped = 0
    for i, prompt in enumerate(tqdm(prompts_subset, desc="Processing prompts")):
        global_idx = start_offset + i

        # Skip if all steps already exist
        prompt_dir = output_dir / f"prompt_{global_idx:04d}"
        last_step_file = prompt_dir / f"step_{num_steps - 1:02d}.pt"
        if last_step_file.exists():
            skipped += 1
            continue

        logger.info(f"[{global_idx}] {prompt[:80]}...")

        generate_features_for_prompt(
            sampler=sampler,
            prompt=prompt,
            prompt_idx=global_idx,
            output_dir=output_dir,
            num_steps=num_steps,
            seed=getattr(args, 'seed', 0),
        )

    if skipped > 0:
        logger.info(f"Skipped {skipped} prompts (already completed)")

    logger.info(f"Feature generation complete! Saved to: {output_dir}")


if __name__ == '__main__':
    main()
