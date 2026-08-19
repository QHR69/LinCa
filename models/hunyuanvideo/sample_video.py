import os
import sys
import json
import time
import fcntl
from pathlib import Path
from loguru import logger
from datetime import datetime

from hyvideo.utils.file_utils import save_videos_grid
from hyvideo.config import parse_args
from hyvideo.inference import HunyuanVideoSampler


def claim_next_index(counter_path, total):
    """Atomically claim the next prompt index from a shared counter; None means done"""
    with open(counter_path, 'r+') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            current = int(f.read().strip())
            if current >= total:
                return None
            f.seek(0)
            f.write(str(current + 1))
            f.truncate()
            return current
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def main():
    args = parse_args()
    models_root_path = Path(args.model_base)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")

    # Create save folder to save the samples
    save_path = args.save_path if args.save_path_suffix == "" else f'{args.save_path}_{args.save_path_suffix}'
    os.makedirs(save_path, exist_ok=True)

    # Load learned predictor if using learned decomposition
    if getattr(args, 'decompose_method', 'FFT') == 'learned' and getattr(args, 'learned_checkpoint', None):
        import torch
        from flux.modules.cache_functions.cache_utils_learned import load_predictor
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading learned predictor from {args.learned_checkpoint}")
        load_predictor(args.learned_checkpoint, device=device)
        logger.info("Learned predictor loaded successfully")

    # Load models
    hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(models_root_path, args=args)
    args = hunyuan_video_sampler.args

    # Determine prompts: VBench JSON or single prompt
    vbench_json_path = getattr(args, "vbench_json_path", None)
    index_start = int(getattr(args, "index_start", 0))
    index_end = int(getattr(args, "index_end", -1))

    prompt_stride = int(getattr(args, "prompt_stride", 1))

    if vbench_json_path:
        with open(vbench_json_path, 'r') as f:
            prompts_data = json.load(f)
        if index_end < 0 or index_end >= len(prompts_data):
            index_end = len(prompts_data) - 1
        # Apply stride: pick every prompt_stride-th prompt within range
        selected_prompts = [item['prompt_en'] for item in prompts_data[index_start:index_end + 1:prompt_stride]]
        logger.info(f"VBench: {len(selected_prompts)} prompts [{index_start}, {index_end}] stride={prompt_stride}")
    else:
        selected_prompts = [args.prompt]

    # Start sampling — dynamic dispatch: each GPU claims the next prompt from a shared counter
    total = len(selected_prompts)
    counter_path = os.path.join(save_path, ".dispatch_counter")

    # Create the counter file if it does not exist
    if not os.path.exists(counter_path):
        with open(counter_path, 'w') as f:
            f.write('0')

    logger.info(f"Dynamic dispatch: total={total} prompts, counter={counter_path}")

    generated = 0
    while True:
        idx = claim_next_index(counter_path, total)
        if idx is None:
            break

        prompt_text = selected_prompts[idx]
        safe_name = prompt_text.replace('/', '').replace('\\', '')
        cur_save_path = f"{save_path}/{safe_name}-0.mp4"

        # Skip if already exists
        if os.path.exists(cur_save_path):
            logger.info(f"[{idx+1}/{total}] Skipping (exists): {cur_save_path}")
            continue

        logger.info(f"[{idx+1}/{total}] {prompt_text[:80]}...")

        outputs = hunyuan_video_sampler.predict(
            prompt=prompt_text,
            height=args.video_size[0],
            width=args.video_size[1],
            video_length=args.video_length,
            seed=args.seed,
            negative_prompt=args.neg_prompt,
            infer_steps=args.infer_steps,
            guidance_scale=args.cfg_scale,
            num_videos_per_prompt=1,
            flow_shift=args.flow_shift,
            batch_size=1,
            embedded_guidance_scale=args.embedded_cfg_scale,
        )
        samples = outputs['samples']
        for i, sample in enumerate(samples):
            sample = samples[i].unsqueeze(0)
            save_videos_grid(sample, cur_save_path, fps=24)
            logger.info(f'Saved: {cur_save_path}')
        generated += 1

    logger.info(f"Sampling complete! Generated {generated}/{total} videos, saved to: {save_path}")


if __name__ == "__main__":
    main()
