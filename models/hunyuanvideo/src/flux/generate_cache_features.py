"""
Generate cached features for offline training

Run a Flux forward pass and save the intermediate feature at every denoise step
Data format:
    cache_data/
    ├── prompt_0000/
    │   ├── step_00.pt
    │   ├── step_01.pt
    │   └── ...
    └── prompt_0001/
        └── ...

Usage:
    python src/flux/generate_cache_features.py \
        --prompt_file prompts/train.txt \
        --output_dir cache_data \
        --model_name flux-dev \
        --num_steps 50 \
        --start_idx 0 \
        --end_idx 100
"""

import os
import torch
import argparse
from pathlib import Path
from tqdm import tqdm

from flux.sampling import get_noise, get_schedule, prepare
from flux.util import load_ae, load_clip, load_flow_model, load_t5
from flux.modules.cache_functions.cache_init import cache_init
from flux.modules.cache_functions.cache_utils_learned import module_cache_init, derivative_approximation


def read_prompts(prompt_file: str):
    """Read the prompt file"""
    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts


def generate_cache_features(
    model,
    t5,
    clip,
    ae,
    prompts,
    output_dir,
    device,
    num_steps=50,
    width=1024,
    height=1024,
    seed=0,
):
    """Generate cached features"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generate cached features")
    print(f"  Prompts: {len(prompts)}")
    print(f"  Steps: {num_steps}")
    print(f"  Output: {output_dir}")
    print()

    for prompt_idx, prompt in enumerate(tqdm(prompts, desc="Processing prompts")):
        # Create the prompt directory (03d naming, matching the original layout)
        prompt_dir = output_dir / f"prompt_{prompt_idx:03d}"
        prompt_dir.mkdir(exist_ok=True)

        # Sample the initial noise
        x = get_noise(
            1, height, width,
            device=device,
            dtype=torch.bfloat16,
            seed=seed + prompt_idx
        )

        # Prepare the inputs
        inp = prepare(t5, clip, x, [prompt])

        # Build the timestep schedule
        timesteps = get_schedule(num_steps, inp["img"].shape[1], shift=True)

        # Initialise the cache dict
        cache_dic, current = cache_init(
            num_steps=num_steps,
            interval=1,
            max_order=0,
            min_order=0,
            first_enhance=num_steps,  # every step is a full compute
            forecast_method='hermite',
            decompose_method='None',
            use_z_cache=False,
            forecast_steps=1,
        )

        img_input = inp["img"]

        # Denoising loop
        with torch.no_grad():
            for step in range(num_steps):
                current['step'] = step
                current['type'] = 'full'

                # Read the current timestep
                t_curr = timesteps[step]
                t_vec = torch.full(
                    (img_input.shape[0],), t_curr,
                    dtype=img_input.dtype, device=device
                )

                # Read the guidance scale
                guidance_vec = torch.full(
                    (img_input.shape[0],), 3.5,
                    device=device, dtype=img_input.dtype
                )

                # Model forward (derivative_approximation stores the feature automatically)
                pred = model(
                    img=img_input,
                    img_ids=inp["img_ids"],
                    txt=inp["txt"],
                    txt_ids=inp["txt_ids"],
                    y=inp["vec"],
                    timesteps=t_vec,
                    guidance=guidance_vec,
                    cache_dic=cache_dic,
                    current=current,
                )

                # Extract features from cache_dic and save them
                if -1 in cache_dic['cache'] and 'denoise' in cache_dic['cache'][-1]:
                    if 'features' in cache_dic['cache'][-1]['denoise']:
                        if step in cache_dic['cache'][-1]['denoise']['features']:
                            feature = cache_dic['cache'][-1]['denoise']['features'][step]

                            # Save the feature (same layout as save_feature_for_training)
                            feature_path = prompt_dir / f"step_{step:02d}.pt"
                            torch.save({
                                'feature': feature.cpu(),  # [B, N, D]
                                'step': step,
                                'prompt_idx': prompt_idx,
                                'shape': feature.shape,
                            }, feature_path)

                # Update the noise
                if step < num_steps - 1:
                    t_prev = timesteps[step + 1]
                    img_input = img_input + (t_prev - t_curr) * pred

        # Free memory
        del inp, x, cache_dic
        torch.cuda.empty_cache()

    print(f"\nFeature generation finished. Saved under: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate Flux cached features")
    parser.add_argument('--prompt_file', type=str, required=True, help='Path to the prompt file')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--model_name', type=str, default='flux-dev',
                       choices=['flux-dev', 'flux-schnell'], help='Model name')
    parser.add_argument('--num_steps', type=int, default=50, help='number of denoise steps')
    parser.add_argument('--width', type=int, default=1024, help='image width')
    parser.add_argument('--height', type=int, default=1024, help='image height')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--start_idx', type=int, default=0, help='start prompt index')
    parser.add_argument('--end_idx', type=int, default=None, help='end prompt index (exclusive)')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load prompts
    prompts = read_prompts(args.prompt_file)
    print(f"Loaded {len(prompts)}  prompts")

    # Select the prompt range
    if args.end_idx is not None:
        prompts = prompts[args.start_idx:args.end_idx]
        print(f"Prompt range: {args.start_idx} - {args.end_idx}")
    else:
        prompts = prompts[args.start_idx:]
        print(f"Prompt range: {args.start_idx} - {len(prompts) + args.start_idx}")

    print(f"\nLoading model: {args.model_name}")

    # Load the model
    t5 = load_t5(device, max_length=512 if args.model_name == 'flux-dev' else 256)
    clip = load_clip(device)
    model = load_flow_model(args.model_name, device=device)
    ae = load_ae(args.model_name, device=device)

    print("Model loaded\n")

    # Generate features
    generate_cache_features(
        model=model,
        t5=t5,
        clip=clip,
        ae=ae,
        prompts=prompts,
        output_dir=args.output_dir,
        device=device,
        num_steps=args.num_steps,
        width=args.width,
        height=args.height,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
