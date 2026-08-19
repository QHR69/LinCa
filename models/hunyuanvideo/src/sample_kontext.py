import os
from dataclasses import dataclass
from pathlib import Path
from PIL import Image
import torch
from einops import rearrange
from datasets import load_from_disk
from tqdm import tqdm
from typing import cast

from flux.sampling import get_schedule, prepare_kontext, unpack, denoise_cache
from flux.util import configs, embed_watermark, load_ae, load_clip, load_flow_model, load_t5


@dataclass
class SamplingOptions:
    dataset_path: str           # Path to the GEdit dataset
    num_steps: int              # Number of sampling steps
    guidance: float             # Guidance value
    seed: int                   # Random seed
    model_name: str             # Model name
    output_dir: str             # Output directory
    english_only: bool          # Whether to process only English tasks
    test_FLOPs: bool            # Whether in FLOPs test mode
    monitor_gpu_usage: bool     # Whether to monitor GPU memory usage
    interval: int               # Cache period length
    max_order: int              # Maximum order of expansion
    min_order: int              # Minimum order of expansion
    first_enhance: int          # Initial enhancement steps
    forecast_method: str        # Forecast method
    decompose_method: str       # Decomposition method
    use_z_cache: bool           # Use Z cache
    forecast_steps: int         # Forecast steps


def create_folders(output_dir: str, task_types: list[str], languages: list[str]):
    """Create directory structure"""
    base_dir = Path(output_dir) / "fullset"
    
    for task_type in task_types:
        for lang in languages:
            task_dir = base_dir / task_type / lang
            task_dir.mkdir(parents=True, exist_ok=True)


def check_images(output_dir: str, task_type: str, instruction_language: str, key: str) -> bool:
    """Check if image has already been generated"""
    image_path = Path(output_dir) / "fullset" / task_type / instruction_language / f"{key}.png"
    return image_path.exists()


def save_image(image: Image.Image, output_dir: str, task_type: str, instruction_language: str, key: str):
    """Save generated image to corresponding directory"""
    save_dir = Path(output_dir) / "fullset" / task_type / instruction_language
    save_dir.mkdir(parents=True, exist_ok=True)
    
    save_path = save_dir / f"{key}.png"
    image.save(save_path)


def main(opts: SamplingOptions):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load dataset
    print(f"Loading dataset: {opts.dataset_path}")
    dataset = load_from_disk(opts.dataset_path)
    print(f"Dataset loaded, total {len(dataset)} samples")
    
    # Get task types and languages
    task_types: list[str] = list(set(str(dataset[i]["task_type"]) for i in range(len(dataset))))
    languages = ["en", "cn"] if not opts.english_only else ["en"]
    
    print(f"Found task types: {task_types}")
    if opts.english_only:
        print("English-only mode enabled: will skip Chinese tasks")
    
    # Create output directory structure
    create_folders(opts.output_dir, task_types, languages)
    
    # Load model
    model_name = opts.model_name
    if model_name not in configs:
        available = ", ".join(configs.keys())
        raise ValueError(f"Unknown model name: {model_name}, available options: {available}")
    
    # Initialize model components
    t5 = load_t5(device, max_length=512)
    clip = load_clip(device)
    model = load_flow_model(model_name, device=device)
    ae = load_ae(model_name, device=device)
    
    # Process dataset
    processed_count = 0
    skipped_count = 0

    progress_bar = tqdm(total=len(dataset), desc="Generating images")
    
    for i in range(len(dataset)):
        item = dataset[i]
        task_type: str = str(item["task_type"])
        instruction: str = str(item["instruction"])
        instruction_language: str = str(item["instruction_language"])
        key: str = str(item["key"])
        input_image = item["input_image"]
        
        # Skip Chinese tasks (if english_only enabled)
        if opts.english_only and instruction_language == "cn":
            progress_bar.update(1)
            skipped_count += 1
            continue
            
        # Check if already exists
        if check_images(opts.output_dir, task_type, instruction_language, key):
            progress_bar.update(1)
            skipped_count += 1
            continue
        
        # Set random seed
        seed = opts.seed + i
        
        batch_prompts = [instruction]
        input_image = cast(Image.Image, input_image)

        inp, target_height, target_width = prepare_kontext(t5, clip, batch_prompts, ae, img_cond_pil=input_image, device=device, seed=seed)
        inp.pop("img_cond_orig", None)
        
        timesteps = get_schedule(opts.num_steps, inp["img"].shape[1], shift=True)
        
        kwargs = {
            'num_steps': opts.num_steps,
            'test_FLOPs': opts.test_FLOPs,
            'monitor_gpu_usage': opts.monitor_gpu_usage,
            'interval': opts.interval,
            'max_order': opts.max_order,
            'min_order': opts.min_order,
            'first_enhance': opts.first_enhance,
            'forecast_method': opts.forecast_method,
            'decompose_method': opts.decompose_method,
            'use_z_cache': opts.use_z_cache,
            'forecast_steps': opts.forecast_steps,
        }

        # Denoising
        with torch.no_grad():
            x = denoise_cache(model, **inp, timesteps=timesteps, guidance=opts.guidance, **kwargs)
            
            # Decode latent variables
            x = unpack(x.float(), target_height, target_width)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                x = ae.decode(x)
        
        # Convert to PIL format and save
        x = x.clamp(-1, 1)
        x = embed_watermark(x.float())
        x = rearrange(x, "b c h w -> b h w c")

        img_array = x[0]
        ima = Image.fromarray((127.5 * (img_array + 1.0)).cpu().byte().numpy())

        save_image(ima, opts.output_dir, task_type, instruction_language, key)
        processed_count += 1

        progress_bar.update(1)

    progress_bar.close()
    
    print(f"Generation complete! Processed {processed_count} images, skipped {skipped_count} images")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate GEdit-Bench images using Flux Kontext model.")
    parser.add_argument('--dataset_path', type=str, default='/data/public/.cache/huggingface/hub/datasets--stepfun-ai--GEdit-Bench/snapshots/50766778e2a737474c7e9bdf84cdce82c3ea3f4f', help='Path to the GEdit dataset.')
    parser.add_argument('--num_steps', type=int, default=50, help='Number of sampling steps.')
    parser.add_argument('--guidance', type=float, default=3.5, help='Guidance value.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--model_name', type=str, default='flux-dev-kontext', help='Model name.')
    parser.add_argument('--output_dir', type=str, default='results/teacache_sample', help='Directory to save images.')
    parser.add_argument('--english_only', action='store_true', help='Process only English tasks.')
    parser.add_argument('--test_FLOPs', action='store_true', help='Test inference computation cost.')
    parser.add_argument('--monitor_gpu_usage', action='store_true', help='Monitor GPU memory usage during sampling.')

    parser.add_argument('--interval', type=int, default=1)
    parser.add_argument('--max_order', type=int, default=2)
    parser.add_argument('--min_order', type=int, default=0)
    parser.add_argument('--first_enhance', type=int, default=3)
    parser.add_argument('--forecast_method', type=str, default='hermite', choices=['taylor', 'hermite'])
    parser.add_argument('--decompose_method', type=str, default='FFT', choices=['None', 'FFT', 'DCT'])
    parser.add_argument('--use_z_cache', action='store_true')
    parser.add_argument('--forecast_steps', type=int, default=1)

    args = parser.parse_args()
    
    opts = SamplingOptions(
        dataset_path=args.dataset_path,
        num_steps=args.num_steps,
        guidance=args.guidance,
        seed=args.seed,
        model_name=args.model_name,
        output_dir=args.output_dir,
        english_only=args.english_only,
        test_FLOPs=args.test_FLOPs,
        monitor_gpu_usage=args.monitor_gpu_usage,
        interval=args.interval,
        max_order=args.max_order,
        min_order=args.min_order,
        first_enhance=args.first_enhance,
        forecast_method=args.forecast_method,
        decompose_method=args.decompose_method,
        use_z_cache=args.use_z_cache,
        forecast_steps=args.forecast_steps,
    )
    
    main(opts)
    # CUDA_VISIBLE_DEVICES=2 python src/sample_kontext.py --output_dir results/freqca_N7_Z --english_only --interval 7 --use_z_cache --forecast_steps 7