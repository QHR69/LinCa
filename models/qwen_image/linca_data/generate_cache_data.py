"""
Data generation script for training the invertible decomposition network.
Saves 50-step full computation intermediate features (after norm_out, before proj_out).

Usage:
    python generate_cache_data.py --prompt_file prompts/prompts_train.txt --output_dir data/cache_data
"""

import os
import json
import torch
import argparse
from PIL import Image, ExifTags
from tqdm import tqdm
from dataclasses import dataclass, asdict
from typing import List, Optional
import types

from pipeline.pipeline_qwenimage_data import QwenImagePipelineForData
from pipeline.transformer_qwenimage_data import QwenImageTransformer2DModelForData


@dataclass
class DataGenConfig:
    prompt_file: str = 'prompts/prompts_train.txt'
    negative_prompt: str = " "  # Empty text for CFG
    output_dir: str = 'data/cache_data'
    image_dir_first200: str = 'data/images_first200'  # Images for prompts 0-199 (for PSNR/SSIM)
    image_dir_last200: str = 'data/images_last200'    # Images for prompts 200-399
    width: int = 1328
    height: int = 1328
    num_steps: int = 50
    guidance_scale: float = 1.0
    true_cfg_scale: float = 4.0  # CFG scale, >1 to enable CFG
    seed: int = 0  # Base seed, actual seed = base_seed + prompt_idx
    max_sequence_length: int = 512


def read_prompts(prompt_file: str) -> List[str]:
    """Read prompts from file, skip empty lines."""
    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts


def setup_transformer_for_data(pipe):
    """
    Replace transformer's forward method to return intermediate features.
    This patches the original transformer to use our data generation version.
    """
    from pipeline.transformer_qwenimage_data import QwenImageTransformer2DModelForData
    
    # Patch the forward method
    pipe.transformer.forward = types.MethodType(
        QwenImageTransformer2DModelForData.forward, 
        pipe.transformer
    )
    return pipe


def main(config: DataGenConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load pipeline
    print("Loading pipeline...")
    pipe = QwenImagePipelineForData.from_pretrained(
        "Qwen/Qwen-Image",
        torch_dtype=torch.bfloat16
    ).to(device=device)
    
    # Patch transformer for data generation
    pipe = setup_transformer_for_data(pipe)
    
    # Read prompts
    prompts = read_prompts(config.prompt_file)
    print(f"Loaded {len(prompts)} prompts from {config.prompt_file}")
    
    # Create output directories
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.image_dir_first200, exist_ok=True)
    os.makedirs(config.image_dir_last200, exist_ok=True)
    
    # Dataset index
    index_data = {
        'config': asdict(config),
        'num_prompts': len(prompts),
        'num_steps': config.num_steps,
        'feature_dim': 3072,
        'prompts': []
    }
    
    # Generate data for each prompt
    for idx, prompt in enumerate(tqdm(prompts, desc="Generating cache data")):
        # Create prompt directory
        prompt_dir = os.path.join(config.output_dir, f'prompt_{idx:04d}')
        cond_dir = os.path.join(prompt_dir, 'cond')
        uncond_dir = os.path.join(prompt_dir, 'uncond')
        
        os.makedirs(cond_dir, exist_ok=True)
        os.makedirs(uncond_dir, exist_ok=True)
        
        # Set seed: base_seed + prompt_idx (same as sample.py logic)
        seed = config.seed + idx
        generator = torch.Generator(device).manual_seed(seed)
        
        try:
            # Generate and get intermediate features
            cache_data = pipe.generate_and_save_cache(
                prompt=prompt,
                negative_prompt=config.negative_prompt,
                height=config.height,
                width=config.width,
                num_inference_steps=config.num_steps,
                guidance_scale=config.guidance_scale,
                true_cfg_scale=config.true_cfg_scale,
                generator=generator,
                max_sequence_length=config.max_sequence_length,
            )
            
            # Save features for each step
            for step in range(config.num_steps):
                # Cond branch (noise_pred)
                cond_feature = cache_data['cond'][step]  # [seq_len, 3072]
                torch.save(cond_feature, os.path.join(cond_dir, f'step_{step:02d}.pt'))
                
                # Uncond branch (neg_noise_pred)
                if len(cache_data['uncond']) > 0:
                    uncond_feature = cache_data['uncond'][step]  # [seq_len, 3072]
                    torch.save(uncond_feature, os.path.join(uncond_dir, f'step_{step:02d}.pt'))
            
            # Save image
            img = cache_data['image']
            
            # Add EXIF metadata
            exif_data = Image.Exif()
            exif_data[ExifTags.Base.Software] = "AI generated;t2i;qwen"
            exif_data[ExifTags.Base.Make] = "Qwen"
            exif_data[ExifTags.Base.Model] = "qwen-image"
            exif_data[ExifTags.Base.ImageDescription] = prompt
            
            # Save to appropriate directory based on index
            if idx < 200:
                # First 200 prompts -> for PSNR/SSIM evaluation
                img_path = os.path.join(config.image_dir_first200, f'img_{idx:04d}.jpg')
            else:
                # Last 200 prompts -> additional training data
                img_path = os.path.join(config.image_dir_last200, f'img_{idx:04d}.jpg')
            
            img.save(img_path, exif=exif_data, quality=95, subsampling=0)
            
            # Save metadata for this prompt
            metadata = {
                'prompt_idx': idx,
                'prompt': prompt,
                'negative_prompt': config.negative_prompt,
                'seed': seed,
                'num_steps': config.num_steps,
                'height': config.height,
                'width': config.width,
                'feature_dim': 3072,
                'seq_length': cache_data['seq_length'],
                'image_path': img_path,
            }
            with open(os.path.join(prompt_dir, 'metadata.json'), 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            # Update index
            index_data['prompts'].append({
                'idx': idx,
                'prompt': prompt,
                'dir': f'prompt_{idx:04d}',
                'seed': seed,
                'seq_length': cache_data['seq_length'],
            })
            
        except Exception as e:
            print(f"Error processing prompt {idx}: {prompt[:50]}...")
            print(f"Error: {e}")
            continue
        
        # Clear cache periodically
        if idx % 10 == 0:
            torch.cuda.empty_cache()
    
    # Save index file
    index_path = os.path.join(config.output_dir, 'index.json')
    with open(index_path, 'w', encoding='utf-8') as f:
        json.dump(index_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nData generation complete!")
    print(f"Cache data saved to: {config.output_dir}")
    print(f"First 200 images saved to: {config.image_dir_first200}")
    print(f"Last 200 images saved to: {config.image_dir_last200}")
    print(f"Index file: {index_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate cache data for training invertible decomposition network")
    parser.add_argument('--prompt_file', type=str, default='prompts/prompts_train.txt',
                        help='Path to the prompt text file')
    parser.add_argument('--output_dir', type=str, default='data/cache_data',
                        help='Directory to save cache data')
    parser.add_argument('--image_dir_first200', type=str, default='data/images_first200',
                        help='Directory to save images for first 200 prompts')
    parser.add_argument('--image_dir_last200', type=str, default='data/images_last200',
                        help='Directory to save images for last 200 prompts')
    parser.add_argument('--seed', type=int, default=0,
                        help='Base random seed')
    parser.add_argument('--width', type=int, default=1328,
                        help='Image width')
    parser.add_argument('--height', type=int, default=1328,
                        help='Image height')
    parser.add_argument('--num_steps', type=int, default=50,
                        help='Number of inference steps')
    parser.add_argument('--true_cfg_scale', type=float, default=4.0,
                        help='CFG scale (>1 to enable CFG)')
    
    args = parser.parse_args()
    
    config = DataGenConfig(
        prompt_file=args.prompt_file,
        output_dir=args.output_dir,
        image_dir_first200=args.image_dir_first200,
        image_dir_last200=args.image_dir_last200,
        seed=args.seed,
        width=args.width,
        height=args.height,
        num_steps=args.num_steps,
        true_cfg_scale=args.true_cfg_scale,
    )
    
    main(config)
