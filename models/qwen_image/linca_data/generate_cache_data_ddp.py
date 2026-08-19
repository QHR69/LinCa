"""
Multi-GPU (DDP) data generation script for training the invertible decomposition network.
Saves 50-step full computation intermediate features (after norm_out, before proj_out).

The seed logic is consistent with single-GPU: seed = base_seed + global_prompt_idx
This ensures multi-GPU and single-GPU generate identical results.

Usage:
    # 4 GPUs
    torchrun --nproc_per_node=4 generate_cache_data_ddp.py --prompt_file prompts/prompts_train.txt
    
    # Or with CUDA_VISIBLE_DEVICES
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 generate_cache_data_ddp.py
"""

import os
import json
import torch
import torch.distributed as dist
import argparse
from PIL import Image, ExifTags
from tqdm import tqdm
from dataclasses import dataclass, asdict
from typing import List
import types

from pipeline.pipeline_qwenimage_data import QwenImagePipelineForData
from pipeline.transformer_qwenimage_data import QwenImageTransformer2DModelForData


@dataclass
class DataGenConfig:
    prompt_file: str = 'prompts/prompts_train.txt'
    negative_prompt: str = " "
    output_dir: str = 'data/cache_data'
    image_dir_first200: str = 'data/images_first200'
    image_dir_last200: str = 'data/images_last200'
    width: int = 1328
    height: int = 1328
    num_steps: int = 50
    guidance_scale: float = 1.0
    true_cfg_scale: float = 4.0
    seed: int = 0
    max_sequence_length: int = 512


def read_prompts(prompt_file: str) -> List[str]:
    """Read prompts from file, skip empty lines."""
    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts


def setup_transformer_for_data(pipe):
    """Patch transformer's forward method to return intermediate features."""
    pipe.transformer.forward = types.MethodType(
        QwenImageTransformer2DModelForData.forward, 
        pipe.transformer
    )
    return pipe


def main(config: DataGenConfig):
    # Initialize distributed
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = f"cuda:{rank}"
    torch.cuda.set_device(rank)
    
    if rank == 0:
        print(f"Running on {world_size} GPUs")
    
    # Load pipeline
    if rank == 0:
        print("Loading pipeline...")
    pipe = QwenImagePipelineForData.from_pretrained(
        "/root/autodl-tmp/Qwen/Qwen-Image",
        torch_dtype=torch.bfloat16
    ).to(device=device)
    
    # Patch transformer for data generation
    pipe = setup_transformer_for_data(pipe)
    
    # Read prompts
    prompts = read_prompts(config.prompt_file)
    total_prompts = len(prompts)
    if rank == 0:
        print(f"Loaded {total_prompts} prompts from {config.prompt_file}")
    
    # Distribute prompts across GPUs
    # Each GPU processes prompts[start:end]
    per_proc = (total_prompts + world_size - 1) // world_size
    start = rank * per_proc
    end = min(start + per_proc, total_prompts)
    local_prompts = prompts[start:end]
    local_indices = list(range(start, end))
    
    if rank == 0:
        print(f"Each GPU processes ~{per_proc} prompts")
    
    # Create output directories (only rank 0)
    if rank == 0:
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(config.image_dir_first200, exist_ok=True)
        os.makedirs(config.image_dir_last200, exist_ok=True)
    
    # Synchronize before starting
    dist.barrier()
    
    # Local index data
    local_index_data = []
    
    # Progress bar only on rank 0
    iterator = tqdm(zip(local_indices, local_prompts), total=len(local_prompts), 
                    desc=f"GPU {rank}", disable=(rank != 0))
    
    for global_idx, prompt in iterator:
        # Create prompt directory
        prompt_dir = os.path.join(config.output_dir, f'prompt_{global_idx:04d}')
        cond_dir = os.path.join(prompt_dir, 'cond')
        uncond_dir = os.path.join(prompt_dir, 'uncond')
        
        os.makedirs(cond_dir, exist_ok=True)
        os.makedirs(uncond_dir, exist_ok=True)
        
        # IMPORTANT: seed = base_seed + global_prompt_idx
        # This is the same logic as single-GPU, ensuring identical results
        seed = config.seed + global_idx
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
                # Cond branch
                cond_feature = cache_data['cond'][step]
                torch.save(cond_feature, os.path.join(cond_dir, f'step_{step:02d}.pt'))
                
                # Uncond branch
                if len(cache_data['uncond']) > 0:
                    uncond_feature = cache_data['uncond'][step]
                    torch.save(uncond_feature, os.path.join(uncond_dir, f'step_{step:02d}.pt'))
            
            # Save image
            img = cache_data['image']
            
            # Add EXIF metadata
            exif_data = Image.Exif()
            exif_data[ExifTags.Base.Software] = "AI generated;t2i;qwen"
            exif_data[ExifTags.Base.Make] = "Qwen"
            exif_data[ExifTags.Base.Model] = "qwen-image"
            exif_data[ExifTags.Base.ImageDescription] = prompt
            
            # Save to appropriate directory based on global index
            if global_idx < 200:
                img_path = os.path.join(config.image_dir_first200, f'img_{global_idx:04d}.jpg')
            else:
                img_path = os.path.join(config.image_dir_last200, f'img_{global_idx:04d}.jpg')
            
            img.save(img_path, exif=exif_data, quality=95, subsampling=0)
            
            # Save metadata
            metadata = {
                'prompt_idx': global_idx,
                'prompt': prompt,
                'negative_prompt': config.negative_prompt,
                'seed': seed,
                'num_steps': config.num_steps,
                'height': config.height,
                'width': config.width,
                'feature_dim': 3072,
                'seq_length': cache_data['seq_length'],
                'image_path': img_path,
                'generated_by_gpu': rank,
            }
            with open(os.path.join(prompt_dir, 'metadata.json'), 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            # Update local index
            local_index_data.append({
                'idx': global_idx,
                'prompt': prompt,
                'dir': f'prompt_{global_idx:04d}',
                'seed': seed,
                'seq_length': cache_data['seq_length'],
            })
            
        except Exception as e:
            print(f"[GPU {rank}] Error processing prompt {global_idx}: {prompt[:50]}...")
            print(f"[GPU {rank}] Error: {e}")
            continue
        
        # Clear cache periodically
        if (global_idx - start) % 10 == 0:
            torch.cuda.empty_cache()
    
    # Synchronize all processes
    dist.barrier()
    
    # Gather index data from all processes (only on rank 0)
    if rank == 0:
        # Save local index first
        local_index_path = os.path.join(config.output_dir, f'index_rank_{rank}.json')
        with open(local_index_path, 'w', encoding='utf-8') as f:
            json.dump(local_index_data, f, indent=2, ensure_ascii=False)
    else:
        # Other ranks also save their local index
        local_index_path = os.path.join(config.output_dir, f'index_rank_{rank}.json')
        with open(local_index_path, 'w', encoding='utf-8') as f:
            json.dump(local_index_data, f, indent=2, ensure_ascii=False)
    
    dist.barrier()
    
    # Merge index files on rank 0
    if rank == 0:
        print("\nMerging index files...")
        all_prompts_data = []
        
        for r in range(world_size):
            index_file = os.path.join(config.output_dir, f'index_rank_{r}.json')
            if os.path.exists(index_file):
                with open(index_file, 'r', encoding='utf-8') as f:
                    rank_data = json.load(f)
                    all_prompts_data.extend(rank_data)
                # Remove temporary file
                os.remove(index_file)
        
        # Sort by index
        all_prompts_data.sort(key=lambda x: x['idx'])
        
        # Create final index
        final_index = {
            'config': asdict(config),
            'num_prompts': len(all_prompts_data),
            'num_steps': config.num_steps,
            'feature_dim': 3072,
            'world_size': world_size,
            'prompts': all_prompts_data,
        }
        
        index_path = os.path.join(config.output_dir, 'index.json')
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(final_index, f, indent=2, ensure_ascii=False)
        
        print(f"\nData generation complete!")
        print(f"Cache data saved to: {config.output_dir}")
        print(f"First 200 images saved to: {config.image_dir_first200}")
        print(f"Last 200 images saved to: {config.image_dir_last200}")
        print(f"Index file: {index_path}")
        print(f"Total prompts processed: {len(all_prompts_data)}")
    
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Multi-GPU cache data generation")
    parser.add_argument('--prompt_file', type=str, default='prompts/prompts_train.txt')
    parser.add_argument('--output_dir', type=str, default='data/cache_data')
    parser.add_argument('--image_dir_first200', type=str, default='data/images_first200')
    parser.add_argument('--image_dir_last200', type=str, default='data/images_last200')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--width', type=int, default=1328)
    parser.add_argument('--height', type=int, default=1328)
    parser.add_argument('--num_steps', type=int, default=50)
    parser.add_argument('--true_cfg_scale', type=float, default=4.0)
    
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
