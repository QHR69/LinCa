import os
import torch
from PIL import Image, ExifTags
from tqdm import tqdm
from dataclasses import dataclass
from transformers import pipeline

from pipeline.pipeline_qwenimage import QwenImagePipeline
from pipeline.pipeline_qwenimage_edit import QwenImageEditPipeline
from cache_functions.cache_utils import pipeline_with_taylorseer
from cache_functions import cache_init

NSFW_THRESHOLD = 0.85

@dataclass
class SamplingOptions:
    image: Image.Image          # Input image
    prompts: list[str]          # List of prompts
    negative_prompt: str        # Negative prompt for guidance
    width: int                  # Image width
    height: int                 # Image height
    num_steps: int              # Number of sampling steps
    guidance_scale: float       # Guidance scale
    seed: int | None            # Random seed
    num_images_per_prompt: int  # Number of images generated per prompt
    batch_size: int             # Batch size (batching of prompts)
    model_name: str             # Model name
    output_dir: str             # Output directory
    add_sampling_metadata: bool # Whether to add metadata
    use_nsfw_filter: bool       # Whether to enable NSFW filter
    test_FLOPs: bool            # Whether in FLOPs test mode
    monitor_gpu_usage: bool    # Whether to monitor GPU memory usage
    interval: int               # Cache period length
    max_order: int              # Maximum order of Taylor expansion
    min_order: int              # Minimum order of Taylor expansion
    first_enhance: int          # Initial enhancement steps
    forecast_method: str        # Forecast method
    decompose_method: str       # Decomposition method
    use_z_cache: bool           # Use Z cache
    forecast_steps: int         # Forecast steps


def main(opts: SamplingOptions):
    if opts.use_z_cache and opts.forecast_steps < opts.interval:
        raise ValueError(f"forecast_steps must be greater than interval when use_z_cache is enabled")
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Optional NSFW classifier
    if opts.use_nsfw_filter:
        nsfw_classifier = pipeline(
            "image-classification",
            model="Falconsai/nsfw_image_detection",
            device=device
        )
    else:
        nsfw_classifier = None

    # Load pipeline
    if opts.model_name == 'qwen-image':
        pipe = QwenImagePipeline.from_pretrained(
            "Qwen/Qwen-Image", 
            torch_dtype=torch.bfloat16
        ).to(device=device)
    elif opts.model_name == 'qwen-image-edit':
        pipe = QwenImageEditPipeline.from_pretrained(
            "Qwen/Qwen-Image-Edit", 
            torch_dtype=torch.bfloat16
        ).to(device=device)
    else:
        raise ValueError(f"Model name {opts.model_name} not supported.")

    pipe = pipeline_with_taylorseer(pipe)

    if not os.path.exists(opts.output_dir):
        os.makedirs(opts.output_dir)
    
    # Set random seed
    if opts.seed is not None:
        base_seed = opts.seed
    else:
        base_seed = torch.randint(0, 2**32, (1,)).item()

    total_images = len(opts.prompts) * opts.num_images_per_prompt
    progress_bar = tqdm(total=total_images, desc="Generating images")

    num_prompt_batches = (len(opts.prompts) + opts.batch_size - 1) // opts.batch_size
    idx = 0

    for batch_idx in range(num_prompt_batches):
        prompt_start = batch_idx * opts.batch_size
        prompt_end = min(prompt_start + opts.batch_size, len(opts.prompts))
        batch_prompts = opts.prompts[prompt_start:prompt_end]
        num_prompts_in_batch = len(batch_prompts)

        # Generate corresponding number of images for each prompt
        for image_idx in range(opts.num_images_per_prompt):
            seed = base_seed + idx
            idx += num_prompts_in_batch

            generators = [
                torch.Generator(device).manual_seed(int(seed + i))
                for i in range(num_prompts_in_batch)
            ]

            # Initialize cache
            cache_dic, current = cache_init(kwargs={
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
                'forecast_steps': opts.forecast_steps
            })
            
            # Generate images
            if opts.model_name == 'qwen-image':
                result = pipe(
                    prompt=batch_prompts,
                    negative_prompt=opts.negative_prompt,
                    height=opts.height,
                    width=opts.width,
                    num_inference_steps=opts.num_steps,
                    guidance_scale=opts.guidance_scale,
                    generator=generators,
                    cache_dic=cache_dic,
                    current=current
                )
            elif opts.model_name == 'qwen-image-edit':
                result = pipe(
                    image=opts.image, # type: ignore
                    prompt=batch_prompts,
                    negative_prompt=opts.negative_prompt,
                    height=opts.height,
                    width=opts.width,
                    num_inference_steps=opts.num_steps,
                    guidance_scale=opts.guidance_scale,
                    generator=generators,
                    cache_dic=cache_dic,
                    current=current
                )
            else:
                raise ValueError(f"Model name {opts.model_name} not supported.")
            
            # Handle different return types from pipeline
            images = getattr(result, 'images', None)
            if images is None:
                if isinstance(result, (list, tuple)):
                    images = list(result)
                else:
                    images = [result]
            
            for i, img in enumerate(images):
                if not isinstance(img, Image.Image):
                    continue

                if opts.use_nsfw_filter and nsfw_classifier is not None:
                    nsfw_result = nsfw_classifier(img)
                    nsfw_score = next((res["score"] for res in nsfw_result if res["label"] == "nsfw"), 0.0)
                else:
                    nsfw_score = 0.0
                
                if nsfw_score < NSFW_THRESHOLD:
                    # Add EXIF metadata
                    exif_data = Image.Exif()
                    exif_data[ExifTags.Base.Software] = "AI generated;t2i;qwen" if opts.model_name == 'qwen-image' else "AI generated;ti2i;qwen"
                    exif_data[ExifTags.Base.Make] = "Qwen"
                    exif_data[ExifTags.Base.Model] = opts.model_name
                    if opts.add_sampling_metadata and i < len(batch_prompts):
                        exif_data[ExifTags.Base.ImageDescription] = batch_prompts[i]
                    
                    # Save image
                    filename = f"{opts.output_dir}/img_{idx - num_prompts_in_batch + i}.jpg"
                    img.save(filename, exif=exif_data, quality=95, subsampling=0)

                else:
                    print(f"Generated image may contain inappropriate content, skipped.")
                
                progress_bar.update(1)
    
    progress_bar.close()
    print(f"Generated {total_images} images in {opts.output_dir}")


def read_prompts(prompt_file: str):
    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Generate images using the flux model.")
    parser.add_argument('--input_image', type=str, default='img.jpg', help='Path to the input image.')
    parser.add_argument('--prompt_file', type=str, default='prompts/DrawBench200.txt', help='Path to the prompt text file.')
    parser.add_argument('--negative_prompt', type=str, default=" ", help='Negative prompt for guidance.')
    parser.add_argument('--width', type=int, default=1328, help='Width of the generated image.')
    parser.add_argument('--height', type=int, default=1328, help='Height of the generated image.')
    parser.add_argument('--num_steps', type=int, default=50, help='Number of sampling steps.')
    parser.add_argument('--guidance_scale', type=float, default=1.0, help='Guidance scale.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--num_images_per_prompt', type=int, default=1, help='Number of images per prompt.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size (prompt batching).')
    parser.add_argument('--model_name', type=str, default='qwen-image', choices=['qwen-image', 'qwen-image-edit'], help='Model name.')
    parser.add_argument('--output_dir', type=str, default='samples/test', help='Directory to save images.')
    parser.add_argument('--add_sampling_metadata', action='store_false', help='Whether to add prompt metadata to images.')
    parser.add_argument('--use_nsfw_filter', action='store_true', help='Enable NSFW filter.')
    parser.add_argument('--test_FLOPs', action='store_true', help='Test inference computation cost.')
    parser.add_argument('--monitor_gpu_usage', action='store_true', help='Monitor GPU memory usage during sampling.')
    
    parser.add_argument('--interval', type=int, default=10)
    parser.add_argument('--max_order', type=int, default=2)
    parser.add_argument('--min_order', type=int, default=0)
    parser.add_argument('--first_enhance', type=int, default=3)
    parser.add_argument('--forecast_method', type=str, default='hermite', choices=['hermite', 'taylor'])
    parser.add_argument('--decompose_method', type=str, default='FFT', choices=['None', 'FFT', 'DCT'])
    parser.add_argument('--use_z_cache', action='store_true')
    parser.add_argument('--forecast_steps', type=int, default=10)
    
    args = parser.parse_args()

    image = Image.open(args.input_image)
    prompts = read_prompts(args.prompt_file)

    opts = SamplingOptions(
        image=image,
        prompts=prompts,
        negative_prompt=args.negative_prompt,
        width=args.width,
        height=args.height,
        num_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        num_images_per_prompt=args.num_images_per_prompt,
        batch_size=args.batch_size,
        model_name=args.model_name,
        output_dir=args.output_dir,
        add_sampling_metadata=args.add_sampling_metadata,
        use_nsfw_filter=args.use_nsfw_filter,
        test_FLOPs=args.test_FLOPs,
        monitor_gpu_usage=args.monitor_gpu_usage,
        interval=args.interval,
        max_order=args.max_order,
        min_order=args.min_order,
        first_enhance=args.first_enhance,
        forecast_method=args.forecast_method,
        decompose_method=args.decompose_method,
        use_z_cache=args.use_z_cache,
        forecast_steps=args.forecast_steps
    )

    main(opts)
    # CUDA_VISIBLE_DEVICES=4 python sample.py --model_name qwen-image-edit --interval 9 --test_FLOPs
    