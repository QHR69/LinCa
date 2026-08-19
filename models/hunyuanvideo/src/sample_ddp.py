import os
import re
import time
from dataclasses import dataclass
from glob import iglob

import torch
import torch.distributed as dist
from einops import rearrange
from PIL import ExifTags, Image
from transformers.pipelines import pipeline
from tqdm import tqdm

from flux.sampling import get_noise, get_schedule, prepare, prepare_kontext, prepare_fill, unpack, denoise_cache
from flux.util import configs, embed_watermark, load_ae, load_clip, load_flow_model, load_t5

NSFW_THRESHOLD = 0.85  # NSFW score threshold


@dataclass
class SamplingOptions:
    prompts: list[str]          # List of prompts
    input_image: str            # Path to the input image
    mask_path: str              # Path to the mask image
    width: int                  # Image width
    height: int                 # Image height
    num_steps: int              # Number of sampling steps
    guidance: float             # Guidance value
    seed: int                   # Random seed
    num_images_per_prompt: int  # Number of images generated per prompt
    batch_size: int             # Batch size (batching of prompts)
    model_name: str             # Model name
    output_dir: str             # Output directory
    add_sampling_metadata: bool # Whether to add metadata
    use_nsfw_filter: bool       # Whether to enable NSFW filter
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


def main(opts: SamplingOptions):
    # Initialize distributed environment
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(rank)

    # Task allocation for distributed processing
    total_prompts = len(opts.prompts)
    per_proc = (total_prompts + world_size - 1) // world_size
    start = rank * per_proc
    end = min(start + per_proc, total_prompts)
    prompts = opts.prompts[start:end]

    if rank == 0 and not os.path.exists(opts.output_dir):
        os.makedirs(opts.output_dir, exist_ok=True)

    # Optional NSFW classifier
    if opts.use_nsfw_filter:
        nsfw_classifier = pipeline(
            "image-classification",
            model="Falconsai/nsfw_image_detection",
            device=device
        )
    else:
        nsfw_classifier = None

    # Load model
    model_name = opts.model_name
    if model_name not in configs:
        available = ", ".join(configs.keys())
        raise ValueError(f"Unknown model name: {model_name}, available options: {available}")

    if opts.num_steps is None:
        opts.num_steps = 4 if model_name == "flux-schnell" else 50

    # Ensure width and height are multiples of 16
    opts.width = 16 * (opts.width // 16)
    opts.height = 16 * (opts.height // 16)

    # Load T5 and CLIP models to GPU
    if rank == 0:
        print("Loading models...")
    t5 = load_t5(device, max_length=256 if model_name == "flux-schnell" else 512)
    clip = load_clip(device)

    # Load model to GPU
    model = load_flow_model(model_name, device=device)
    ae = load_ae(model_name, device=device)

    total_images = len(prompts) * opts.num_images_per_prompt
    progress_bar = tqdm(total=total_images, desc="Generating images") if rank == 0 else None

    # Compute number of prompt batches
    num_prompt_batches = (len(prompts) + opts.batch_size - 1) // opts.batch_size

    idx = 0  # Image index for this process

    for batch_idx in range(num_prompt_batches):
        prompt_start = batch_idx * opts.batch_size
        prompt_end = min(prompt_start + opts.batch_size, len(prompts))
        batch_prompts = prompts[prompt_start:prompt_end]
        num_prompts_in_batch = len(batch_prompts)

        # Generate corresponding number of images for each prompt
        for image_idx in range(opts.num_images_per_prompt):
            # Calculate global indices for consistent seeding across processes
            global_img_idx = (start + prompt_start) * opts.num_images_per_prompt + image_idx

            # Prepare random seed using global index
            seed = int(opts.seed + global_img_idx)
            # Prepare input
            batch_size = num_prompts_in_batch
            x = get_noise(
                batch_size,
                opts.height,
                opts.width,
                device=device,
                dtype=torch.bfloat16,
                seed=seed,
            )

            # Prepare prompts
            # batch_prompts is a list containing the prompts in the current batch
            if model_name == "flux-dev" or model_name == "flux-schnell":
                inp = prepare(t5, clip, x, batch_prompts)
            elif model_name == "flux-dev-kontext":
                inp, _, _ = prepare_kontext(t5, clip, batch_prompts, ae, seed=seed, device=device, img_cond_path=opts.input_image)
                inp.pop("img_cond_orig")
            elif model_name == "flux-dev-fill":
                inp = prepare_fill(t5, clip, x, batch_prompts, ae, img_cond_path=opts.input_image, mask_path=opts.mask_path)

            timesteps = get_schedule(opts.num_steps, inp["img"].shape[1], shift=(model_name != "flux-schnell")) # type: ignore
            
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
                x = unpack(x.float(), opts.height, opts.width)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    x = ae.decode(x)

            # Convert to PIL format and save
            x = x.clamp(-1, 1)
            x = embed_watermark(x.float())
            x = rearrange(x, "b c h w -> b h w c")

            for i in range(batch_size):
                img_array = x[i]
                img = Image.fromarray((127.5 * (img_array + 1.0)).cpu().byte().numpy())

                # Optional NSFW filtering
                nsfw_score = 0.0
                if opts.use_nsfw_filter and nsfw_classifier is not None:
                    try:
                        nsfw_result = nsfw_classifier(img)
                        nsfw_score = next((res["score"] for res in nsfw_result if res["label"] == "nsfw"), 0.0) # type: ignore
                    except Exception as e:
                        pass

                if nsfw_score < NSFW_THRESHOLD:
                    exif_data = Image.Exif()
                    exif_data[ExifTags.Base.Software] = "AI generated;t2i;flux" if model_name == "flux-schnell" or model_name == "flux-dev" else "AI generated;ti2i;flux"
                    exif_data[ExifTags.Base.Make] = "Black Forest Labs"
                    exif_data[ExifTags.Base.Model] = model_name
                    if opts.add_sampling_metadata:
                        exif_data[ExifTags.Base.ImageDescription] = batch_prompts[i]
                    
                    # Use global index for consistent file naming
                    global_prompt_idx = start + prompt_start + i
                    global_file_idx = global_prompt_idx * opts.num_images_per_prompt + image_idx
                    fn = os.path.join(opts.output_dir, f"img_{global_file_idx}.jpg")
                    img.save(fn, exif=exif_data, quality=95, subsampling=0)

                if rank == 0 and progress_bar is not None:
                    progress_bar.update(1)

            idx += num_prompts_in_batch

    if rank == 0 and progress_bar is not None:
        progress_bar.close()

    dist.barrier()
    
    if rank == 0:
        print("All images generated.")
    
    dist.destroy_process_group()


def read_prompts(prompt_file: str):
    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Generate images using the flux model.")
    parser.add_argument('--prompt_file', type=str, default='prompts/DrawBench200.txt', help='Path to the prompt text file.')
    parser.add_argument('--input_image', type=str, default='img.jpg', help='Path to the input image.')
    parser.add_argument('--mask_path', type=str, default='mask.jpg', help='Path to the mask image.') # TODO
    parser.add_argument('--width', type=int, default=1024, help='Width of the generated image.')
    parser.add_argument('--height', type=int, default=1024, help='Height of the generated image.')
    parser.add_argument('--num_steps', type=int, default=50, help='Number of sampling steps.')
    parser.add_argument('--guidance', type=float, default=3.5, help='Guidance value.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--num_images_per_prompt', type=int, default=1, help='Number of images per prompt.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size (prompt batching).')
    parser.add_argument('--model_name', type=str, default='flux-schnell', choices=['flux-schnell', 'flux-dev', 'flux-dev-kontext', 'flux-dev-fill'], help='Model name.')
    parser.add_argument('--output_dir', type=str, default='samples/test', help='Directory to save images.')
    parser.add_argument('--add_sampling_metadata', action='store_true', help='Whether to add prompt metadata to images.')
    parser.add_argument('--use_nsfw_filter', action='store_true', help='Enable NSFW filter.')
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

    prompts = read_prompts(args.prompt_file)

    opts = SamplingOptions(
        prompts=prompts,
        input_image=args.input_image,
        mask_path=args.mask_path,
        width=args.width,
        height=args.height,
        num_steps=args.num_steps,
        guidance=args.guidance,
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
        forecast_steps=args.forecast_steps,
    )

    main(opts)
    # CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 src/sample_ddp.py --model_name flux-dev-kontext --add_sampling_metadata
