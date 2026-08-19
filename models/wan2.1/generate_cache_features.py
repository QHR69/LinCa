import argparse
import gc
import json
import logging
import math
import os
import random
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.cuda.amp as amp
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

if not hasattr(np, "_core") and hasattr(np, "core"):
    np._core = np.core
if hasattr(np, "_core") and not hasattr(np._core, "multiarray"):
    np._core.multiarray = np.core.multiarray

import wan
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES
from wan.utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                                  get_sampling_sigmas, retrieve_timesteps)
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


def normalize_prompt_item(item):
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        return item.get("prompt_en") or item.get("prompt") or item.get("caption") or ""
    return str(item)


def load_vbench_prompts(json_path, index_start=0, index_end=-1):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if index_end < 0 or index_end >= len(data):
        index_end = len(data) - 1
    prompts = [normalize_prompt_item(item) for item in data]
    return prompts[index_start:index_end + 1], index_start


def load_text_prompts(prompt_file, start_idx=0, end_idx=None):
    with open(prompt_file, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts[start_idx:end_idx], start_idx


def save_stream_feature(model, prompt_dir, step_idx, stream):
    feature = model.collected_features[stream]
    if feature.dim() == 3 and feature.shape[0] == 1:
        feature = feature.squeeze(0)
    suffix = "cond" if stream == "cond_stream" else "uncond"
    torch.save(feature.cpu(), prompt_dir / f"step_{step_idx:02d}_{suffix}.pt")


def reset_collect_cache(model, num_steps):
    model.cache_init()
    model.collect_features = True
    model.collected_features = {}

    cache_dic = model.cache_dic
    cache_dic["taylor_cache"] = True
    cache_dic["lite_cache"] = True
    cache_dic["fresh_threshold"] = 1
    cache_dic["first_enhance"] = num_steps
    cache_dic["max_order"] = 0
    cache_dic["cache_counter"] = 0

    model.current["activated_steps"] = [0]
    model.current["step"] = 0
    model.current["num_steps"] = num_steps


def collect_features_for_prompt(
    wan_t2v,
    prompt,
    prompt_idx,
    output_dir,
    size,
    frame_num,
    shift,
    sample_solver,
    sampling_steps,
    guide_scale,
    seed,
    offload_model,
):
    model = wan_t2v.model
    reset_collect_cache(model, sampling_steps)

    prompt_dir = output_dir / f"prompt_{prompt_idx:04d}"
    prompt_dir.mkdir(parents=True, exist_ok=True)

    target_shape = (
        wan_t2v.vae.model.z_dim,
        (frame_num - 1) // wan_t2v.vae_stride[0] + 1,
        size[1] // wan_t2v.vae_stride[1],
        size[0] // wan_t2v.vae_stride[2],
    )
    seq_len = math.ceil(
        (target_shape[2] * target_shape[3])
        / (wan_t2v.patch_size[1] * wan_t2v.patch_size[2])
        * target_shape[1]
        / wan_t2v.sp_size
    ) * wan_t2v.sp_size

    n_prompt = wan_t2v.sample_neg_prompt
    seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
    seed_g = torch.Generator(device=wan_t2v.device)
    seed_g.manual_seed(seed)

    if not wan_t2v.t5_cpu:
        wan_t2v.text_encoder.model.to(wan_t2v.device)
        context = wan_t2v.text_encoder([prompt], wan_t2v.device)
        context_null = wan_t2v.text_encoder([n_prompt], wan_t2v.device)
        if offload_model:
            wan_t2v.text_encoder.model.cpu()
    else:
        context = wan_t2v.text_encoder([prompt], torch.device("cpu"))
        context_null = wan_t2v.text_encoder([n_prompt], torch.device("cpu"))
        context = [t.to(wan_t2v.device) for t in context]
        context_null = [t.to(wan_t2v.device) for t in context_null]

    noise = [
        torch.randn(
            target_shape[0],
            target_shape[1],
            target_shape[2],
            target_shape[3],
            dtype=torch.float32,
            device=wan_t2v.device,
            generator=seed_g,
        )
    ]

    @contextmanager
    def noop_no_sync():
        yield

    no_sync = getattr(model, "no_sync", noop_no_sync)

    with amp.autocast(dtype=wan_t2v.param_dtype), torch.no_grad(), no_sync():
        if sample_solver == "unipc":
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=wan_t2v.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False,
            )
            sample_scheduler.set_timesteps(sampling_steps, device=wan_t2v.device, shift=shift)
            timesteps = sample_scheduler.timesteps
        elif sample_solver == "dpm++":
            sample_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=wan_t2v.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False,
            )
            sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
            timesteps, _ = retrieve_timesteps(
                sample_scheduler, device=wan_t2v.device, sigmas=sampling_sigmas)
        else:
            raise NotImplementedError("Unsupported solver.")

        latents = noise
        arg_c = {"context": context, "seq_len": seq_len}
        arg_null = {"context": context_null, "seq_len": seq_len}
        model.to(wan_t2v.device)

        for i, t in enumerate(tqdm(timesteps, desc=f"prompt_{prompt_idx:04d}", leave=False)):
            latent_model_input = latents
            timestep = torch.stack([t])

            noise_pred_cond = model(
                latent_model_input,
                t=timestep,
                current_step=i,
                current_stream="cond_stream",
                **arg_c,
            )[0]
            save_stream_feature(model, prompt_dir, i, "cond_stream")

            noise_pred_uncond = model(
                latent_model_input,
                t=timestep,
                current_step=i,
                current_stream="uncond_stream",
                **arg_null,
            )[0]
            save_stream_feature(model, prompt_dir, i, "uncond_stream")

            noise_pred = noise_pred_uncond + guide_scale * (noise_pred_cond - noise_pred_uncond)
            temp_x0 = sample_scheduler.step(
                noise_pred.unsqueeze(0),
                t,
                latents[0].unsqueeze(0),
                return_dict=False,
                generator=seed_g,
            )[0]
            latents = [temp_x0.squeeze(0)]

    del noise, latents, sample_scheduler
    if offload_model:
        model.cpu()
        gc.collect()
        torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(description="Collect Wan transformer-output features for learned cache training.")
    parser.add_argument("--vbench-json-path", type=str, default=None)
    parser.add_argument("--prompt_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--index-start", type=int, default=0)
    parser.add_argument("--index-end", type=int, default=-1)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--task", type=str, default="t2v-1.3B", choices=list(WAN_CONFIGS.keys()))
    parser.add_argument("--size", type=str, default="832*480", choices=list(SIZE_CONFIGS.keys()))
    parser.add_argument("--frame_num", type=int, default=81)
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--sample_steps", type=int, default=50)
    parser.add_argument("--sample_shift", type=float, default=8.0)
    parser.add_argument("--sample_solver", type=str, default="unipc", choices=["unipc", "dpm++"])
    parser.add_argument("--sample_guide_scale", type=float, default=6.0)
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--offload_model", action="store_true")
    parser.add_argument("--t5_cpu", action="store_true")
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
    args = parse_args()

    assert args.task in SUPPORTED_SIZES, f"Unsupported task: {args.task}"
    assert args.size in SUPPORTED_SIZES[args.task], (
        f"Unsupported size {args.size} for {args.task}: {SUPPORTED_SIZES[args.task]}")

    if args.vbench_json_path:
        prompts, start_offset = load_vbench_prompts(args.vbench_json_path, args.index_start, args.index_end)
    elif args.prompt_file:
        prompts, start_offset = load_text_prompts(args.prompt_file, args.start_idx, args.end_idx)
    else:
        raise ValueError("Must provide --vbench-json-path or --prompt_file")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = WAN_CONFIGS[args.task]
    logging.info("Loading Wan model from %s", args.ckpt_dir)
    wan_t2v = wan.WanT2V(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=args.t5_cpu,
    )
    logging.info("Model loaded. Collecting %d prompts to %s", len(prompts), output_dir)

    skipped = 0
    for local_idx, prompt in enumerate(tqdm(prompts, desc="prompts")):
        global_idx = start_offset + local_idx
        prompt_dir = output_dir / f"prompt_{global_idx:04d}"
        last_cond = prompt_dir / f"step_{args.sample_steps - 1:02d}_cond.pt"
        last_uncond = prompt_dir / f"step_{args.sample_steps - 1:02d}_uncond.pt"
        if last_cond.exists() and last_uncond.exists():
            skipped += 1
            continue

        logging.info("[%04d] %s", global_idx, prompt[:120])
        collect_features_for_prompt(
            wan_t2v=wan_t2v,
            prompt=prompt,
            prompt_idx=global_idx,
            output_dir=output_dir,
            size=SIZE_CONFIGS[args.size],
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model,
        )

    if skipped:
        logging.info("Skipped %d completed prompts", skipped)
    logging.info("Done.")


if __name__ == "__main__":
    main()
