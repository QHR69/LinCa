import gc
import logging
import math
import os
import random
import sys
import types
from contextlib import contextmanager
from functools import partial

import torch
import torch.cuda.amp as amp
import torch.distributed as dist
from tqdm import tqdm

from wan.distributed.fsdp import shard_model
from wan.modules.model import WanModel
from wan.modules.t5 import T5EncoderModel
from wan.modules.vae import WanVAE
from wan.utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                               get_sampling_sigmas, retrieve_timesteps)
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

from wan import WanT2V
from wan.taylorseer.cache_functions import cache_init, cal_type

def wan_t2v_generate(self:WanT2V,
             input_prompt,
             size=(1280, 720),
             frame_num=81,
             shift=5.0,
             sample_solver='unipc',
             sampling_steps=50,
             guide_scale=5.0,
             n_prompt="",
             seed=-1,
             offload_model=True):
    r"""
    Generates video frames from text prompt using diffusion process.
    Args:
        input_prompt (`str`):
            Text prompt for content generation
        size (tupele[`int`], *optional*, defaults to (1280,720)):
            Controls video resolution, (width,height).
        frame_num (`int`, *optional*, defaults to 81):
            How many frames to sample from a video. The number should be 4n+1
        shift (`float`, *optional*, defaults to 5.0):
            Noise schedule shift parameter. Affects temporal dynamics
        sample_solver (`str`, *optional*, defaults to 'unipc'):
            Solver used to sample the video.
        sampling_steps (`int`, *optional*, defaults to 40):
            Number of diffusion sampling steps. Higher values improve quality but slow generation
        guide_scale (`float`, *optional*, defaults 5.0):
            Classifier-free guidance scale. Controls prompt adherence vs. creativity
        n_prompt (`str`, *optional*, defaults to ""):
            Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
        seed (`int`, *optional*, defaults to -1):
            Random seed for noise generation. If -1, use random seed.
        offload_model (`bool`, *optional*, defaults to True):
            If True, offloads models to CPU during generation to save VRAM
    Returns:
        torch.Tensor:
            Generated video frames tensor. Dimensions: (C, N H, W) where:
            - C: Color channels (3 for RGB)
            - N: Number of frames (81)
            - H: Frame height (from size)
            - W: Frame width from size)
    """
    learned_cache_config = getattr(self.model, "learned_cache_config", None)
    if learned_cache_config is not None:
        from flux.modules.cache_functions.cache_init import cache_init as learned_cache_init
        cache_kwargs = dict(
            num_steps=sampling_steps,
            test_FLOPs=False,
            monitor_gpu_usage=False,
            interval=learned_cache_config.get("interval", 3),
            max_order=learned_cache_config.get("max_order", 2),
            min_order=learned_cache_config.get("min_order", 0),
            first_enhance=learned_cache_config.get("first_enhance", 3),
            forecast_method=learned_cache_config.get("forecast_method", "hermite"),
            decompose_method="learned",
            use_z_cache=learned_cache_config.get("use_z_cache", False),
            forecast_steps=learned_cache_config.get("forecast_steps", learned_cache_config.get("interval", 3)),
        )
        self.model.cache_dic, self.model.current = learned_cache_init(**cache_kwargs)

    # preprocess
    F = frame_num
    
    target_shape = (self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
                    size[1] // self.vae_stride[1],
                    size[0] // self.vae_stride[2])
    
    seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                        (self.patch_size[1] * self.patch_size[2]) *
                        target_shape[1] / self.sp_size) * self.sp_size
    if n_prompt == "":
        n_prompt = self.sample_neg_prompt
    seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
    seed_g = torch.Generator(device=self.device)
    seed_g.manual_seed(seed)
    
    if not self.t5_cpu:
        self.text_encoder.model.to(self.device)
        context = self.text_encoder([input_prompt], self.device)
        context_null = self.text_encoder([n_prompt], self.device)
        if offload_model:
            self.text_encoder.model.cpu()
    else:
        context = self.text_encoder([input_prompt], torch.device('cpu'))
        context_null = self.text_encoder([n_prompt], torch.device('cpu'))
        context = [t.to(self.device) for t in context]
        context_null = [t.to(self.device) for t in context_null]
    
    noise = [
        torch.randn(
            target_shape[0],
            target_shape[1],
            target_shape[2],
            target_shape[3],
            dtype=torch.float32,
            device=self.device,
            generator=seed_g)
    ]
    
    @contextmanager
    
    def noop_no_sync():
        yield
    no_sync = getattr(self.model, 'no_sync', noop_no_sync)
    
    # evaluation mode
    with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():
        
        if sample_solver == 'unipc':
            
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False)
            
            sample_scheduler.set_timesteps(
                sampling_steps, device=self.device, shift=shift)
            
            timesteps = sample_scheduler.timesteps
        
        elif sample_solver == 'dpm++':
            
            sample_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False)
            
            sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
            
            timesteps, _ = retrieve_timesteps(
                sample_scheduler,
                device=self.device,
                sigmas=sampling_sigmas)
        else:
            raise NotImplementedError("Unsupported solver.")
        
        # sample videos
        latents = noise

        arg_c = {'context': context, 'seq_len': seq_len}
        arg_null = {'context': context_null, 'seq_len': seq_len}

        self.model.to(self.device)

        for i, t in enumerate(tqdm(timesteps)):
            torch.compiler.cudagraph_mark_step_begin()
            current_step = i

            latent_model_input = latents
            timestep = [t]
            timestep = torch.stack(timestep)
        
            current_stream = 'cond_stream'

            noise_pred_cond = self.model(
                latent_model_input, t=timestep,
                current_step = current_step,
                current_stream = current_stream,
                **arg_c)[0]
            
            current_stream =  'uncond_stream'

            noise_pred_uncond = self.model(
                latent_model_input, t=timestep,
                current_step = current_step,
                current_stream = current_stream,
                **arg_null)[0]
            
            noise_pred = noise_pred_uncond + guide_scale * (
                noise_pred_cond - noise_pred_uncond)
            
            temp_x0 = sample_scheduler.step(
                noise_pred.unsqueeze(0),
                t,
                latents[0].unsqueeze(0),
                return_dict=False,
                generator=seed_g)[0]
            latents = [temp_x0.squeeze(0)]

        x0 = latents

        if offload_model:
            self.model.cpu()
            torch.cuda.empty_cache()
        
        if self.rank == 0:
            videos = self.vae.decode(x0)
    
    del noise, latents
    del sample_scheduler
    
    if offload_model:
        gc.collect()
        torch.cuda.synchronize()
    
    if dist.is_initialized():
        dist.barrier()
    
    return videos[0] if self.rank == 0 else None
