"""
Pipeline for qwen_edit cache data generation.
Extends QwenImageEditPipeline with generate_and_save_cache for saving cond/uncond features.
Edit: latent_model_input = [latents, image_latents], cache saves only latent part [:, :latents.size(1)].
"""

import sys
import importlib.util
from pathlib import Path
import numpy as np
import torch
from typing import Any, Dict, List, Optional, Union

# Load freqca_qwen pipeline without package conflict (local pipeline/ shadows it)
_linca_root = Path(__file__).resolve().parent.parent.parent.parent.parent
_freqca_edit_path = _linca_root / "freqca_qwen" / "pipeline" / "pipeline_qwenimage_edit.py"
_spec = importlib.util.spec_from_file_location("qwenimage_edit_pipeline", _freqca_edit_path)
_mod = importlib.util.module_from_spec(_spec)
sys.path.insert(0, str(_linca_root / "freqca_qwen"))
_spec.loader.exec_module(_mod)
QwenImageEditPipeline = _mod.QwenImageEditPipeline
calculate_shift = _mod.calculate_shift
retrieve_latents = _mod.retrieve_latents
retrieve_timesteps = _mod.retrieve_timesteps

try:
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
except ImportError:
    XLA_AVAILABLE = False


class QwenImageEditPipelineForData(QwenImageEditPipeline):
    """
    QwenImageEditPipeline for data generation.
    Adds generate_and_save_cache to save cond/uncond features (latent part only).
    """

    @torch.no_grad()
    def generate_and_save_cache(
        self,
        image,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = " ",
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 1.0,
        true_cfg_scale: float = 4.0,
        generator: Optional[torch.Generator] = None,
        max_sequence_length: int = 512,
    ) -> Dict[str, Any]:
        """
        Generate edit result and save intermediate features for each step.
        Image must be pre-resized to 1024x1024 by caller.
        Cache saves only latent part: cond_hidden[:, :latents.size(1)].

        Returns:
            dict: {
                'cond': List[Tensor],    # each [4096, 3072]
                'uncond': List[Tensor],  # each [4096, 3072]
                'image': PIL.Image,
                'seq_length': int,       # 4096 for 1024x1024
            }
        """
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError("prompt must be provided")

        device = self._execution_device
        has_neg_prompt = negative_prompt is not None
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt

        # Resize image to 1024x1024 if PIL (caller may have done this)
        if hasattr(image, "size") and (image.size[0] != width or image.size[1] != height):
            image = self.image_processor.resize(image, height, width)
        prompt_image = image

        # Encode prompt with image (edit uses VL processor, needs PIL)
        prompt_embeds, prompt_embeds_mask = self.encode_prompt(
            prompt=prompt,
            image=prompt_image,
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=max_sequence_length,
        )
        prompt_embeds = prompt_embeds[:, :max_sequence_length]
        prompt_embeds_mask = prompt_embeds_mask[:, :max_sequence_length]

        if do_true_cfg:
            negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
                prompt=negative_prompt,
                image=prompt_image,
                device=device,
                num_images_per_prompt=1,
                max_sequence_length=max_sequence_length,
            )
            negative_prompt_embeds = negative_prompt_embeds[:, :max_sequence_length]
            negative_prompt_embeds_mask = negative_prompt_embeds_mask[:, :max_sequence_length]

        # Preprocess image for VAE/latents
        image_tensor = self.image_processor.preprocess(prompt_image, height, width)
        image_tensor = image_tensor.unsqueeze(2).to(device=device, dtype=torch.float32)

        # Prepare latents
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, image_latents = self.prepare_latents(
            image_tensor,
            batch_size,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
        )
        latent_model_input = latents
        if image_latents is not None:
            latent_model_input = torch.cat([latents, image_latents], dim=1)

        # img_shapes for 1024x1024: [(1, 64, 64), (1, 64, 64)]
        latent_h, latent_w = height // self.vae_scale_factor // 2, width // self.vae_scale_factor // 2
        img_shapes = [[(1, latent_h, latent_w), (1, latent_h, latent_w)]] * batch_size

        # Timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, _ = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )

        if self.transformer.config.guidance_embeds and guidance_scale is not None:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist() if prompt_embeds_mask is not None else None
        negative_txt_seq_lens = (
            negative_prompt_embeds_mask.sum(dim=1).tolist()
            if do_true_cfg and negative_prompt_embeds_mask is not None
            else None
        )

        cache_data = {"cond": [], "uncond": []}
        latent_seq_len = latents.size(1)

        for i, t in enumerate(timesteps):
            timestep = t.expand(latents.shape[0]).to(latents.dtype)

            # Conditional branch
            noise_pred, cond_hidden = self.transformer(
                hidden_states=latent_model_input,
                timestep=timestep / 1000,
                guidance=guidance,
                encoder_hidden_states_mask=prompt_embeds_mask,
                encoder_hidden_states=prompt_embeds,
                img_shapes=img_shapes,
                txt_seq_lens=txt_seq_lens,
                return_dict=False,
                return_hidden_for_cache=True,
            )
            noise_pred = noise_pred[:, :latent_seq_len]
            # Save only latent part for cache
            cond_cache = cond_hidden[:, :latent_seq_len].squeeze(0).cpu()
            assert cond_cache.shape == (latent_seq_len, 3072), f"cond shape {cond_cache.shape}"
            cache_data["cond"].append(cond_cache)

            if do_true_cfg:
                neg_noise_pred, uncond_hidden = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    encoder_hidden_states_mask=negative_prompt_embeds_mask,
                    encoder_hidden_states=negative_prompt_embeds,
                    img_shapes=img_shapes,
                    txt_seq_lens=negative_txt_seq_lens,
                    return_dict=False,
                    return_hidden_for_cache=True,
                )
                neg_noise_pred = neg_noise_pred[:, :latent_seq_len]
                uncond_cache = uncond_hidden[:, :latent_seq_len].squeeze(0).cpu()
                assert uncond_cache.shape == (latent_seq_len, 3072), f"uncond shape {uncond_cache.shape}"
                cache_data["uncond"].append(uncond_cache)

                comb_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)
                cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
                noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
                noise_pred = comb_pred * (cond_norm / noise_norm)

            latents_dtype = latents.dtype
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            if latents.dtype != latents_dtype and hasattr(torch.backends.mps, "is_available") and torch.backends.mps.is_available():
                latents = latents.to(latents_dtype)

            latent_model_input = latents
            if image_latents is not None:
                latent_model_input = torch.cat([latents, image_latents], dim=1)

            if XLA_AVAILABLE:
                xm.mark_step()

        # Decode to image
        latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
        latents = latents.to(self.vae.dtype)
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
            1, self.vae.config.z_dim, 1, 1, 1
        ).to(latents.device, latents.dtype)
        latents = latents / latents_std + latents_mean
        image_out = self.vae.decode(latents, return_dict=False)[0][:, :, 0]
        image_out = self.image_processor.postprocess(image_out, output_type="pil")

        cache_data["image"] = image_out[0] if len(image_out) == 1 else image_out
        cache_data["seq_length"] = latent_seq_len

        return cache_data
