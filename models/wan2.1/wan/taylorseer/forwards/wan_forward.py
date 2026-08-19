import math

import torch
import torch.cuda.amp as amp
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from wan.modules import WanModel
from wan.modules.model import sinusoidal_embedding_1d
from wan.taylorseer.cache_functions import cal_type as taylor_cal_type
from wan.taylorseer.taylorseer_utils import taylor_cache_init, derivative_approximation, taylor_formula
from .wan_cache_forward import wan_cache_forward

def wan_forward(
    self:WanModel,
    x,
    t,
    context,
    seq_len,
    current_step,
    current_stream,
    clip_fea=None,
    y=None,
    ):
    r"""
    Forward pass through the diffusion model

    Args:
        x (List[Tensor]):
            List of input video tensors, each with shape [C_in, F, H, W]
        t (Tensor):
            Diffusion timesteps tensor of shape [B]
        context (List[Tensor]):
            List of text embeddings each with shape [L, C]
        seq_len (`int`):
            Maximum sequence length for positional encoding
        clip_fea (Tensor, *optional*):
            CLIP image features for image-to-video mode
        y (List[Tensor], *optional*):
            Conditional video inputs for image-to-video mode, same shape as x

    Returns:
        List[Tensor]:
            List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
    """

    self.current['step'] = current_step
    self.current['stream'] = current_stream
    learned_cache = self.cache_dic.get('decompose_method') == 'learned'
    if current_stream == 'cond_stream':
        if learned_cache:
            from flux.modules.cache_functions.cal_type import cal_type as learned_cal_type
            learned_cal_type(self.cache_dic, self.current)
        else:
            taylor_cal_type(self.cache_dic, self.current)

    if self.model_type == 'i2v':
        assert clip_fea is not None and y is not None
    # params
    device = self.patch_embedding.weight.device
    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)

    if y is not None:
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

    # embeddings
    x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
    grid_sizes = torch.stack(
        [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
    x = [u.flatten(2).transpose(1, 2) for u in x]
    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
    assert seq_lens.max() <= seq_len
    x = torch.cat([
        torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                  dim=1) for u in x
    ])

    # time embeddings
    with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t).float())
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))
        assert e.dtype == torch.float32 and e0.dtype == torch.float32

    # context
    context_lens = None
    context = self.text_embedding(
        torch.stack([
            torch.cat(
                [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
            for u in context
        ]))

    if clip_fea is not None:
        context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
        context = torch.concat([context_clip, context], dim=1)

    # arguments
    kwargs = dict(
        e=e0,
        seq_lens=seq_lens,
        grid_sizes=grid_sizes,
        freqs=self.freqs,
        context=context,
        context_lens=context_lens,
        cache_dic=self.cache_dic,
        current=self.current)
    
    if learned_cache:
        from flux.modules.cache_functions.cache_utils_learned import (
            module_cache_init,
            derivative_approximation as learned_derivative_approximation,
            cache_step,
            cache_step_merge,
        )
        self.current['module'] = 'cond' if current_stream == 'cond_stream' else 'uncond'
        if self.current['type'] == 'full':
            for i, block in enumerate(self.blocks):
                self.current['layer'] = i
                x = block(x, **kwargs)
            module_cache_init(cache_dic=self.cache_dic, current=self.current)
            learned_derivative_approximation(cache_dic=self.cache_dic, current=self.current, feature=x)
        else:
            if not self.cache_dic['use_z_cache'] or not self.current["merge"]:
                x = cache_step(cache_dic=self.cache_dic, current=self.current)
            else:
                x = cache_step_merge(cache_dic=self.cache_dic, current=self.current)
                if self.current["update"]:
                    learned_derivative_approximation(cache_dic=self.cache_dic, current=self.current, feature=x)

    elif self.cache_dic.get('lite_cache', False) and self.current['type'] == 'Taylor':
        self.current['layer'] = 0
        self.current['module'] = 'output'
        x = taylor_formula(
            derivative_dict=self.cache_dic['cache'][-1][self.current['stream']][0]['output'],
            distance=self.current['step'] - self.current['activated_steps'][-1])
    else:
        for i, block in enumerate(self.blocks):
            self.current['layer'] = i
            x = block(x, **kwargs)

        if getattr(self, "collect_features", False):
            if not hasattr(self, "collected_features"):
                self.collected_features = {}
            self.collected_features[self.current['stream']] = x.detach()

        if self.cache_dic.get('lite_cache', False):
            self.current['layer'] = 0
            self.current['module'] = 'output'
            taylor_cache_init(cache_dic=self.cache_dic, current=self.current)
            derivative_approximation(cache_dic=self.cache_dic, current=self.current, feature=x)

    # head
    x = self.head(x, e)

    # unpatchify
    x = self.unpatchify(x, grid_sizes)
    return [u.float() for u in x]
