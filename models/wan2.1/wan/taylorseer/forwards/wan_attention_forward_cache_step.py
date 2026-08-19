import torch
import torch.cuda.amp as amp
from diffusers.configuration_utils import ConfigMixin, register_to_config
from wan.modules.model import WanAttentionBlock

from .wan_attention_cache_forward import wan_attention_cache_forward

def wan_attention_forward_cache_step(
    self:WanAttentionBlock,
    x,
    e,
    layer_cache_dict,
    distance,
):
    r"""
    Args:
        x(Tensor): Shape [B, L, C]
        e(Tensor): Shape [B, 6, C]
        seq_lens(Tensor): Shape [B], length of each sequence in batch
        grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
        freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
    """
    assert e.dtype == torch.float32
    with amp.autocast(dtype=torch.float32):
        e = (self.modulation + e).chunk(6, dim=1)
    assert e[0].dtype == torch.float32

    x = wan_attention_cache_forward(
        sa_dict=  layer_cache_dict['self-attention'],
        ca_dict=  layer_cache_dict['cross-attention'],
        ffn_dict= layer_cache_dict['ffn'],
        e=e,
        x=x,
        distance= distance
    )

    return x
