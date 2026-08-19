import torch
import torch.cuda.amp as amp
from wan.modules import WanModel

@torch.compile
def wan_cache_forward(self:WanModel,
                      e:torch.Tensor,
                      cond_cache_dict:dict,
                      distance:int,
                      x:torch.Tensor) -> torch.Tensor:

    for i, block in enumerate(self.blocks):
        x = block.cache_step_forward(x,
                                     e=e,
                                     layer_cache_dict=cond_cache_dict[i], 
                                     distance=distance)
    
    return x