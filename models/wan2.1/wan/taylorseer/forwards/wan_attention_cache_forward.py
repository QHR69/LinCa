import torch
import torch.cuda.amp as amp
from typing import Dict
from wan.taylorseer.taylorseer_utils import taylor_cache_init, derivative_approximation, taylor_formula

@torch.compile
def wan_attention_cache_forward(sa_dict:Dict, ca_dict:Dict, ffn_dict:Dict, e:tuple, x:torch.Tensor, distance:int):

    seer_sa  = taylor_formula(derivative_dict=sa_dict,  distance=distance)
    seer_ca  = taylor_formula(derivative_dict=ca_dict,  distance=distance)
    seer_ffn = taylor_formula(derivative_dict=ffn_dict, distance=distance)

    x = cache_add(x, seer_sa, seer_ca, seer_ffn, e)
    
    return x

def cache_add(x, sa, ca, ffn, e):
    with amp.autocast(dtype=torch.float32):
        x = x + sa * e[2]
    x = x + ca
    with amp.autocast(dtype=torch.float32):
        x = x + ffn * e[5]
    return x