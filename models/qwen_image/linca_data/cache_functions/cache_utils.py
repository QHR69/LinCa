import torch
import math
import torch_dct as dct
from typing import Dict, Tuple


@torch.compile
def decomposition_FFT(x: torch.Tensor, cutoff_ratio: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fast Fourier Transform frequency domain decomposition
    
    Args:
        x: Input tensor [B, H*W, D]
        cutoff_ratio: Cutoff frequency ratio (0~0.5)
        
    Returns:
        Tuple of (low_freq, high_freq) tensors with same dtype as input
    """
    orig_dtype = x.dtype
    device = x.device

    x_fp32 = x.to(torch.float32)  # Convert to fp32 for FFT compatibility

    B, HW, D = x_fp32.shape
    freq = torch.fft.fft(x_fp32, dim=1)  # FFT on spatial dimension

    freqs = torch.fft.fftfreq(HW, d=1.0, device=device)
    cutoff = cutoff_ratio * freqs.abs().max()

    # Create frequency masks
    low_mask = freqs.abs() <= cutoff
    high_mask = ~low_mask

    low_mask = low_mask[None, :, None]  # Broadcast to (B, HW, D)
    high_mask = high_mask[None, :, None]

    low_freq_complex  = freq * low_mask
    high_freq_complex = freq * high_mask

    # IFFT and take real part
    low_fp32  = torch.fft.ifft(low_freq_complex,  dim=1).real
    high_fp32 = torch.fft.ifft(high_freq_complex, dim=1).real

    low  = low_fp32.to(device=device, dtype=orig_dtype)
    high = high_fp32.to(device=device, dtype=orig_dtype)

    return low, high


@torch.compile
def decomposition_DCT(x: torch.Tensor, cutoff_ratio: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Discrete Cosine Transform frequency domain decomposition
    
    Args:
        x: Input tensor [B, H*W, D]
        cutoff_ratio: Cutoff frequency ratio (0~0.5)
        
    Returns:
        Tuple of (low_freq, high_freq) tensors
    """
    orig_dtype = x.dtype
    device = x.device
    B, HW, D = x.shape
    H = W = int(math.sqrt(HW))
    
    x_fp32 = x.to(torch.float32)  # Convert to fp32 for DCT compatibility
    
    x_spatial = x_fp32.transpose(1, 2).reshape(B, D, H, W)  # Reshape for 2D DCT
    
    dct_2d = dct.dct_2d(x_spatial, norm='ortho')  # 2D DCT-II
    
    # Create 2D frequency grid
    freq_h = torch.arange(H, device=device, dtype=torch.float32)
    freq_w = torch.arange(W, device=device, dtype=torch.float32)
    freq_grid_h, freq_grid_w = torch.meshgrid(freq_h, freq_w, indexing='ij')
    
    freq_magnitude = torch.sqrt(freq_grid_h**2 + freq_grid_w**2)  # Distance from origin
    
    max_freq = freq_magnitude.max()
    cutoff = cutoff_ratio * max_freq
    
    # Create frequency masks (low freq concentrates at top-left)
    low_mask = freq_magnitude <= cutoff
    high_mask = ~low_mask
    
    low_dct_2d = dct_2d * low_mask.unsqueeze(0).unsqueeze(0)
    high_dct_2d = dct_2d * high_mask.unsqueeze(0).unsqueeze(0)
    
    # 2D IDCT reconstruction
    low_spatial = dct.idct_2d(low_dct_2d, norm='ortho')
    high_spatial = dct.idct_2d(high_dct_2d, norm='ortho')
    
    low = low_spatial.reshape(B, D, HW).transpose(1, 2).to(dtype=orig_dtype)
    high = high_spatial.reshape(B, D, HW).transpose(1, 2).to(dtype=orig_dtype)
    
    return low, high


@torch.compile
def reconstruction(low_freq: torch.Tensor, high_freq: torch.Tensor) -> torch.Tensor:
    return low_freq + high_freq
    

@torch.compile
def hermite_polynomial(x: torch.Tensor, n: int) -> torch.Tensor:
    """
    Physicist's Hermite polynomial H_n(x) calculation
    
    Args:
        x: Input tensor
        n: Polynomial order
        
    Returns:
        Hermite polynomial result tensor
    """
    if n == 0:
        return torch.ones_like(x)
    elif n == 1:
        return 2 * x
    
    H_prev = torch.ones_like(x)
    H_curr = 2 * x
    
    # Recurrence: H_{n+1}(x) = 2x H_n(x) - 2n H_{n-1}(x)
    for k in range(2, n + 1):
        H_next = 2 * x * H_curr - 2 * (k - 1) * H_prev
        H_prev, H_curr = H_curr, H_next
    
    return H_curr


def hermite_formula(module_dict: Dict, distance: int, max_order: int) -> torch.Tensor:
    """
    Hermite polynomial based feature prediction
    
    Args:
        module_dict: Module dictionary containing derivatives
        distance: Prediction distance
        max_order: Maximum order to use
        
    Returns:
        Predicted feature tensor
    """
    F_latest = module_dict[0].clone()
    
    x_tensor = torch.tensor(float(distance), dtype=F_latest.dtype, device=F_latest.device)
    
    scale_factor = 0.71  # Control polynomial growth
    x_scaled = x_tensor * scale_factor
    
    pred = F_latest.clone()
    
    available_order = min(max_order, len(module_dict) - 1)
    
    # Hermite-based expansion for better convergence
    for k in range(1, available_order + 1):
        diff_k = module_dict[k]
        
        Hk = hermite_polynomial(x_scaled, k)
        
        alpha = Hk / math.factorial(k) * (scale_factor ** k)
        
        pred.add_(diff_k, alpha=float(alpha))
    
    return pred


def taylor_formula(module_dict: Dict, distance: int, max_order: int) -> torch.Tensor: 
    """
    Taylor expansion formula calculation
    
    Args:
        module_dict: Module dictionary containing derivatives
        distance: Prediction distance
        max_order: Maximum order of Taylor expansion
        
    Returns:
        Taylor expansion result tensor
    """
    output = module_dict[0].clone() * 0  # Initialize with zeros
    num_terms = min(len(module_dict), max_order + 1)
    
    # Taylor series: f(x+h) = f(x) + f'(x)h + f''(x)h²/2! + ...
    for i in range(num_terms):
        output += (1 / math.factorial(i)) * module_dict[i] * (distance ** i)
    
    return output


def cache_formula(cache_dic: Dict, module_dict: Dict, distance: int, max_order: int):
    if cache_dic['forecast_method'] == 'hermite':
        return hermite_formula(module_dict, distance, max_order)
    elif cache_dic['forecast_method'] == 'taylor':
        return taylor_formula(module_dict, distance, max_order)
    else:
        raise ValueError(f"Unsupported forecast method: {cache_dic['forecast_method']}")


def module_cache_init(cache_dic: Dict, current: Dict):
    """
    Initialize cache for frequency domain decomposition
    
    Args:
        cache_dic: Cache dictionary
        current: Current step information
        
    Returns:
        None (updates cache_dic in place)
    """
    if cache_dic['use_z_cache']:
        if (current['step'] == 0):
            if cache_dic['decompose_method'] == 'None':
                cache_dic['cache'][-1][current['module']] = {}
                cache_dic['last_cache'][-1][current['module']] = {}
            else:
                cache_dic['cache'][-1][f"{current['module']}_low_freq"] = {}
                cache_dic['cache'][-1][f"{current['module']}_high_freq"] = {}
                cache_dic['last_cache'][-1][f"{current['module']}_low_freq"] = {}
                cache_dic['last_cache'][-1][f"{current['module']}_high_freq"] = {}
    else:
        if (current['step'] == 0):
            if cache_dic['decompose_method'] == 'None':
                cache_dic['cache'][-1][current['module']] = {}
            else:
                cache_dic['cache'][-1][f"{current['module']}_low_freq"] = {}
                cache_dic['cache'][-1][f"{current['module']}_high_freq"] = {}


def derivative_approximation(cache_dic: Dict, current: Dict, feature: torch.Tensor):
    """
    Frequency domain based derivative approximation
    
    Args:
        cache_dic: Cache dictionary
        current: Current step information
        feature: Input feature tensor
        
    Returns:
        None (updates cache_dic in place)
    """
    if cache_dic['use_z_cache']: # pseudo full computation
        difference_distance = current['activated_steps'][-2] - current['activated_steps'][-1]
        input_dic = cache_dic['last_cache'] if current['type'] == 'cache' else cache_dic['cache']
        update_dic = cache_dic['cache']
        
        if current['type'] == 'full':
            if cache_dic['decompose_method'] == 'None':
                if "last_cache" in cache_dic:
                    cache_dic["last_cache"][-1][current["module"]] = cache_dic["cache"][-1][current["module"]]
            else:
                if "last_cache" in cache_dic:
                    cache_dic["last_cache"][-1][f"{current['module']}_low_freq"] = cache_dic['cache'][-1][f"{current['module']}_low_freq"]
                    cache_dic["last_cache"][-1][f"{current['module']}_high_freq"] = cache_dic['cache'][-1][f"{current['module']}_high_freq"]
    else:
        difference_distance = current['activated_steps'][-2] - current['activated_steps'][-1]# weird
        input_dic = cache_dic['cache']
        update_dic = cache_dic['cache']


    if cache_dic['decompose_method'] == 'None':
        updated_taylor_factors = {}
        updated_taylor_factors[0] = feature

        for i in range(cache_dic['max_order']):
            if (input_dic[-1][current['module']].get(i, None) is not None) and (current['step'] > cache_dic['first_enhance'] - 2):
                updated_taylor_factors[i + 1] = (updated_taylor_factors[i] - input_dic[-1][current['module']][i]) / difference_distance
            else:
                break
        
        update_dic[-1][current['module']] = updated_taylor_factors

    else:
        if cache_dic['decompose_method'] == 'FFT':
            low_freq, high_freq = decomposition_FFT(feature)
        elif cache_dic['decompose_method'] == 'DCT':
            low_freq, high_freq = decomposition_DCT(feature)

        for freq_type, freq_feature in [('low_freq', low_freq), ('high_freq', high_freq)]:
            module_key = f"{current['module']}_{freq_type}"
            
            updated_taylor_factors = {}
            updated_taylor_factors[0] = freq_feature

            for i in range(cache_dic['max_order']):
                if (input_dic[-1].get(module_key, {}).get(i, None) is not None) and (current['step'] > cache_dic['first_enhance'] - 2):
                    updated_taylor_factors[i + 1] = (updated_taylor_factors[i] - input_dic[-1][module_key][i]) / difference_distance
                else:
                    break
            
            update_dic[-1][module_key] = updated_taylor_factors


def cache_step(cache_dic: Dict, current: Dict) -> torch.Tensor:
    """
    Frequency domain based cache step prediction
    
    Args:
        cache_dic: Cache dictionary
        current: Current step information
        distance: Prediction distance
        
    Returns:
        Reconstructed feature tensor
    """
    if cache_dic['decompose_method'] == 'None':
        distance = current['activated_steps'][-1] - current['step']
        return cache_formula(cache_dic, cache_dic['cache'][-1][current['module']], distance, cache_dic['max_order'])
    else:
        low_freq_dict = cache_dic['cache'][-1][f"{current['module']}_low_freq"]
        high_freq_dict = cache_dic['cache'][-1][f"{current['module']}_high_freq"]

        distance = current['activated_steps'][-1] - current['step']
        
        # Use 0-order Hermite for low frequency (stable)
        low_freq = cache_formula(cache_dic, low_freq_dict, distance, cache_dic['min_order'])

        # Use 2-order Hermite for high frequency (adaptive)
        high_freq = cache_formula(cache_dic, high_freq_dict, distance, cache_dic['max_order'])
        
        return reconstruction(low_freq, high_freq)


def cache_step_merge(cache_dic: Dict, current: Dict) -> torch.Tensor:
    """
    Merge values from two caches.
    :param cache_dic: Cache dictionary.
    :param current: Current step information.
    """
    if cache_dic['decompose_method'] == 'None':
        module_cache = cache_dic['last_cache'][-1][current['module']]
        future_module_cache = cache_dic['cache'][-1][current['module']]

        distance = current['activated_steps'][-2] - current['step']
        future_distance = current['activated_steps'][-1] - current['step']
        if current['update']:
            current['activated_steps'][-1] = current['step'] 

        values = cache_formula(cache_dic, module_cache, distance, cache_dic['max_order'])
        future_values = cache_formula(cache_dic, future_module_cache, future_distance, cache_dic['max_order'])
        merged_values = values * (1 - current['weight']) + future_values * current['weight']
        return merged_values
    else:
        low_freq_dict = cache_dic['last_cache'][-1][f"{current['module']}_low_freq"]
        high_freq_dict = cache_dic['last_cache'][-1][f"{current['module']}_high_freq"]
        future_low_freq_dict = cache_dic['cache'][-1][f"{current['module']}_low_freq"]
        future_high_freq_dict = cache_dic['cache'][-1][f"{current['module']}_high_freq"]
        
        distance = current['activated_steps'][-2] - current['step']
        future_distance = current['activated_steps'][-1] - current['step']
        if current['update']:
            current['activated_steps'][-1] = current['step']

        # Use 0-order Hermite for low frequency (stable)
        low_freq = cache_formula(cache_dic, low_freq_dict, distance, cache_dic['min_order'])
        high_freq = cache_formula(cache_dic, high_freq_dict, distance, cache_dic['max_order'])
        # Use 2-order Hermite for high frequency (adaptive)
        future_low_freq = cache_formula(cache_dic, future_low_freq_dict, future_distance, cache_dic['min_order'])
        future_high_freq = cache_formula(cache_dic, future_high_freq_dict, future_distance, cache_dic['max_order'])
        
        low_merged_values = low_freq * (1 - current['weight']) + future_low_freq * current['weight']
        high_merged_values = high_freq * (1 - current['weight']) + future_high_freq * current['weight']

        return reconstruction(low_merged_values, high_merged_values)


def pipeline_with_taylorseer(pipe):
    """
    Pipeline with Taylorseer.
    
    :param pipe: FluxKontextPipeline
    """
    import types
    from pipeline.transformer_qwenimage import QwenImageTransformer2DModel as LocalQwenImageTransformer2DModel

    pipe.transformer.forward = types.MethodType(LocalQwenImageTransformer2DModel.forward, pipe.transformer)

    return pipe
