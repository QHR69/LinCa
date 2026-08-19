"""
Cache utilities for the learned invertible decomposition network (two-stage)

Two-stage design:
- Stage 1: steps 0-24, uses predictor_stage1
- Stage 2: steps 25-49, uses predictor_stage2
- Steps 0-24 store both stage1 and stage2 decompositions (later stages may reuse them)
- Steps 25-49 store only the stage2 decomposition
"""

import torch
import math
from typing import Dict, Tuple, Optional, List
from ..invertible_net import LearnedDecompositionPredictor, FixedPredictionStrategy

# Stage boundaries
STAGE1_MAX_STEP = 24
STAGE2_MIN_STEP = 25

# Globals: the two predictors
_GLOBAL_PREDICTOR_STAGE1: Optional[LearnedDecompositionPredictor] = None
_GLOBAL_PREDICTOR_STAGE2: Optional[LearnedDecompositionPredictor] = None


def _torch_load(path: str, device: str):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _build_predictor_from_state(config: Dict, state_dict: Dict, device: str) -> LearnedDecompositionPredictor:
    model = LearnedDecompositionPredictor(
        dim=config.get('dim', 1536),
        num_blocks=config.get('num_blocks', 6),
        hidden_dim=config.get('hidden_dim', 512),
        split_dims=config.get('split_dims', [1024, 256, 256]),
        dropout=config.get('dropout', 0.1),
    )

    # 与 from_pretrained 对齐：组合 checkpoint 里可能包含预计算的 weight_inv buffer
    inv_weights = {k: v for k, v in state_dict.items() if k.endswith('.weight_inv')}
    main_weights = {k: v for k, v in state_dict.items() if not k.endswith('.weight_inv')}

    model.load_state_dict(main_weights, strict=True)

    for i, block in enumerate(model.net.blocks):
        key = f'net.blocks.{i}.conv1x1.weight_inv'
        if key in inv_weights:
            block.conv1x1.weight_inv = inv_weights[key]

    model = model.to(device)
    model.eval()
    return model


def _load_predictor_flexible(
    checkpoint_path: str,
    stage_key: Optional[str],
    device: str,
) -> LearnedDecompositionPredictor:
    ckpt = _torch_load(checkpoint_path, device)
    if isinstance(ckpt, dict) and 'config' in ckpt and stage_key is not None and stage_key in ckpt:
        return _build_predictor_from_state(ckpt['config'], ckpt[stage_key], device)
    return LearnedDecompositionPredictor.from_pretrained(checkpoint_path, device=device)


def load_predictor(
    checkpoint_path_stage1: str,
    checkpoint_path_stage2: str,
    device: str = 'cuda',
) -> Tuple[LearnedDecompositionPredictor, LearnedDecompositionPredictor]:
    """Load the two-stage predictors"""
    global _GLOBAL_PREDICTOR_STAGE1, _GLOBAL_PREDICTOR_STAGE2
    
    if _GLOBAL_PREDICTOR_STAGE1 is None:
        print(f"Loading two-stage predictors: {checkpoint_path_stage1}, {checkpoint_path_stage2}")
        same_file = checkpoint_path_stage1 == checkpoint_path_stage2
        _GLOBAL_PREDICTOR_STAGE1 = _load_predictor_flexible(
            checkpoint_path_stage1,
            'predictor_stage1' if same_file else None,
            device=device,
        )
        _GLOBAL_PREDICTOR_STAGE2 = _load_predictor_flexible(
            checkpoint_path_stage2,
            'predictor_stage2' if same_file else None,
            device=device,
        )
        _GLOBAL_PREDICTOR_STAGE1.eval()
        _GLOBAL_PREDICTOR_STAGE2.eval()
    
    return _GLOBAL_PREDICTOR_STAGE1, _GLOBAL_PREDICTOR_STAGE2


def set_predictor_two_stage(
    predictor_stage1: LearnedDecompositionPredictor,
    predictor_stage2: LearnedDecompositionPredictor,
):
    """Install the two-stage predictors"""
    global _GLOBAL_PREDICTOR_STAGE1, _GLOBAL_PREDICTOR_STAGE2
    _GLOBAL_PREDICTOR_STAGE1 = predictor_stage1
    _GLOBAL_PREDICTOR_STAGE2 = predictor_stage2


def get_predictor(step: Optional[int] = None) -> Optional[LearnedDecompositionPredictor]:
    """Return the predictor for this step. step=None returns stage1 (used as a loaded-or-not check)"""
    if step is not None:
        return _GLOBAL_PREDICTOR_STAGE2 if step >= STAGE2_MIN_STEP else _GLOBAL_PREDICTOR_STAGE1
    return _GLOBAL_PREDICTOR_STAGE1


def get_predictor_key(step: int) -> str:
    """Return the storage key for a decomposed cache entry"""
    return 'stage1' if step <= STAGE1_MAX_STEP else 'stage2'


def module_cache_init_learned(cache_dic: Dict, current: Dict):
    """Initialise the cache (two-stage: decomposed entries are dicts)"""
    if cache_dic['use_z_cache']:
        if current['step'] == 0:
            cache_dic['cache'][-1][current['module']] = {
                'features': {},
                'decomposed': {},
            }
            cache_dic['last_cache'][-1][current['module']] = {
                'features': {},
                'decomposed': {},
            }
    else:
        if current['step'] == 0:
            cache_dic['cache'][-1][current['module']] = {
                'features': {},
                'decomposed': {},
            }


def derivative_approximation_learned(cache_dic: Dict, current: Dict, feature: torch.Tensor):
    """
    Store the feature and its decompositions (two-stage)
    - Steps 0-24: store {'stage1': (z0,z1,z2), 'stage2': (z0,z1,z2)}
    - Steps 25-49: store {'stage2': (z0,z1,z2)}
    """
    step = current['step']
    
    if cache_dic['use_z_cache']:
        if current['type'] == 'full':
            if "last_cache" in cache_dic:
                cache_dic["last_cache"][-1][current["module"]] = cache_dic["cache"][-1][current["module"]].copy()
                cache_dic["last_cache"][-1][current["module"]]['features'] = cache_dic["cache"][-1][current["module"]]['features'].copy()
                cache_dic["last_cache"][-1][current["module"]]['decomposed'] = cache_dic["cache"][-1][current["module"]]['decomposed'].copy()
        
        input_dic = cache_dic['last_cache'] if current['type'] == 'cache' else cache_dic['cache']
        update_dic = cache_dic['cache']
    else:
        input_dic = cache_dic['cache']
        update_dic = cache_dic['cache']
    
    update_dic[-1][current['module']]['features'][step] = feature.clone()
    
    pred_s1 = _GLOBAL_PREDICTOR_STAGE1
    pred_s2 = _GLOBAL_PREDICTOR_STAGE2
    
    if pred_s1 is not None and pred_s2 is not None:
        with torch.no_grad():
            original_dtype = feature.dtype
            feature_float = feature.float() if original_dtype != torch.float32 else feature
            
            if step <= STAGE1_MAX_STEP:
                z0_1, z1_1, z2_1 = pred_s1.decompose(feature_float)
                z0_2, z1_2, z2_2 = pred_s2.decompose(feature_float)
                update_dic[-1][current['module']]['decomposed'][step] = {
                    'stage1': (z0_1.to(original_dtype).clone(), z1_1.to(original_dtype).clone(), z2_1.to(original_dtype).clone()),
                    'stage2': (z0_2.to(original_dtype).clone(), z1_2.to(original_dtype).clone(), z2_2.to(original_dtype).clone()),
                }
            else:
                z0, z1, z2 = pred_s2.decompose(feature_float)
                update_dic[-1][current['module']]['decomposed'][step] = {
                    'stage2': (z0.to(original_dtype).clone(), z1.to(original_dtype).clone(), z2.to(original_dtype).clone()),
                }


def cache_step_learned(cache_dic: Dict, current: Dict) -> torch.Tensor:
    """Two-stage predict: pick the predictor and decomposed key from current_step"""
    current_step = current['step']
    predictor = get_predictor(current_step)
    if predictor is None:
        raise RuntimeError("Predictor not loaded! Call load_predictor() or set_predictor_two_stage() first.")
    
    key = get_predictor_key(current_step)
    module_cache = cache_dic['cache'][-1][current['module']]
    activated_steps = current['activated_steps']
    
    decomposed_cache_list = []
    for act_step in reversed(activated_steps):
        if act_step < current_step and act_step in module_cache['decomposed']:
            dec = module_cache['decomposed'][act_step]
            if key in dec:
                distance = current_step - act_step
                z0, z1, z2 = dec[key]
                decomposed_cache_list.append((distance, z0, z1, z2))
    
    decomposed_cache_list.sort(key=lambda x: x[0])
    decomposed_cache_list = decomposed_cache_list[:3]
    
    if len(decomposed_cache_list) == 0:
        raise RuntimeError(f"No cache available for step {current_step}")
    
    order = 2 if len(decomposed_cache_list) >= 3 else (1 if len(decomposed_cache_list) >= 2 else 0)
    with torch.no_grad():
        original_dtype = decomposed_cache_list[0][1].dtype
        decomposed_cache_float = [
            (d, z0.float(), z1.float(), z2.float())
            for d, z0, z1, z2 in decomposed_cache_list
        ]
        predicted = predictor.predict_from_decomposed(decomposed_cache_float, order=order)
        predicted = predicted.to(original_dtype)
    
    return predicted


def cache_step_merge_learned(cache_dic: Dict, current: Dict) -> torch.Tensor:
    """Two-stage combined prediction"""
    current_step = current['step']
    predictor = get_predictor(current_step)
    if predictor is None:
        raise RuntimeError("Predictor not loaded!")
    
    key = get_predictor_key(current_step)
    last_module_cache = cache_dic['last_cache'][-1][current['module']]
    curr_module_cache = cache_dic['cache'][-1][current['module']]
    activated_steps = current['activated_steps']
    
    def _build_list(module_cache, steps, exclude_last=False):
        step_list = steps[:-1] if exclude_last else steps
        lst = []
        for act_step in reversed(step_list):
            if act_step < current_step and act_step in module_cache['decomposed']:
                dec = module_cache['decomposed'][act_step]
                if key in dec:
                    distance = current_step - act_step
                    z0, z1, z2 = dec[key]
                    lst.append((distance, z0, z1, z2))
        lst.sort(key=lambda x: x[0])
        return lst[:3]
    
    last_decomposed_list = _build_list(last_module_cache, activated_steps, exclude_last=True)
    curr_decomposed_list = _build_list(curr_module_cache, activated_steps, exclude_last=False)
    
    with torch.no_grad():
        original_dtype = curr_decomposed_list[0][1].dtype if len(curr_decomposed_list) > 0 else torch.float32
        
        last_decomposed_float = [(d, z0.float(), z1.float(), z2.float()) for d, z0, z1, z2 in last_decomposed_list]
        curr_decomposed_float = [(d, z0.float(), z1.float(), z2.float()) for d, z0, z1, z2 in curr_decomposed_list]
        
        last_order = min(2, len(last_decomposed_float) - 1) if len(last_decomposed_float) > 0 else 0
        curr_order = min(2, len(curr_decomposed_float) - 1) if len(curr_decomposed_float) > 0 else 0
        last_pred = predictor.predict_from_decomposed(last_decomposed_float, order=last_order).to(original_dtype) if len(last_decomposed_float) > 0 else None
        curr_pred = predictor.predict_from_decomposed(curr_decomposed_float, order=curr_order).to(original_dtype)
        
        if last_pred is None:
            return curr_pred
        
        if current['update']:
            current['activated_steps'][-1] = current['step']
        
        weight = current['weight']
        merged = last_pred * (1 - weight) + curr_pred * weight
    
    return merged


# ============ Compatibility wrappers ============

def module_cache_init(cache_dic: Dict, current: Dict):
    if cache_dic.get('decompose_method') == 'learned':
        module_cache_init_learned(cache_dic, current)
    else:
        from .cache_utils import module_cache_init as original_init
        original_init(cache_dic, current)


def derivative_approximation(cache_dic: Dict, current: Dict, feature: torch.Tensor):
    if cache_dic.get('decompose_method') == 'learned':
        derivative_approximation_learned(cache_dic, current, feature)
    else:
        from .cache_utils import derivative_approximation as original_func
        original_func(cache_dic, current, feature)


def cache_step(cache_dic: Dict, current: Dict) -> torch.Tensor:
    if cache_dic.get('decompose_method') == 'learned':
        return cache_step_learned(cache_dic, current)
    else:
        from .cache_utils import cache_step as original_func
        return original_func(cache_dic, current)


def cache_step_merge(cache_dic: Dict, current: Dict) -> torch.Tensor:
    if cache_dic.get('decompose_method') == 'learned':
        return cache_step_merge_learned(cache_dic, current)
    else:
        from .cache_utils import cache_step_merge as original_func
        return original_func(cache_dic, current)




def pipeline_with_learned_cache(pipe, checkpoint_path: str = None, checkpoint_path_stage2: str = None):
    """Configure learned caching on a pipeline (two-stage: pass two paths)"""
    import types
    from pipeline.transformer_qwenimage import QwenImageTransformer2DModel as LocalQwenImageTransformer2DModel
    
    pipe.transformer.forward = types.MethodType(LocalQwenImageTransformer2DModel.forward, pipe.transformer)
    
    if checkpoint_path is not None and checkpoint_path_stage2 is not None:
        device = next(pipe.transformer.parameters()).device
        load_predictor(checkpoint_path, checkpoint_path_stage2, device=str(device))
    
    return pipe
