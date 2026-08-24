"""
Cache utilities for learned invertible decomposition network

Keeps the same interface as FreqCa's cache_utils.py,
but decomposes and predicts with a learned invertible network.

Key differences:
- decomposition: FFT/DCT becomes a learned invertible network
- prediction: Hermite/Taylor becomes fixed per-partition orders (0 / 1 / 2)
- reconstruction: a plain sum becomes the inverse of the invertible transform
"""

import torch
import math
from typing import Dict, Tuple, Optional, List
import os
import sys

# Import invertible_net
from ..invertible_net import LearnedDecompositionPredictor, FixedPredictionStrategy


# Module-level storage for the loaded predictor
_GLOBAL_PREDICTOR: Optional[LearnedDecompositionPredictor] = None


def load_predictor(
    checkpoint_path: str,
    device: str = 'cuda',
) -> LearnedDecompositionPredictor:
    """
    Load a trained predictor.

    Args:
        checkpoint_path: path to the checkpoint
        device: device

    Returns:
        The loaded predictor.
    """
    global _GLOBAL_PREDICTOR

    if _GLOBAL_PREDICTOR is None:
        print(f"Loading learned predictor from {checkpoint_path}")
        _GLOBAL_PREDICTOR = LearnedDecompositionPredictor.from_pretrained(
            checkpoint_path, device=device
        )
        _GLOBAL_PREDICTOR.eval()

    return _GLOBAL_PREDICTOR


def get_predictor() -> Optional[LearnedDecompositionPredictor]:
    """Return the globally registered predictor."""
    return _GLOBAL_PREDICTOR


def set_predictor(predictor: LearnedDecompositionPredictor):
    """Register the predictor globally."""
    global _GLOBAL_PREDICTOR
    _GLOBAL_PREDICTOR = predictor


def module_cache_init_learned(cache_dic: Dict, current: Dict):
    """
    Initialize cache for learned decomposition

    Same interface as the baseline module_cache_init.
    """
    if cache_dic['use_z_cache']:
        if current['step'] == 0:
            # Store both the raw feature and its decomposition
            cache_dic['cache'][-1][current['module']] = {
                'features': {},      # step -> feature tensor
                'decomposed': {},    # step -> (z0, z1, z2) tuple
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
    Store feature and its decomposition for later prediction

    Same interface as the baseline derivative_approximation.
    Rather than derivatives, this stores the raw feature and its decomposition.
    """
    predictor = get_predictor()

    if cache_dic['use_z_cache']:
        if current['type'] == 'full':
            # Update last_cache
            if "last_cache" in cache_dic:
                cache_dic["last_cache"][-1][current["module"]] = cache_dic["cache"][-1][current["module"]].copy()
                cache_dic["last_cache"][-1][current["module"]]['features'] = cache_dic["cache"][-1][current["module"]]['features'].copy()
                cache_dic["last_cache"][-1][current["module"]]['decomposed'] = cache_dic["cache"][-1][current["module"]]['decomposed'].copy()

        input_dic = cache_dic['last_cache'] if current['type'] == 'cache' else cache_dic['cache']
        update_dic = cache_dic['cache']
    else:
        input_dic = cache_dic['cache']
        update_dic = cache_dic['cache']

    # Store this step's feature
    step = current['step']
    update_dic[-1][current['module']]['features'][step] = feature.clone()

    # With a predictor loaded, also store the decomposition
    if predictor is not None:
        with torch.no_grad():
            # Remember the incoming dtype, compute in float32, then cast back
            original_dtype = feature.dtype
            feature_float = feature.float() if original_dtype != torch.float32 else feature
            z0, z1, z2 = predictor.decompose(feature_float)
            # Cast back to the original dtype
            z0 = z0.to(original_dtype)
            z1 = z1.to(original_dtype)
            z2 = z2.to(original_dtype)
            update_dic[-1][current['module']]['decomposed'][step] = (z0.clone(), z1.clone(), z2.clone())


def cache_step_learned(cache_dic: Dict, current: Dict) -> torch.Tensor:
    """
    Predict current step feature using learned decomposition

    Reads the cached (z0, z1, z2) straight from the decomposition cache
    and predicts with predict_from_decomposed, so decompose never runs twice.
    """
    predictor = get_predictor()
    if predictor is None:
        raise RuntimeError("Predictor not loaded! Call load_predictor() first.")

    module_cache = cache_dic['cache'][-1][current['module']]

    # Collect the active steps
    activated_steps = current['activated_steps']
    current_step = current['step']

    # Build the decomposed cache list as [(distance, z0, z1, z2), ...]
    decomposed_cache_list = []
    for act_step in reversed(activated_steps):  # nearest first
        if act_step < current_step and act_step in module_cache['decomposed']:
            distance = current_step - act_step
            z0, z1, z2 = module_cache['decomposed'][act_step]
            decomposed_cache_list.append((distance, z0, z1, z2))

    # Sort by distance, nearest first
    decomposed_cache_list.sort(key=lambda x: x[0])

    # At most three points
    decomposed_cache_list = decomposed_cache_list[:3]

    if len(decomposed_cache_list) == 0:
        raise RuntimeError(f"No cache available for step {current_step}")

    # The number of cached points caps the prediction order
    if len(decomposed_cache_list) >= 3:
        order = 2
    elif len(decomposed_cache_list) >= 2:
        order = 1
    else:
        order = 0

    # Predict straight from the cached decomposition, avoiding a second forward
    with torch.no_grad():
        # Remember the original dtype (decomposed cache is bfloat16, compute in float32)
        original_dtype = decomposed_cache_list[0][1].dtype

        # Compute in float32
        decomposed_cache_float = [
            (d, z0.float(), z1.float(), z2.float())
            for d, z0, z1, z2 in decomposed_cache_list
        ]

        predicted = predictor.predict_from_decomposed(decomposed_cache_float, order=order)

        # Cast back to the original dtype
        predicted = predicted.to(original_dtype)

    return predicted


def cache_step_merge_learned(cache_dic: Dict, current: Dict) -> torch.Tensor:
    """
    Merge predictions from two cache periods (for z_cache mode)

    Reads the cached decomposition
    and predicts with predict_from_decomposed, avoiding a second forward.
    """
    predictor = get_predictor()
    if predictor is None:
        raise RuntimeError("Predictor not loaded! Call load_predictor() first.")

    # Predict from last_cache
    last_module_cache = cache_dic['last_cache'][-1][current['module']]
    # Predict from the cache
    curr_module_cache = cache_dic['cache'][-1][current['module']]

    activated_steps = current['activated_steps']
    current_step = current['step']

    # Build the decomposed cache list for last_cache
    last_decomposed_list = []
    for act_step in reversed(activated_steps[:-1]):  # excluding the most recent active step
        if act_step < current_step and act_step in last_module_cache['decomposed']:
            distance = current_step - act_step
            z0, z1, z2 = last_module_cache['decomposed'][act_step]
            last_decomposed_list.append((distance, z0, z1, z2))
    last_decomposed_list.sort(key=lambda x: x[0])
    last_decomposed_list = last_decomposed_list[:3]

    # Build the decomposed cache list for curr_cache
    curr_decomposed_list = []
    for act_step in reversed(activated_steps):
        if act_step < current_step and act_step in curr_module_cache['decomposed']:
            distance = current_step - act_step
            z0, z1, z2 = curr_module_cache['decomposed'][act_step]
            curr_decomposed_list.append((distance, z0, z1, z2))
    curr_decomposed_list.sort(key=lambda x: x[0])
    curr_decomposed_list = curr_decomposed_list[:3]

    with torch.no_grad():
        # Remember the original dtype (decomposed cache is bfloat16, compute in float32)
        original_dtype = curr_decomposed_list[0][1].dtype if len(curr_decomposed_list) > 0 else torch.float32

        # Compute in float32
        last_decomposed_float = [
            (d, z0.float(), z1.float(), z2.float())
            for d, z0, z1, z2 in last_decomposed_list
        ]
        curr_decomposed_float = [
            (d, z0.float(), z1.float(), z2.float())
            for d, z0, z1, z2 in curr_decomposed_list
        ]

        # Predict from last_cache
        if len(last_decomposed_float) >= 3:
            last_order = 2
        elif len(last_decomposed_float) >= 2:
            last_order = 1
        else:
            last_order = 0

        if len(last_decomposed_float) > 0:
            last_pred = predictor.predict_from_decomposed(last_decomposed_float, order=last_order)
        else:
            last_pred = None

        # Predict from curr_cache
        if len(curr_decomposed_float) >= 3:
            curr_order = 2
        elif len(curr_decomposed_float) >= 2:
            curr_order = 1
        else:
            curr_order = 0

        curr_pred = predictor.predict_from_decomposed(curr_decomposed_float, order=curr_order)

        # Without last_pred there is nothing to blend, so return curr_pred in the original dtype
        if last_pred is None:
            return curr_pred.to(original_dtype)

        # Update activated_steps
        if current['update']:
            current['activated_steps'][-1] = current['step']

        # Blend the two predictions
        weight = current['weight']
        merged = last_pred * (1 - weight) + curr_pred * weight

        # Cast back to the original dtype
        merged = merged.to(original_dtype)

    return merged


# ============ Compatibility wrappers ============

def module_cache_init(cache_dic: Dict, current: Dict):
    """Compatibility wrapper: picks the init path from decompose_method."""
    if cache_dic.get('decompose_method') == 'learned':
        module_cache_init_learned(cache_dic, current)
    else:
        # Fall through to the baseline
        from .cache_utils import module_cache_init as original_init
        original_init(cache_dic, current)


def derivative_approximation(cache_dic: Dict, current: Dict, feature: torch.Tensor):
    """Compatibility wrapper: picks the path from decompose_method."""
    if cache_dic.get('decompose_method') == 'learned':
        derivative_approximation_learned(cache_dic, current, feature)
    else:
        from .cache_utils import derivative_approximation as original_func
        original_func(cache_dic, current, feature)


def cache_step(cache_dic: Dict, current: Dict) -> torch.Tensor:
    """Compatibility wrapper: picks the prediction path from decompose_method."""
    if cache_dic.get('decompose_method') == 'learned':
        return cache_step_learned(cache_dic, current)
    else:
        from .cache_utils import cache_step as original_func
        return original_func(cache_dic, current)


def cache_step_merge(cache_dic: Dict, current: Dict) -> torch.Tensor:
    """Compatibility wrapper: picks the merge path from decompose_method."""
    if cache_dic.get('decompose_method') == 'learned':
        return cache_step_merge_learned(cache_dic, current)
    else:
        from .cache_utils import cache_step_merge as original_func
        return original_func(cache_dic, current)
