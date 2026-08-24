"""
Cache utilities for the learned invertible decomposition network (multi-stage)

Multi-stage design:
- The user sets per-stage lengths via stage_splits, e.g. "12,12,13,13" for 4 stages
- Stage 0: step 0-11, Stage 1: step 12-23, Stage 2: step 24-36, Stage 3: step 37-49
- each stage has its own predictor
- for a step in stage i, store the decomposition for stage i and any later stage that may need it
  (a later stage can use an earlier step's cached value as a reference point)
"""

import torch
import math
from typing import Dict, Tuple, Optional, List
import os
import sys

# Import invertible_net
from ..invertible_net import LearnedDecompositionPredictor, FixedPredictionStrategy
# Import the non-learned forecast formulae (hermite / taylor)
from .cache_utils import cache_formula as _cache_formula_base

# Globals: list of per-stage predictors
_GLOBAL_PREDICTORS: List[Optional[LearnedDecompositionPredictor]] = []
# Stage boundaries: list of inclusive (start_step, end_step)
_STAGE_BOUNDARIES: List[Tuple[int, int]] = []


def parse_stage_splits(stage_splits_str: str, num_steps: int = 50) -> List[Tuple[int, int]]:
    """
    Parse the stage-split string into a list of (start_step, end_step)

    Args:
        stage_splits_str: comma-separated per-stage lengths, e.g. "12,12,13,13"
        num_steps: total number of steps

    Returns:
        List of (start_step, end_step) tuples, end_step is inclusive
    """
    splits = [int(x) for x in stage_splits_str.split(',')]
    total = sum(splits)
    if total != num_steps:
        raise ValueError(f"Stage splits sum ({total}) != num_steps ({num_steps}). "
                         f"splits={splits}")

    boundaries = []
    start = 0
    for size in splits:
        end = start + size - 1  # inclusive
        boundaries.append((start, end))
        start = end + 1

    return boundaries


def get_stage_index(step: int, boundaries: List[Tuple[int, int]] = None) -> int:
    """Return the stage index that owns this step"""
    if boundaries is None:
        boundaries = _STAGE_BOUNDARIES
    for i, (start, end) in enumerate(boundaries):
        if start <= step <= end:
            return i
    raise ValueError(f"Step {step} not in any stage boundary: {boundaries}")


def get_stage_key(step: int, boundaries: List[Tuple[int, int]] = None) -> str:
    """Return the storage key for a decomposed cache entry"""
    idx = get_stage_index(step, boundaries)
    return f'stage{idx}'


def load_predictor(
    checkpoint_paths: List[str],
    stage_splits_str: str,
    num_steps: int = 50,
    device: str = 'cuda',
) -> List[LearnedDecompositionPredictor]:
    """
    Load the multi-stage predictors

    Args:
        checkpoint_paths: checkpoint path for each stage
        stage_splits_str: stage-split string, e.g. "12,12,13,13"
        num_steps: total number of steps
        device: device
    """
    global _GLOBAL_PREDICTORS, _STAGE_BOUNDARIES

    _STAGE_BOUNDARIES = parse_stage_splits(stage_splits_str, num_steps)
    num_stages = len(_STAGE_BOUNDARIES)

    if len(checkpoint_paths) != num_stages:
        raise ValueError(f"Number of checkpoints ({len(checkpoint_paths)}) != "
                         f"number of stages ({num_stages})")

    if len(_GLOBAL_PREDICTORS) == 0 or _GLOBAL_PREDICTORS[0] is None:
        print(f"Loading {num_stages}-stage predictors:")
        _GLOBAL_PREDICTORS = []
        for i, path in enumerate(checkpoint_paths):
            start, end = _STAGE_BOUNDARIES[i]
            print(f"  Stage {i} (step {start}-{end}): {path}")
            predictor = LearnedDecompositionPredictor.from_pretrained(path, device=device)
            predictor.eval()
            _GLOBAL_PREDICTORS.append(predictor)

    return _GLOBAL_PREDICTORS


def set_predictors(
    predictors: List[LearnedDecompositionPredictor],
    stage_splits_str: str,
    num_steps: int = 50,
):
    """Install the multi-stage predictors (used during training)"""
    global _GLOBAL_PREDICTORS, _STAGE_BOUNDARIES
    _STAGE_BOUNDARIES = parse_stage_splits(stage_splits_str, num_steps)
    _GLOBAL_PREDICTORS = predictors


def get_predictor(step: Optional[int] = None) -> Optional[LearnedDecompositionPredictor]:
    """Return the predictor for a step; step=None returns the first, which is enough to test whether any are loaded."""
    if len(_GLOBAL_PREDICTORS) == 0:
        return None
    if step is not None:
        idx = get_stage_index(step)
        return _GLOBAL_PREDICTORS[idx]
    return _GLOBAL_PREDICTORS[0]


def get_num_stages() -> int:
    return len(_STAGE_BOUNDARIES)


def get_stage_boundaries() -> List[Tuple[int, int]]:
    return _STAGE_BOUNDARIES


def _build_module_dict(values: list, distances: list, max_order: int) -> dict:
    """Build the derivative-approximation dict from the cached partition values, using the same sign convention as the non-learned path.

    distances[i] = current_step - act_step_i, positive, with distances[0] the smallest
    To match cache_formula's sign convention:
    - the prediction distance is passed as -distances[0], i.e. negative
    - difference_distance = distances[i] - distances[i+1], also negative
    """
    module_dict = {0: values[0]}
    n = len(values)

    if n >= 2 and max_order >= 1:
        step_diff_01 = distances[0] - distances[1]
        if abs(step_diff_01) > 1e-6:
            module_dict[1] = (values[0] - values[1]) / step_diff_01

    if n >= 3 and max_order >= 2 and 1 in module_dict:
        step_diff_01 = distances[0] - distances[1]
        step_diff_12 = distances[1] - distances[2]
        if abs(step_diff_01) > 1e-6 and abs(step_diff_12) > 1e-6:
            prev_1st = (values[1] - values[2]) / step_diff_12
            module_dict[2] = (module_dict[1] - prev_1st) / step_diff_01

    return module_dict


def _predict_with_formula(
    cache_dic: Dict,
    predictor: LearnedDecompositionPredictor,
    decomposed_cache_list: list,
    max_order: int,
    original_dtype,
) -> torch.Tensor:
    """Apply the hermite or taylor formula to each partition independently, then recompose.

    decomposed_cache_list: [(d, z0, z1, z2), ...] sorted by ascending d, float32, d positive.
    """
    distances = [d for d, _, _, _ in decomposed_cache_list]
    z0_vals = [z0 for _, z0, _, _ in decomposed_cache_list]
    z1_vals = [z1 for _, _, z1, _ in decomposed_cache_list]
    z2_vals = [z2 for _, _, _, z2 in decomposed_cache_list]

    pred_dist = -distances[0]  # negative, matching the sign the non-learned path uses

    # z0: zeroth order, reused directly
    z0_dict = _build_module_dict(z0_vals, distances, 0)
    z0_pred = _cache_formula_base(cache_dic, z0_dict, pred_dist, 0)

    # z1: first order at most
    n = len(decomposed_cache_list)
    z1_order = min(1, max_order, n - 1)
    z1_dict = _build_module_dict(z1_vals, distances, z1_order)
    z1_pred = _cache_formula_base(cache_dic, z1_dict, pred_dist, z1_order)

    # z2: up to max_order
    z2_order = min(max_order, n - 1)
    z2_dict = _build_module_dict(z2_vals, distances, z2_order)
    z2_pred = _cache_formula_base(cache_dic, z2_dict, pred_dist, z2_order)

    z_pred = torch.cat([z0_pred, z1_pred, z2_pred], dim=-1)
    predicted = predictor.inverse(z_pred).to(original_dtype)
    return predicted


def module_cache_init_learned(cache_dic: Dict, current: Dict):
    """Initialise the cache; the multi-stage path keeps decompositions in a dict."""
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
    Store the feature and its decompositions, across stages.

    For every full step, decompose with the current stage's predictor and the later stages'.
    A later stage may need this step's decomposition as a reference point.
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

    if len(_GLOBAL_PREDICTORS) == 0 or _GLOBAL_PREDICTORS[0] is None:
        return

    current_stage_idx = get_stage_index(step)

    with torch.no_grad():
        original_dtype = feature.dtype
        feature_float = feature.float() if original_dtype != torch.float32 else feature

        decomposed_entry = {}
        # Decompose with the current stage's predictor and every later stage's
        for si in range(current_stage_idx, len(_GLOBAL_PREDICTORS)):
            pred = _GLOBAL_PREDICTORS[si]
            z0, z1, z2 = pred.decompose(feature_float)
            decomposed_entry[f'stage{si}'] = (
                z0.to(original_dtype).clone(),
                z1.to(original_dtype).clone(),
                z2.to(original_dtype).clone(),
            )

        update_dic[-1][current['module']]['decomposed'][step] = decomposed_entry


def cache_step_learned(cache_dic: Dict, current: Dict) -> torch.Tensor:
    """Multi-stage prediction: current_step selects both the predictor and the decomposition key."""
    current_step = current['step']
    predictor = get_predictor(current_step)
    if predictor is None:
        raise RuntimeError("Predictor not loaded! Call load_predictor() first.")

    key = get_stage_key(current_step)
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

    forecast_method = cache_dic.get('forecast_method', 'lagrange')

    with torch.no_grad():
        original_dtype = decomposed_cache_list[0][1].dtype
        decomposed_cache_float = [
            (d, z0.float(), z1.float(), z2.float())
            for d, z0, z1, z2 in decomposed_cache_list
        ]
        if forecast_method == 'lagrange':
            predicted = predictor.predict_from_decomposed(decomposed_cache_float, order=order)
        else:
            # hermite or taylor, applied to each partition independently
            max_order = cache_dic.get('max_order', 2)
            predicted = _predict_with_formula(
                cache_dic, predictor, decomposed_cache_float, max_order, torch.float32
            )
        predicted = predicted.to(original_dtype)

    return predicted


def cache_step_merge_learned(cache_dic: Dict, current: Dict) -> torch.Tensor:
    """Merge predictions across stages."""
    current_step = current['step']
    predictor = get_predictor(current_step)
    if predictor is None:
        raise RuntimeError("Predictor not loaded!")

    key = get_stage_key(current_step)
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

        forecast_method = cache_dic.get('forecast_method', 'lagrange')
        max_order = cache_dic.get('max_order', 2)

        if forecast_method == 'lagrange':
            last_order = min(2, len(last_decomposed_float) - 1) if len(last_decomposed_float) > 0 else 0
            curr_order = min(2, len(curr_decomposed_float) - 1) if len(curr_decomposed_float) > 0 else 0

            last_pred = predictor.predict_from_decomposed(last_decomposed_float, order=last_order).to(original_dtype) if len(last_decomposed_float) > 0 else None
            curr_pred = predictor.predict_from_decomposed(curr_decomposed_float, order=curr_order).to(original_dtype)
        else:
            # hermite or taylor
            last_pred = _predict_with_formula(
                cache_dic, predictor, last_decomposed_float, max_order, torch.float32
            ).to(original_dtype) if len(last_decomposed_float) > 0 else None
            curr_pred = _predict_with_formula(
                cache_dic, predictor, curr_decomposed_float, max_order, torch.float32
            ).to(original_dtype)

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


def pipeline_with_learned_cache(pipe, checkpoint_paths: List[str] = None, stage_splits_str: str = None, num_steps: int = 50):
    """Attach the learned cache to a pipeline; multi-stage takes a list of paths plus the stage splits."""
    import types
    from flux.modules.flux_model import FluxTransformerBlock

    if checkpoint_paths is not None and stage_splits_str is not None:
        device = next(pipe.transformer.parameters()).device
        load_predictor(checkpoint_paths, stage_splits_str, num_steps=num_steps, device=str(device))

    return pipe
