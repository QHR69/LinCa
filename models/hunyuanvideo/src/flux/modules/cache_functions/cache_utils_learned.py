"""
Cache utilities for learned invertible decomposition network

与 FreqCa 的 cache_utils.py 保持相同接口,
但使用可学习的可逆网络进行分解和预测

关键区别:
- 分解方法: FFT/DCT → 可逆网络 (RealNVP)
- 预测策略: Hermite/Taylor → 固定分区策略 (0阶/1阶/2阶)
- 重建方法: 简单相加 → 可逆网络逆变换
"""

import torch
import math
from typing import Dict, Tuple, Optional, List
import os
import sys

# Import invertible_net
from ..invertible_net import LearnedDecompositionPredictor, FixedPredictionStrategy


# 全局变量:存储加载的预测器
_GLOBAL_PREDICTOR: Optional[LearnedDecompositionPredictor] = None


def load_predictor(
    checkpoint_path: str,
    device: str = 'cuda',
) -> LearnedDecompositionPredictor:
    """
    加载训练好的预测器

    Args:
        checkpoint_path: 检查点路径
        device: device

    Returns:
        加载好的预测器
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
    """获取全局预测器"""
    return _GLOBAL_PREDICTOR


def set_predictor(predictor: LearnedDecompositionPredictor):
    """设置全局预测器"""
    global _GLOBAL_PREDICTOR
    _GLOBAL_PREDICTOR = predictor


def module_cache_init_learned(cache_dic: Dict, current: Dict):
    """
    Initialize cache for learned decomposition

    与原版 module_cache_init 保持相同接口
    """
    if cache_dic['use_z_cache']:
        if current['step'] == 0:
            # 存储原始特征和分解后的特征
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

    与原版 derivative_approximation 保持相同接口
    这里不计算导数,而是存储原始特征和分解后的特征
    """
    predictor = get_predictor()

    if cache_dic['use_z_cache']:
        if current['type'] == 'full':
            # 更新 last_cache
            if "last_cache" in cache_dic:
                cache_dic["last_cache"][-1][current["module"]] = cache_dic["cache"][-1][current["module"]].copy()
                cache_dic["last_cache"][-1][current["module"]]['features'] = cache_dic["cache"][-1][current["module"]]['features'].copy()
                cache_dic["last_cache"][-1][current["module"]]['decomposed'] = cache_dic["cache"][-1][current["module"]]['decomposed'].copy()

        input_dic = cache_dic['last_cache'] if current['type'] == 'cache' else cache_dic['cache']
        update_dic = cache_dic['cache']
    else:
        input_dic = cache_dic['cache']
        update_dic = cache_dic['cache']

    # 存储当前步的特征
    step = current['step']
    update_dic[-1][current['module']]['features'][step] = feature.clone()

    # 如果有预测器,存储分解后的特征
    if predictor is not None:
        with torch.no_grad():
            # 保存原始dtype,Cast to float32进行计算,然后转换回来
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

    优化版本:直接从 decomposed 缓存取已分解的 (z0, z1, z2),
    避免重复调用 forward,使用 predict_from_decomposed 进行预测
    """
    predictor = get_predictor()
    if predictor is None:
        raise RuntimeError("Predictor not loaded! Call load_predictor() first.")

    module_cache = cache_dic['cache'][-1][current['module']]

    # 获取激活步
    activated_steps = current['activated_steps']
    current_step = current['step']

    # 构建已分解的缓存列表 [(distance, z0, z1, z2), ...]
    decomposed_cache_list = []
    for act_step in reversed(activated_steps):  # 从最近的开始
        if act_step < current_step and act_step in module_cache['decomposed']:
            distance = current_step - act_step
            z0, z1, z2 = module_cache['decomposed'][act_step]
            decomposed_cache_list.append((distance, z0, z1, z2))

    # 按距离排序(最近的在前)
    decomposed_cache_list.sort(key=lambda x: x[0])

    # 取最多3个
    decomposed_cache_list = decomposed_cache_list[:3]

    if len(decomposed_cache_list) == 0:
        raise RuntimeError(f"No cache available for step {current_step}")

    # 根据缓存数量决定预测阶数
    if len(decomposed_cache_list) >= 3:
        order = 2
    elif len(decomposed_cache_list) >= 2:
        order = 1
    else:
        order = 0

    # 使用预测器进行预测(直接用已分解的缓存,避免重复 forward)
    with torch.no_grad():
        # Remember the original dtype (decomposed cache is bfloat16, compute in float32)
        original_dtype = decomposed_cache_list[0][1].dtype

        # Cast to float32进行计算
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

    优化版本:从 decomposed 缓存取已分解的数据,
    使用 predict_from_decomposed 避免重复 forward
    """
    predictor = get_predictor()
    if predictor is None:
        raise RuntimeError("Predictor not loaded! Call load_predictor() first.")

    # 从 last_cache 预测
    last_module_cache = cache_dic['last_cache'][-1][current['module']]
    # 从 cache 预测
    curr_module_cache = cache_dic['cache'][-1][current['module']]

    activated_steps = current['activated_steps']
    current_step = current['step']

    # 构建 last_cache 的已分解缓存列表
    last_decomposed_list = []
    for act_step in reversed(activated_steps[:-1]):  # 排除最新的激活步
        if act_step < current_step and act_step in last_module_cache['decomposed']:
            distance = current_step - act_step
            z0, z1, z2 = last_module_cache['decomposed'][act_step]
            last_decomposed_list.append((distance, z0, z1, z2))
    last_decomposed_list.sort(key=lambda x: x[0])
    last_decomposed_list = last_decomposed_list[:3]

    # 构建 curr_cache 的已分解缓存列表
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

        # Cast to float32进行计算
        last_decomposed_float = [
            (d, z0.float(), z1.float(), z2.float())
            for d, z0, z1, z2 in last_decomposed_list
        ]
        curr_decomposed_float = [
            (d, z0.float(), z1.float(), z2.float())
            for d, z0, z1, z2 in curr_decomposed_list
        ]

        # 从 last_cache 预测
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

        # 从 curr_cache 预测
        if len(curr_decomposed_float) >= 3:
            curr_order = 2
        elif len(curr_decomposed_float) >= 2:
            curr_order = 1
        else:
            curr_order = 0

        curr_pred = predictor.predict_from_decomposed(curr_decomposed_float, order=curr_order)

        # 如果没有 last_pred,直接返回 curr_pred (Cast back to the original dtype)
        if last_pred is None:
            return curr_pred.to(original_dtype)

        # 更新 activated_steps
        if current['update']:
            current['activated_steps'][-1] = current['step']

        # 合并两个预测
        weight = current['weight']
        merged = last_pred * (1 - weight) + curr_pred * weight

        # Cast back to the original dtype
        merged = merged.to(original_dtype)

    return merged


# ============ 兼容性包装函数 ============

def module_cache_init(cache_dic: Dict, current: Dict):
    """兼容性包装:根据 decompose_method 选择初始化方式"""
    if cache_dic.get('decompose_method') == 'learned':
        module_cache_init_learned(cache_dic, current)
    else:
        # 调用原版
        from .cache_utils import module_cache_init as original_init
        original_init(cache_dic, current)


def derivative_approximation(cache_dic: Dict, current: Dict, feature: torch.Tensor):
    """兼容性包装:根据 decompose_method 选择方式"""
    if cache_dic.get('decompose_method') == 'learned':
        derivative_approximation_learned(cache_dic, current, feature)
    else:
        from .cache_utils import derivative_approximation as original_func
        original_func(cache_dic, current, feature)


def cache_step(cache_dic: Dict, current: Dict) -> torch.Tensor:
    """兼容性包装:根据 decompose_method 选择预测方式"""
    if cache_dic.get('decompose_method') == 'learned':
        return cache_step_learned(cache_dic, current)
    else:
        from .cache_utils import cache_step as original_func
        return original_func(cache_dic, current)


def cache_step_merge(cache_dic: Dict, current: Dict) -> torch.Tensor:
    """兼容性包装:根据 decompose_method 选择合并方式"""
    if cache_dic.get('decompose_method') == 'learned':
        return cache_step_merge_learned(cache_dic, current)
    else:
        from .cache_utils import cache_step_merge as original_func
        return original_func(cache_dic, current)
