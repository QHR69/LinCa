from .cache_init import cache_init
from .cal_type import cal_type
# 导入兼容性包装函数（会根据decompose_method自动选择learned或原版）
from .cache_utils_learned import (
    module_cache_init,
    derivative_approximation,
    cache_step,
    cache_step_merge,
    load_predictor,
    get_predictor,
    set_predictor_two_stage,
    module_cache_init_learned,
    derivative_approximation_learned,
    cache_step_learned,
    cache_step_merge_learned,
    pipeline_with_learned_cache,
)