import copy

def cache_init(kwargs):   
    '''
    Initialization for cache.
    '''
    cache = {}
    cache[-1]={}

    cache_dic = {}
    cache_dic['cache'] = cache
    cache_dic['last_cache'] = copy.deepcopy(cache)
    cache_dic['num_steps'] = kwargs['num_steps']
    cache_dic['test_FLOPs'] = kwargs['test_FLOPs']
    cache_dic['monitor_gpu_usage'] = kwargs['monitor_gpu_usage']
    cache_dic['interval'] = kwargs['interval']
    cache_dic['max_order'] = kwargs['max_order']
    cache_dic['min_order'] = kwargs['min_order']
    cache_dic['first_enhance'] = kwargs['first_enhance']
    cache_dic['forecast_method'] = kwargs['forecast_method']
    cache_dic['decompose_method'] = kwargs['decompose_method']
    cache_dic['use_z_cache'] = kwargs['use_z_cache']
    cache_dic['forecast_steps'] = kwargs['forecast_steps']       

    current = {}
    current['step'] = 0
    current['activated_steps'] = [0]
    current['prev'] = 0
    current['cache_counter'] = 0
    current['img'] = None
    current['img_backup'] = None
    current['update'] = False
    current['merge'] = False
    current['weight'] = 0.0

    return cache_dic, current

