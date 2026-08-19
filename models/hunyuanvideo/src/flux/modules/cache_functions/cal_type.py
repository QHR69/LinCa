import math

def cal_type(cache_dic, current):
    '''
    Determine calculation type for this step
    '''
    if current['step'] < cache_dic['first_enhance']:
        current['type'] = 'full'

        current['cache_counter'] = 0
        current['activated_steps'].append(current['step'])

    else:
        if cache_dic['use_z_cache'] and (current['step'] == cache_dic['first_enhance'] or (current['merge'] and current['update'])):
            current['merge'] = False
            current['update'] = False
            current['prev'] = current['step'] + cache_dic['forecast_steps']

        if cache_dic['use_z_cache'] and current['prev'] < cache_dic['num_steps'] - 1 - cache_dic['forecast_steps']:
            if not current['merge'] and not current['update']:
                if current['cache_counter'] != cache_dic['forecast_steps'] - 1:
                    if current['step'] != cache_dic['first_enhance']:
                        current['type'] = 'cache'
                    else:
                        current['type'] = 'full'
                        current['activated_steps'].append(current['step'])

                    if current['img_backup'] is None:
                        current['img_backup'] = current['img'].clone()

                    current['cache_counter'] += 1

                else:
                    current['type'] = 'full'

                    current['merge'] = True
                    current['cache_counter'] = 0
                    current['activated_steps'].append(current['step'])

            elif current['merge'] and not current['update']:
                current['type'] = 'cache'

                if current['img_backup'] is not None:
                    current['img'] = current['img_backup'].clone()
                    current['img_backup'] = None

                    current['step'] = current['prev'] - cache_dic['forecast_steps']

                curr_span = abs(current['activated_steps'][-2] - current['step'])
                total_span = abs(current['activated_steps'][-2] - current['activated_steps'][-1])
                current['weight'] = math.log(curr_span + 1) / math.log(total_span + 1)

                if current['cache_counter'] == cache_dic['interval'] - 1:
                    current['update'] = True
                    current['cache_counter'] = -1

                current['cache_counter'] += 1

        else:
            if "last_cache" in cache_dic:
                del cache_dic["last_cache"]

            if current['cache_counter'] == cache_dic['interval'] - 1:
                current['type'] = 'full'

                current['cache_counter'] = 0
                current['activated_steps'].append(current['step'])
            
            else:
                current['cache_counter'] += 1
                current['type'] = 'cache'
