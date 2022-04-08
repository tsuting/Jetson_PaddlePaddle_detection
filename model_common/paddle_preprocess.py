import copy
import numpy as np
from .paddle_operators import *


PREPROCESS_CONFIG = [
    {
        'DetResizeForTest': {
                # 'image_shape': [1088, 1440]  # uncomment this if we are using static model
        }
    }, 
    {
        'NormalizeImage': {
            'std': [0.229, 0.224, 0.225],
            'mean': [0.485, 0.456, 0.406],
            'scale': '1./255.',
            'order': 'hwc'
        }
    }, 
    {
        'ToCHWImage': None
    }, 
    {
        'KeepKeys': {
        'keep_keys': ['image', 'shape']
    }
}]


def transform(data, ops=None):
    """transform"""
    if ops is None:
        ops = []
    for op in ops:
        data = op(data)
        if data is None:
            return None
    return data


def create_operators(op_param_list, global_config=None):
    """
    create operators based on the config
    Args:
        params(list): a dict list, used to create some operators
    """
    assert isinstance(op_param_list, list), "operator config should be a list"
    ops = []
    for operator in op_param_list:
        assert isinstance(operator, dict) and len(operator) == 1, "yaml format error"
        op_name = list(operator)[0]
        param = {} if operator[op_name] is None else operator[op_name]
        if global_config is not None:
            param.update(global_config)
        op = eval(op_name)(**param)
        ops.append(op)
    return ops



class td_PaddlePaddlePreProcess(object):

    def __init__(self):

        self.preprocess_config = copy.deepcopy(PREPROCESS_CONFIG)

    def __call__(self, raw_img, dimension):

        data = {"image": raw_img}

        if dimension:
            h, w = dimension
            self.preprocess_config[0]["DetResizeForTest"]["image_shape"] = [h, w]
 
        pre_processors = create_operators(self.preprocess_config)   
        data = transform(data, pre_processors)
    
        img, shape_list = data
        img = np.expand_dims(img, axis=0)
        shape_list = np.expand_dims(shape_list, axis=0)

        # to_postprocess
        to_postprocess = {"shape_list": shape_list}
        
        return img, to_postprocess