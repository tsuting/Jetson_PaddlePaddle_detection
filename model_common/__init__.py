from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import copy
from .paddle_decoder import td_PaddlePaddlePostProcess
from .paddle_preprocess import td_PaddlePaddlePreProcess


def build_post_process(config, global_config=None):
    support_dict = [
        "td_PaddlePaddlePostProcess",
    ]

    config = copy.deepcopy(config)
    module_name = config.pop("name")
    if global_config is not None:
        config.update(global_config)
    assert module_name in support_dict, Exception(
        "post process only support {}".format(support_dict)
    )
    module_class = eval(module_name)(**config)
    return module_class


def build_pre_process(config, global_config=None):
    support_dict = [
        "td_PaddlePaddlePreProcess",
    ]

    config = copy.deepcopy(config)
    module_name = config.pop("name")
    if global_config is not None:
        config.update(global_config)
    assert module_name in support_dict, Exception(
        "pre process only support {}".format(support_dict)
    )
    module_class = eval(module_name)(**config)
    return module_class