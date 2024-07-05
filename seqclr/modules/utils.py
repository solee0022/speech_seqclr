import logging
import os
import time

import numpy as np
import torch
import yaml
from matplotlib import colors
from matplotlib import pyplot as plt
from torch import Tensor, nn

def if_none(a, b):
    return b if a is None else a

class Config(object):

    def __init__(self, config_path, host=True):
        def __dict2attr(d, prefix=''):
            for k, v in d.items():
                if isinstance(v, dict):
                    __dict2attr(v, f'{prefix}{k}_')
                else:
                    if k == 'phase':
                        assert v in ['train', 'test']
                    if k == 'stage':
                        assert v in ['pretrain-speech', 'train-decoder', 'pretrain-vision']
                    self.__setattr__(f'{prefix}{k}', v)

        assert os.path.exists(config_path), '%s does not exists!' % config_path
        with open(config_path) as file:
            config_dict = yaml.safe_load(file)
        with open('seqclr/configs/template.yaml') as file:
            default_config_dict = yaml.safe_load(file)
        __dict2attr(default_config_dict)
        if 'global' in config_dict.keys() and 'experiment_template' in config_dict['global'].keys():
            with open(f"configs/{config_dict['global']['experiment_template']}") as file:
                default_exp_config_dict = yaml.safe_load(file)
            __dict2attr(default_exp_config_dict)
        __dict2attr(config_dict)
        self.global_workdir = os.path.join(self.global_workdir, self.global_name)

    def __getattr__(self, item):
        attr = self.__dict__.get(item)
        if attr is None:
            attr = dict()
            prefix = f'{item}_'
            for k, v in self.__dict__.items():
                if k.startswith(prefix):
                    n = k.replace(prefix, '')
                    attr[n] = v
            return attr if len(attr) > 0 else None
        else:
            return attr

    def __repr__(self):
        str = 'ModelConfig(\n'
        for i, (k, v) in enumerate(sorted(vars(self).items())):
            str += f'\t({i}): {k} = {v}\n'
        str += ')'
        return str
