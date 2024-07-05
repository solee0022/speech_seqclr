import collections
import logging
import torch
import torch.nn as nn


_default_tfmer_cfg = dict(d_model=512, nhead=8, d_inner=2048,  # 1024
                          dropout=0.1, activation='relu')


class Model(nn.Module):

    def __init__(self):
        super().__init__()

    def load(self, source, device=None, strict=True, submodule=None, exclude=None):
        state = torch.load(source, map_location=device)
        if source.endswith('.ckpt'):
            model_dict = state['state_dict']
            if list(model_dict.keys())[0].startswith('model.'):
                model_dict = collections.OrderedDict(
                    {k[6:]: v for k, v in model_dict.items() if k.startswith('model.')})
        else:
            model_dict = state
            if 'model' in model_dict:
                model_dict = model_dict['model']

        if submodule is None:
            stat = self.load_state_dict(model_dict, strict=strict)
        else:
            submodule_dict = collections.OrderedDict(
                {k.split('.', 1)[1]: v for k, v in model_dict.items()
                 if k.split('.', 1)[0] == submodule and k.split('.')[1] != exclude}
            )
            stat = self.load_state_dict(submodule_dict, strict=strict and exclude is None)
        if stat.missing_keys or stat.unexpected_keys:
            logging.warning(f'Loading model with missing keys: {stat.missing_keys}'
                            f' and unexpected keys: {stat.unexpected_keys}')