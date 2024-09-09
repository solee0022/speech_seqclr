import logging
import torch
import torch.nn as nn
import numpy as np

from seqclr.modules.projections import BidirectionalLSTM, AttnLinear
from seqclr.modules.utils import if_none
from seqclr.modules.model import Model

class SeqCLRProj(Model):
    def __init__(self, config):
        super().__init__()
        self.model_instance_mapping_frame_to_instance = config.seqclr_instance_mapping_frame_to_instance
        if config.seqclr_speech_backbone == 'large':
            self.emb_dim = 1024
        else:
            self.emb_dim = 768
        projection_input_size = self.emb_dim # emb_dim of w2v-base: 768, w2v-large: 1024
        projection_hidden_size = self.emb_dim
        projection_output_size = self.emb_dim

        # 3/Projection head
        if config.seqclr_proj_scheme is None:
            self.projection = nn.Identity()
        elif config.seqclr_proj_scheme == 'bilstm':
            self.projection = BidirectionalLSTM(projection_input_size, 
                                                projection_hidden_size,
                                                projection_output_size)
        elif config.seqclr_proj_scheme == 'linear_per_column':
            self.projection = nn.Linear(projection_input_size, projection_output_size)
        elif config.seqclr_proj_scheme == 'attn_linear_per_column':
            self.projection = AttnLinear(projection_input_size,
                                         projection_hidden_size,
                                         projection_output_size)
        else:
            raise NotImplementedError(f'The projection scheme of {config.seqclr_proj_scheme} is not supported.')


        # 4/Instance-mapping
        if config.seqclr_instance_mapping_frame_to_instance:
            self.instance_mapping_func = nn.Identity()
        else:
            instance_mapping_fixed = if_none(config.seqclr_instance_mapping_fixed, 'instances')
            w = if_none(config.seqclr_instance_mapping_w, 5)
            if instance_mapping_fixed == 'instances':
                self.instance_mapping_func = nn.AdaptiveAvgPool2d((w, projection_output_size))
            elif instance_mapping_fixed == 'frames':
                self.instance_mapping_func = AvgPool(kernel_size=w, stride=w)
            else:
                raise NotImplementedError(f'instance_mapping_fixed of {instance_mapping_fixed} is not supported')
            
        if config.seqclr_proj_checkpoint is not None:
            logging.info(f'Read projection head model from {config.seqclr_proj_checkpoint}.')
            self.load(config.seqclr_proj_checkpoint)

    def _single_forward(self, output):
        """
        output: one sample of base encoder output (torch.FloatTensor of shape (sequence_length, hidden_size))
        """
        projected_features = self.projection(output)
        projected_instances = self.instance_mapping_func(projected_features)
        return projected_instances
    
    def forward(self, output, *args):
        #if isinstance(output, (tuple, list)):
        with torch.no_grad():
            if self.model_instance_mapping_frame_to_instance:
                projected_instances = np.array([self._single_forward(o).cpu().numpy() for o in output])
            else:
                projected_instances = np.array([self._single_forward(o.unsqueeze(0)).cpu().numpy() for o in output])
        return projected_instances

class AvgPool(nn.Module):
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.avg_pool = nn.AvgPool1d(kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        return self.avg_pool(x.permute(0, 2, 1)).permute(0, 2, 1).contiguous()
    
    