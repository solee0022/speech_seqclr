from seqclr.modules.seqclr_proj import SeqCLRProj
from seqclr.modules.model import Model
from seqclr.modules.utils import if_none

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model

from peft import prepare_model_for_kbit_training
from peft import LoraConfig, PeftModel, LoraModel, LoraConfig, get_peft_model

class SeqCLRModel(Model):
    def __init__(self, config):
        super().__init__()
        self.encoder = Wav2Vec2Model.from_pretrained(f"facebook/wav2vec2-{config.seqclr_speech_backbone}-960h")
        self.seqclr_proj = SeqCLRProj(config)
        self.loss_weight = 1.0

    def forward(self, x, *args, **kwargs):
        """
        x: log-mel spectrogram, output of w2v feature-extractor
        """
        # Get output of base encoder as input
        with torch.no_grad():
            outputs = self.encoder(x['input_values'])
        base_emb = outputs.last_hidden_state
        projected_instances = torch.tensor(self.seqclr_proj(base_emb))
        #print(f"projected_features.shape: {projected_features['instances'].shape}") 
        return {'instances': projected_instances, #(batch_size, seq_len, emb_size) ex. [1, 292, 768]
                'loss_weight': self.loss_weight}
