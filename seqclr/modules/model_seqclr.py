from transformers import WhisperConfig, WhisperModel
import whisper
from seqclr.modules.seqclr_proj import SeqCLRProj
from seqclr.modules.model import Model
from seqclr.modules.utils import if_none

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForSpeechSeq2Seq


base_config = AutoConfig.from_pretrained("openai/whisper-base")
base_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        "openai/whisper-base",
        config=base_config,
    )

class SeqCLRModel(Model):
    def __init__(self, config):
        super().__init__()
        self.encoder = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-base").model.encoder
        self.seqclr_proj = SeqCLRProj(config)
        self.loss_weight = 1.0

    def forward(self, x, *args, **kwargs):
        """
        x: log-mel spectrogram, output of whisper feature-extractor
        """
        # Get output of base encoder as input
        base_emb = self.encoder(x).last_hidden_state
        projected_instances = torch.tensor(self.seqclr_proj(base_emb))
        #print(f"projected_features.shape: {projected_features['instances'].shape}") 
        return {'instances': projected_instances, #(batch_size, seq_len, emb_size)
                'loss_weight': self.loss_weight}

    