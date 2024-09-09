import os
import torch
import torch.nn.functional as F
import numpy as np
from torch import nn

import logging
from typing import Any, Dict, List, Optional, Union

from seqclr.modules.dtw import window_mapping_and_character
from seqclr.losses.seqclr_loss import SeqCLRLoss
from seqclr.dataset.sampler import UniqueClassSampler
from transformers import Wav2Vec2Processor

from transformers import Trainer
from torch.utils.data import DataLoader

class CustomTrainer(Trainer):
    def my_collate_fn(self, samples):
        collate_X = [{"input_values": feature["input_values"]} for feature in samples]
        processor = Wav2Vec2Processor.from_pretrained(f"facebook/wav2vec2-base-960h")
        collate_X = processor.feature_extractor.pad(collate_X, return_tensors="pt")
        collate_y = [sample['label'] for sample in samples]
        collate_seg = []
        max_len = max([len(sample['segments']) for sample in samples])
        for sample in samples:
            diff = max_len-len(sample['segments'])
            if diff > 0:
                zero_pad = np.array([[0,0] for _ in range(diff)], dtype=np.float16)
                collate_seg.append(np.concatenate((sample['segments'], zero_pad), axis=0))
            else:
                collate_seg.append(sample['segments'])
        return {'input_values': collate_X,
                'label': collate_y,
                'segments': collate_seg}
        
    def get_train_dataloader(self) -> DataLoader:
        train_dataset = self.train_dataset
        labels = train_dataset.ys
        num_samples=8
        
        train_sampler = UniqueClassSampler(
            labels=labels, m_per_class=num_samples, rank=0, world_size=self.args.world_size
        )
        return DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.my_collate_fn,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def get_eval_dataloader(self, eval_dataset) -> DataLoader:
        labels = eval_dataset.ys
        num_samples=8
        
        eval_sampler = UniqueClassSampler(
            labels=labels, m_per_class=num_samples, rank=0, world_size=self.args.world_size
        )
        return DataLoader(
            eval_dataset,
            batch_size=self.args.eval_batch_size,
            sampler=eval_sampler,
            collate_fn=self.my_collate_fn,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
    
    def compute_loss(self, model, inputs, return_outputs=False):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loss_f = SeqCLRLoss().to(device)
        
        # forward pass
        x = inputs["input_values"]
        y = inputs["label"]
        SEGMENTS = inputs["segments"]
        num_samples = 4
            
        # 2+3+4. base_emb + Projection head + Instance Mapping
        x = x.to(device)
        z = model(x)['instances'] # [num_samples, n_mels, 3000] -> [num_samples, seq_len, emb_dim]

        loss = 0
        for i in range(num_samples):
            for j in range(num_samples):
                if i!=j:
                    # 1. Embedding mapping
                    # 2+3+4. Projection head+Instance Mapping    
                    zi, zj = z[i].unsqueeze(0), z[j].unsqueeze(0) #torch.Size([1,1500,512])
                    mapped_zi, mapped_zj = window_mapping_and_character(zi, zj, SEGMENTS[i], SEGMENTS[j])
                        
                    # 5. Contrastive loss
                    l = loss_f(mapped_zi, mapped_zj)
                    loss += l
        loss = loss.to(device)
        loss.requires_grad_(True)
        return loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ):
        loss = self.compute_loss(model, inputs, return_outputs=False)
        return (loss, None, None)