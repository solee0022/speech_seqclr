import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torch.nn.functional as F

from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import Wav2Vec2Processor
from accelerate import Accelerator
                          
import os
import argparse
from tqdm import tqdm
import wandb
import numpy as np

from seqclr.dataset.sampler import UniqueClassSampler
from seqclr.dataset.ua import UASpeechDataset
from seqclr.losses.seqclr_loss import SeqCLRLoss
from seqclr.modules.model_seqclr import SeqCLRModel
from seqclr.modules.utils import Config
from seqclr.modules.dtw import window_mapping_and_character

from dataclasses import dataclass
import logging
from typing import Any, Dict, List, Optional, Union

# 0. Arguments Setting
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default="seqclr/configs/seqclr_model.yaml",
                    required=True, help='path to config file')
args = parser.parse_args()
config = Config(args.config)
wandb.init(project=config.wandb_project_name, name=config.wandb_run_name)

@dataclass
class DataCollatorWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = "longest"
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        collate_y = [sample['label'] for sample in features]

        collate_seg = []
        max_len = max([len(sample['segments']) for sample in features])
        for sample in features:
            diff = max_len-len(sample['segments'])
            if diff > 0:
                zero_pad = np.array([[0,0] for _ in range(diff)], dtype=np.float16)
                collate_seg.append(np.concatenate((sample['segments'], zero_pad), axis=0))
            else:
                collate_seg.append(sample['segments'])

        batch["label"] = collate_y
        batch["segments"] = collate_seg
        if "attention_mask" in batch:
            batch["attention_mask"] = batch["attention_mask"].to(torch.long)

        return batch
    
if __name__ == "__main__":
    # 0. Multi-GPUs Settings
    # (1) Initialize GPU process
    dist_url = 'env://'
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    num_samples = 2
    
    # torch.distributed.init_process_group(backend='nccl', init_method=dist_url, world_size=world_size, rank=rank)
    # torch.cuda.set_device(local_rank)
    # torch.distributed.barrier()
    # if torch.distributed.get_rank() == 0:
    #     print(f'RANK {rank}   WORLD_SIZE {world_size}   LOCAL_RANK {local_rank}')
        
    # 0. outdir
    try:
        os.makedirs(config.seqclr_outdir)
    except FileExistsError:
        pass
    
    # 0. Load dataset
    # (2) Setting sampler
    ds_train = UASpeechDataset(config.seqclr_dataset_train_mode, config.seqclr_speech_backbone) # UASpeechDataset("train", cfg) -> cfg.train_roots
    ds_eval= UASpeechDataset(config.seqclr_dataset_test_mode, config.seqclr_speech_backbone)
    
    processor = Wav2Vec2Processor.from_pretrained(f"facebook/wav2vec2-{config.seqclr_speech_backbone}")
    collate_fn = DataCollatorWithPadding(processor=processor)
    train_sampler = UniqueClassSampler(ds_train.ys, num_samples, rank=rank, world_size=world_size)
    eval_sampler = UniqueClassSampler(ds_eval.ys, num_samples, rank=0, world_size=1)
    
    dl_train = DataLoader(
    dataset=ds_train,
    sampler=train_sampler,
    collate_fn=collate_fn,
    batch_size=config.training_train_bs//world_size,
    num_workers=mp.cpu_count() // world_size,
    pin_memory=True,
    drop_last=True,
    )
    
    dl_eval = DataLoader(
    dataset=ds_eval,
    sampler=eval_sampler,
    collate_fn=collate_fn,
    batch_size=config.training_eval_bs,
    num_workers=mp.cpu_count(),
    pin_memory=True,
    drop_last=True,
    )
    
    # 0. Load model
    # (3) DDP model
    model = SeqCLRModel(config).to(rank) # Put model on GPU
    if config.model_backbone_freeze:
        for param in model.encoder.parameters():
            param.requires_grad = False
        for param in model.seqclr_proj.parameters():
            param.requires_grad = True
    
    ## loss
    loss_f = SeqCLRLoss().to(local_rank)
    ## optimizer
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    ## scheduler
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                            lr_lambda=lambda epoch: 0.95 ** epoch,
                                            last_epoch=-1,
                                            verbose=False)
    # Print model's state_dict
    for child in model.children():
        print('===')
        print(child)

    # compute_loss func
    def compute_loss(batch):
        # forward pass
        x = batch["input_values"]
        y = batch["label"]
        SEGMENTS = batch["segments"]
            
        # 2+3+4. base_emb + Projection head + Instance Mapping
        x = x.cuda(non_blocking=True)
        z = model(x)['instances'] # [num_samples, n_mels, 3000] -> [num_samples, seq_len, emb_dim]

        loss = 0
        for i in range(len(y)):
            for j in range(len(y)):
                if i!=j:
                    # 1. Embedding mapping
                    # 2+3+4. Projection head+Instance Mapping    
                    zi, zj = torch.from_numpy(z[i]).unsqueeze(0), torch.from_numpy(z[j]).unsqueeze(0) #torch.Size([1,seq_len, emb_dim])
                    mapped_zi, mapped_zj = window_mapping_and_character(zi, zj, SEGMENTS[i], SEGMENTS[j])
                        
                    # 5. Contrastive loss
                    l = loss_f(mapped_zi, mapped_zj)
                    loss += l
        return loss
    
    accelerator = Accelerator()
    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)
    # 1. Train!
    global_step = 0
    for ep in range(config.training_epochs):
        # (4) Let sampler know current epoch.
        train_sampler.set_epoch(ep)
        model.train()
        avg_loss = 0.0
        dataloader_with_bar = tqdm(dl_train, desc=f"Epoch {ep + 1}/{config.training_epochs}", leave=False)
        

        for step, batch in enumerate(dataloader_with_bar):
            # We could avoid this line since we set the accelerator with `device_placement=True`
            # batch.to(accelerator.device)
            global_step += 1
            
            
            with accelerator.accumulate(model):
                
                loss = compute_loss(batch)
                # for gradient accumulation
                avg_loss += loss
                
            # -----Update----- #
            wandb.log({"train/loss": loss})
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
    
            # 2. Save model
            print(f"Step: {step+1},    Training Loss: {loss.item()}")
            if (step+1) % 500 == 0:
                model.eval()
                with torch.no_grad():
                    val_loss = []
                    for batch in dl_eval:
                        batch = torch.tensor(batch)
                        batch.to(accelerator.device)
                        loss = compute_loss(batch)
                        val_loss.append(loss)
                    val_loss = np.mean(val_loss)
                    print(f"Validation Loss: {val_loss}")
                    wandb.log({"eval/loss": val_loss})
                    unwrapped_model = accelerator.unwrap_model(model)
                    accelerator.save({
                        "model": unwrapped_model.state_dict(),
                        "optimizer": optimizer.optimizer.state_dict() # optimizer is an AcceleratedOptimizer object
                    }, os.path.join(config.seqclr_outdir, f'seqclr-{step}.pt'))
                    print(f"save model at {step} step")