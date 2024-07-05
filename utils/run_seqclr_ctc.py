import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

import os
import argparse
from tqdm import trange
import wandb
import multiprocessing
from functools import partial
from typing_extensions import Literal
import librosa

from seqclr.dataset.sampler import UniqueClassSampler
from seqclr.dataset.ua import UASpeechDataset
from seqclr.losses.seqclr_loss import SeqCLRLoss
from seqclr.modules.model_seqclr import SeqCLRModel
#import seqclr.modules.whisper as whisper
import whisper
from seqclr.modules.utils import Config

from seqclr.modules.dtw import dtw_mapping, dtw_mapping_window


# 0. Arguments Setting
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default="seqclr/configs/seqclr_model.yaml",
                    required=True, help='path to config file')
args = parser.parse_args()
config = Config(args.config)
wandb.init(project=config.wandb_project_name, name=config.wandb_run_name)

num_samples = 4
epochs = config.training_epochs


if __name__ == "__main__":
    # 0. Multi-GPUs Settings
    # (1) Initialize GPU process
    dist_url = 'env://'
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    
    torch.distributed.init_process_group(backend='nccl', init_method=dist_url, world_size=world_size, rank=rank)

    torch.cuda.set_device(local_rank)
    torch.distributed.barrier()

    if torch.distributed.get_rank() == 0:
        print(f'RANK {rank}   WORLD_SIZE {world_size}   LOCAL_RANK {local_rank}')
        
    # 0. outdir
    try:
        os.makedirs(config.model_outdir)
    except FileExistsError:
        pass


    # 0. Load dataset
    # (2) Setting sampler
    ds_train = UASpeechDataset(config.model_dataset_train_mode) # UASpeechDataset("train", cfg) -> cfg.train_roots
    ds_eval = UASpeechDataset(config.model_dataset_test_mode) 

    # sampler = UniqueClassSampler(
    #     ds_train.ys, cfg.num_samples, cfg.local_rank, world_size
    # )

    sampler = UniqueClassSampler(ds_train.ys, num_samples, rank=rank, world_size=world_size)

    dl_train = DataLoader(
    dataset=ds_train,
    sampler=sampler,
    batch_size=config.training_train_bs//world_size,
    num_workers=mp.cpu_count() // world_size,
    pin_memory=True,
    drop_last=True,
    )

    dl_eval = DataLoader(
    dataset=ds_eval,
    sampler=sampler,
    batch_size=config.training_eval_bs//world_size,
    num_workers=mp.cpu_count() // world_size,
    pin_memory=True,
    drop_last=True,
    )
    
    # 0. Load model
    # (3) DDP model
    ## seqclr_model
    seqclr_model = SeqCLRModel(config).to(rank) # Put model on GPU
    if config.model_backbone_freeze:
        for param in seqclr_model.encoder.parameters():
            param.requires_grad = False
        for param in seqclr_model.seqclr_proj.parameters():
            param.requires_grad = True

    seqclr_model = DDP(seqclr_model, device_ids=[local_rank], find_unused_parameters=True)

    ## loss
    loss_f = SeqCLRLoss().to(local_rank)
    ## optimizer
    optimizer = optim.AdamW(seqclr_model.parameters(), lr=3e-4)
    ## scheduler
    scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                            lr_lambda=lambda epoch: 0.95 ** epoch,
                                            last_epoch=-1,
                                            verbose=False)

    # Print model's state_dict
    for child in seqclr_model.children():
        print('===')
        print(child)
        

    # 1. Train!
    for ep in trange(epochs):
        # (4) Let sampler know current epoch.
        sampler.set_epoch(ep)
        stats_ep = []
        for iter, batch in enumerate(dl_train): # 1-batch
            x = batch["input_features"]
            y = batch["label"]
            y = torch.tensor(list(y))
            y = y.view(len(y) // num_samples, num_samples)
            assert (y[:, 0] == y[:, -1]).all()
            s = y[:, 0].tolist()
            assert len(set(s)) == len(s)

            # Base Encoder
            x = x.cuda(non_blocking=True)
            #r = base_encoder(x) 

            # 2+3+4. base_emb + Projection head + Instance Mapping
            z = seqclr_model(x)['instances'] 
            if config.model_instance_mapping_frame_to_instance:
                seq_len = 1500
            else:
                seq_len = config.model_instance_mapping_w
            x = x.view(len(x) // num_samples, num_samples, 80, 3000) #[batch_size, whisper-base,small=80 or large=128, CHUNK_LENGTH*100)
            z = z.view(len(z) // num_samples, num_samples, seq_len, 512) #emb=output embedding size

            for word in range(len(z)):
                # pair of i,j in same class 
                for i in range(num_samples):
                    for j in range(num_samples):
                        if i!=j:
                            # 1.1. Dynamic Time Warping(get index pair)
                            xi_mel = x[word][i].squeeze(0)
                            xj_mel = x[word][j].squeeze(0)
                            D, wp = librosa.sequence.dtw(X=xi_mel.cpu().numpy(), Y=xj_mel.cpu().numpy(), metric='cosine')
                            wp = wp[::-1] # index seq is reversed.
                            # 1.2. Embedding mapping
                            zi, zj = z[word][i].unsqueeze(0), z[word][j].unsqueeze(0) #torch.Size([1,1500,512])
                            
                            if config.model_instance_mapping_frame_to_instance:
                                mapped_zi, mapped_zj = dtw_mapping(zi, zj, wp) #torch.Size([1,1500,512])
                            elif not config.model_instance_mapping_dtw:
                                mapped_zi, mapped_zj = zi, zj
                            else:
                                old_wp = wp
                                new_wp = []
                                w = config.model_instance_mapping_w
                                term = 1500//w
                                for wp_t in range(term-1, w, term): # 150
                                    x1_t, x2_t = old_wp[wp_t]
                                    new_wp_t = []
                                    new_wp_t.append(x1_t//term)
                                    new_wp_t.append(x2_t//term)
                                    new_wp.append(new_wp_t)
                                mapped_zi, mapped_zj = dtw_mapping_window(zi, zj, new_wp, w) #torch.Size([1,1500,512])

                            # 2+3+4. Projection head+Instance Mapping       
                            # 5. Contrastive loss
                            loss = 0
                            l = loss_f(mapped_zi, mapped_zj)
                            loss += l
                            stats_ep.append({"loss": l.item()})
            # -----Update----- #
            wandb.log({"train/loss": loss})
            optimizer.zero_grad()
            loss.requires_grad_(True)
            loss.backward()
            optimizer.step()

        # 2. Save model
        if ep in range(0, 401, 20):
            print(f'loss: {loss}')
            if torch.distributed.get_rank() == 0:
                torch.save(seqclr_model, os.path.join(config.model_outdir, f'seqclr-{ep}.pt'))
                print(f"save model {ep} ep !!")


