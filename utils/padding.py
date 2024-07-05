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
from seqclr.modules.model_seqclr import SeqCLRModel_tmp
#import seqclr.modules.whisper as whisper
import whisper
from seqclr.modules.utils import Config


ds_eval = UASpeechDataset("eval") 
dl_eval = DataLoader(
dataset=ds_eval,
batch_size=4,
num_workers=mp.cpu_count(),
pin_memory=True,
drop_last=True,
)

for idx, batch in enumerate(dl_eval):
    if idx < 2:
        print(batch["input_features"])
        print(batch["input_features"].shape)
    else:
        break