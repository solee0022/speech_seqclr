import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import CrossEntropyLoss

import os
import argparse
from tqdm import trange
import wandb
import multiprocessing
from functools import partial
from typing_extensions import Literal
import librosa
from typing import Any, Dict, List, Optional, Union

from seqclr.dataset.ua import UASpeechDataset
from seqclr.modules.model_seqclr import SeqCLRModel
import whisper
from transformers import WhisperConfig, WhisperProcessor, WhisperModel
from seqclr.modules.utils import Config
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer
import evaluate
from dataclasses import dataclass, field

from seqclr.modules.model_seqclr import SeqCLRModel

from statistics import mean 

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default="seqclr/configs/seqclr_model.yaml",
                    required=True, help='path to config file')

# data collator
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor ([`WhisperProcessor`])
            The processor used for processing the data.
        decoder_start_token_id (`int`)
            The begin-of-sentence of the decoder.
    """

    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methodss

        # feature[0]: mel, feature[1]: label, feature[2]: text
        input_features = [{"input_features": feature[0]} for feature in features]
        label_features = [{"input_ids": feature[2]} for feature in features]

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        
        # replace -100 with the pad_token_id
        labels[labels == -100] = processor.tokenizer.pad_token_id

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        
        batch["labels"] = labels

        return batch
    

if __name__ == "__main__":
    # 0. Arguments Setting
    args = parser.parse_args()
    config = Config(args.config)

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


    # 1. Load model
    # (2) DDP model
    processor = WhisperProcessor.from_pretrained("openai/whisper-base", language="English", task="transcribe")
    tokenizer = processor.tokenizer

    # encoder
    base_model = whisper.load_model("base")
    base_encoder = base_model.encoder
    seqclr_model = SeqCLRModel(config)
    seqclr_ckp = torch.load("seqclr/checkpoints/checkpoint-seqclr/seqclr-19.pth.tar")
    seqclr_model.load_state_dict(seqclr_ckp['model_state_dict'])

    # decoder
    from safetensors import safe_open

    tensors = {}
    state_dict_path = 'seqclr/checkpoints/checkpoint-ua-ablation/checkpoint-200/model.safetensors'
    with safe_open(state_dict_path, framework="pt", device=0) as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)

    decoder = base_model.decoder
    decoder.load_state_dict(tensors)
    #decoder.load_state_dict(decoder_ckp['model_state_dict'])

    # Put model on GPU
    base_encoder = base_encoder.to(rank) 
    base_encoder = DDP(base_encoder, device_ids=[local_rank], find_unused_parameters=True)
    seqclr_model = seqclr_model.to(rank) 
    seqclr_model = DDP(seqclr_model, device_ids=[local_rank], find_unused_parameters=True)
    decoder = decoder.to(rank) 
    decoder = DDP(decoder, device_ids=[local_rank], find_unused_parameters=True)

    # 2. Load dataset
    # Initializing a Whisper tiny style configuration
    configuration = WhisperConfig()
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=configuration.decoder_start_token_id,
    )

    ds_eval_VL = UASpeechDataset("eval_VL")
    ds_eval_L = UASpeechDataset("eval_L")
    ds_eval_M = UASpeechDataset("eval_M")
    ds_eval_H = UASpeechDataset("eval_H")

    def dataloader(ds):
        dl = DataLoader(
        dataset=ds,
        batch_size=8//world_size,
        shuffle=True,
        num_workers=mp.cpu_count() // world_size,
        collate_fn=data_collator,
        pin_memory=True,
        drop_last=True,
        )
        return dl

    dl_eval_VL = dataloader(ds_eval_VL)
    dl_eval_L = dataloader(ds_eval_L)
    dl_eval_M = dataloader(ds_eval_M)
    dl_eval_H = dataloader(ds_eval_H)


    # 3. Load Metric
    metric = evaluate.load("wer")

    def compute_metrics(pred):
        logits = pred["logits"]
        pred_ids = torch.argmax(logits, dim=-1)
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        # we do not want to group tokens when computing the metrics
        label_str = tokenizer.batch_decode(pred["label_ids"], skip_special_tokens=True)
        wer = metric.compute(predictions=pred_str, references=label_str)
        return wer
    
    # 4. eval
    def eval_loop(dl):
        wer_ls = []
        for batch in dl:
            x = batch["input_features"]
            y = batch["labels"]

            # Base Encoder
            x = x.cuda(non_blocking=True)
            r = base_encoder(x) #(batch_size, seq_len, emb_size)

            # seqclr(Projection head+Instance Mapping)
            z = seqclr_model(r)
            z = z['instances']
            logits = decoder(y, z).to("cuda")
            y = y.to("cuda")
            pred = {"logits": logits, "label_ids": y}
            wer = compute_metrics(pred)
            wer_ls.append(wer)
        print(wer_ls)
        WER = mean(wer_ls)
        return WER
    
    # wer
    VL_wer = eval_loop(dl_eval_VL)
    L_wer = eval_loop(dl_eval_L)
    M_wer = eval_loop(dl_eval_M)
    H_wer = eval_loop(dl_eval_H)

    print(f"VL_wer: {VL_wer}")
    print(f"L_wer: {L_wer}")
    print(f"M_wer: {M_wer}")
    print(f"H_wer: {H_wer}")

    