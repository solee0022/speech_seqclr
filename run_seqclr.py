import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import argparse
import wandb
from torch import nn

import logging
from typing import Any, Dict, List, Optional, Union

from seqclr.dataset.ua import UASpeechDataset
from seqclr.modules.model_seqclr import SeqCLRModel
from seqclr.modules.dtw import window_mapping_and_character
from seqclr.losses.seqclr_loss import SeqCLRLoss
from seqclr.dataset.sampler import UniqueClassSampler

from transformers import (WhisperConfig, WhisperProcessor, WhisperForConditionalGeneration, EarlyStoppingCallback, set_seed)
from transformers import (TrainingArguments, Trainer) 
from transformers.trainer_utils import get_last_checkpoint
from seqclr.modules.utils import Config
from torch.utils.data import DataLoader

from peft import prepare_model_for_kbit_training
from peft import LoraConfig, PeftModel, LoraModel, LoraConfig, get_peft_model

logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default="seqclr/configs/seqclr_model.yaml",
                    required=True, help='path to config file')

class CustomTrainer(Trainer):
    def my_collate_fn(self, samples):
        collate_X = [sample['input_features'] for sample in samples]
        collate_X = torch.stack(collate_X, dim=0)
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
        return {'input_features': collate_X,
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
        x = inputs["input_features"]
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
        

def main():
    # 0. Arguments Setting
    args = parser.parse_args()
    config = Config(args.config)
    os.environ["WANDB_PROJECT"] = config.wandb_project_name
    
    # 0. outdir
    try:
        os.makedirs(config.model_outdir)
    except FileExistsError:
        pass
    
    training_args = TrainingArguments(
        output_dir=config.model_outdir,
        num_train_epochs=config.training_epochs,
        per_device_train_batch_size=config.training_train_bs,
        per_device_eval_batch_size=config.training_eval_bs,
        evaluation_strategy="steps",
        save_steps=config.training_save_iters,
        eval_steps=config.training_eval_iters,
        logging_steps=config.training_logg_iters,
        save_total_limit=config.training_save_total_limit,
        learning_rate=config.optimizer_lr,
        weight_decay=config.optimizer_wd,
        #lr_scheduler_type=config.optimizer_lr_scheduler_type,
        warmup_steps=config.training_warmup_steps, 
        remove_unused_columns=False,
        data_seed=1234,
        do_train=True,
        load_best_model_at_end=True,
        report_to=[config.wandb_report_to],
        run_name=config.wandb_run_name,
    )

    # 0. Detecting last checkpoint and eventually continue from last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
            
    # 1. Load model
    processor = WhisperProcessor.from_pretrained(f"openai/whisper-{config.model_speech_backbone}", language="English", task="transcribe")
    seqclr_model = SeqCLRModel(config)
    # lora 
    if config.model_speech_backbone == 'large-v3':
        seqclr_model.encoder = prepare_model_for_kbit_training(seqclr_model.encoder)
        lora_config = LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none")

        seqclr_model.encoder = get_peft_model(seqclr_model.encoder, lora_config)
        seqclr_model.encoder.print_trainable_parameters()
            
    # Set seed before initializing model.
    set_seed(training_args.seed)

    # 2. early stopping
    early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=20)
    
    # 3. Load dataset
    ds_train = UASpeechDataset(config.model_dataset_train_mode, config.model_speech_backbone) # UASpeechDataset("train", cfg) -> cfg.train_roots
    ds_test = UASpeechDataset(config.model_dataset_test_mode, config.model_speech_backbone) 

    # 4. Train!
    logger.info(f"*** Train stage: {config.global_stage} ***")
    trainer = CustomTrainer(
        model=seqclr_model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_test,
        # callbacks=[early_stopping_callback],
    )

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
        
    trainer.save_model()
    trainer.save_state()


    # 6. Evaluation!
    logger.info("*** Evaluate ***")
    metrics = trainer.evaluate(
        metric_key_prefix="eval",
    )
    
if __name__ == "__main__":
    main()