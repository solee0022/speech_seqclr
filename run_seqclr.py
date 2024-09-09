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

from seqclr.modules.model_seqclr import SeqCLRModel
from seqclr.modules.dtw import window_mapping_and_character
from seqclr.modules.custom_trainer import CustomTrainer
from seqclr.dataset.ua import UASpeechDataset

from transformers import (Wav2Vec2ForCTC, Wav2Vec2Processor, EarlyStoppingCallback, set_seed)
from transformers import (TrainingArguments, Trainer) 
from transformers.trainer_utils import get_last_checkpoint
from seqclr.modules.utils import Config
from torch.utils.data import DataLoader
from dataclasses import dataclass

from peft import prepare_model_for_kbit_training
from peft import LoraConfig, PeftModel, LoraModel, LoraConfig, get_peft_model

logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default="seqclr/configs/seqclr_model.yaml",
                    required=True, help='path to config file')

@dataclass
class DataCollatorWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.AutoProcessor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

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
    
def main():
    # 0. Arguments Setting
    args = parser.parse_args()
    config = Config(args.config)
    os.environ["WANDB_PROJECT"] = config.wandb_project_name
    
    # 0. outdir
    try:
        os.makedirs(config.seqclr_outdir)
    except FileExistsError:
        pass
    
    training_args = TrainingArguments(
        output_dir=config.seqclr_outdir,
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
    processor = Wav2Vec2Processor.from_pretrained(f"facebook/wav2vec2-{config.seqclr_speech_backbone}-960h")
    seqclr_model = SeqCLRModel(config)
    # lora 
    if config.seqclr_speech_backbone == 'large-v3':
        seqclr_model.encoder = prepare_model_for_kbit_training(seqclr_model.encoder)
        lora_config = LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none")

        seqclr_model.encoder = get_peft_model(seqclr_model.encoder, lora_config)
        seqclr_model.encoder.print_trainable_parameters()
            
    # Set seed before initializing model.
    set_seed(training_args.seed)

    # 2. early stopping
    early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=20)
    
    # 3. Load dataset
    ds_train = UASpeechDataset(config.seqclr_dataset_train_mode, config.seqclr_speech_backbone) # UASpeechDataset("train", cfg) -> cfg.train_roots
    ds_test = UASpeechDataset(config.seqclr_dataset_test_mode, config.seqclr_speech_backbone) 
    data_collator = DataCollatorWithPadding(processor=processor)
    
    # 4. Train!
    logger.info(f"*** Train stage: {config.global_stage} ***")
    trainer = CustomTrainer(
        model=seqclr_model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_test,
        data_collator=data_collator,
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