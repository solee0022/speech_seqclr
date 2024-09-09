import os
import torch
import torch.optim as optim
import torch.nn.functional as F

import argparse
import wandb

import logging
from typing import Any, Dict, List, Optional, Union

from seqclr.dataset.ua import UASpeechDataset
from seqclr.modules.model_seqclr import SeqCLRModel

from transformers import (Wav2Vec2Config, Wav2Vec2ForCTC, Wav2Vec2Processor, EarlyStoppingCallback, set_seed)
from transformers import (TrainingArguments, Trainer)
from transformers.trainer_utils import get_last_checkpoint
from seqclr.modules.utils import Config
import evaluate
from dataclasses import dataclass
from safetensors import safe_open

from peft import prepare_model_for_kbit_training
from peft import LoraConfig, PeftModel, LoraModel, LoraConfig, get_peft_model

logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default="seqclr/configs/seqclr_model.yaml",
                    required=True, help='path to config file')

@dataclass
class DataCollatorCTCWithPadding:
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
        label_features = [{"input_ids": feature["input_ids"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        labels_batch = self.processor.pad(
            labels=label_features,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of_labels,
            return_tensors="pt",
        )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels
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
        os.makedirs(config.asr_outdir)
    except FileExistsError:
        pass
    
    training_args = TrainingArguments(
        output_dir=config.asr_outdir,
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
        metric_for_best_model="eval_wer", 
        load_best_model_at_end=True,
        report_to=[config.wandb_report_to],
        run_name=config.wandb_run_name,
    )

    # 1.1. Load model
    processor = Wav2Vec2Processor.from_pretrained(f"facebook/wav2vec2-{config.seqclr_speech_backbone}-960h")
    tokenizer = processor.tokenizer
    model = Wav2Vec2ForCTC.from_pretrained(f"facebook/wav2vec2-{config.seqclr_speech_backbone}-960h")
    
    if not config.asr_seqclr_ckp:
        logger.info("*** Use Non-Contrastive Encoder ***")
        pass
    else:
        logger.info("*** Use Contrastive Encoder ***")
        # 1.2. load state_dict of seqclr(encoder)
        seqclr_state_dict = safe_open(f'{config.asr_seqclr_ckp}/model.safetensors', framework='pt')  #['model_state_dict']

        # 1.3. seq2seq state_dict
        current_state_dict = model.state_dict()

        # 1.4. apply loaded state_dict to current_state_dict(correspond to seqclr)
        for key in seqclr_state_dict.keys():
            if key not in current_state_dict.keys():
                pass
            else:
                current_state_dict[key] = seqclr_state_dict.get_tensor(key)

        # 1.5. apply to current model
        model.load_state_dict(current_state_dict)
    
    # 1.6. freeze model
    # freeze encoder
    if config.asr_freeze_feature_encoder:
        model.freeze_feature_encoder()

    if config.seqclr_speech_backbone == 'large-v3':
        model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none")

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    # 2. Detecting last checkpoint and eventually continue from last checkpoint
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
            
    # Set seed before initializing model.
    set_seed(training_args.seed)

    # early stopping
    early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=5)
    
    # 3. Load dataset
    data_collator = DataCollatorCTCWithPadding(processor=processor)

    if config.asr_dataset_type == 'ua':
        # trainset
        ds_train = UASpeechDataset(config.asr_dataset_mode_ua_train_mode[0], config.seqclr_speech_backbone)
        # evalset
        # run separate evaluations on each dataset
        if len(config.asr_dataset_mode_ua_test_mode)>1:
            ds_eval = {}
            for eval_mode in config.asr_dataset_mode_ua_test_mode:
                key = eval_mode
                value = UASpeechDataset(eval_mode, config.seqclr_speech_backbone)
                ds_eval[key] = value
        else:
            ds_eval = UASpeechDataset(config.asr_dataset_mode_ua_test_mode[0], config.seqclr_speech_backbone)


    # 4. Load Metric
    metric = evaluate.load("wer", experiment_id='loss') 
    
    def compute_metrics(pred):
        pred_ids = pred.predictions

        pred.label_ids[pred.label_ids == -100] = tokenizer.pad_token_id

        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        # we do not want to group tokens when computing the metrics
        label_str = tokenizer.batch_decode(pred.label_ids, skip_special_tokens=True)

        wer = metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}


    # 5. Train!
    logger.info(f"*** Train stage: {config.global_stage} ***")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_eval,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping_callback],
    )

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
        
    metrics = train_result.metrics
    trainer.save_model()
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


    # 6. Evaluation!
    logger.info("*** Evaluate ***")
    metrics = trainer.evaluate(
        metric_key_prefix="eval",
        max_length=training_args.generation_max_length,
        num_beams=training_args.generation_num_beams,
    )
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    
    
if __name__ == "__main__":
    main()