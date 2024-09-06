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

from transformers import (WhisperConfig, WhisperProcessor, WhisperForConditionalGeneration, EarlyStoppingCallback, set_seed)
from transformers import (Seq2SeqTrainingArguments, Seq2SeqTrainer)
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

        # feature['input_features']: mel, feature['label']: label, feature['text']: text
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [{"input_ids": feature["text"]} for feature in features]

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        
        # replace -100 with the pad_token_id
        labels[labels == -100] = self.processor.tokenizer.pad_token_id

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        
        return batch # input_features, labels


def main():
    # 0. Arguments Setting
    args = parser.parse_args()
    config = Config(args.config)
    os.environ["WANDB_PROJECT"] = config.wandb_project_name
    
    # 0. outdir
    try:
        os.makedirs(config.decoder_outdir)
    except FileExistsError:
        pass
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=config.decoder_outdir,
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
        predict_with_generate=True,
        # generation_max_length=200,
        # generation_num_beams=5,
        data_seed=1234,
        do_train=True,
        metric_for_best_model="eval_wer", 
        load_best_model_at_end=True,
        report_to=[config.wandb_report_to],
        run_name=config.wandb_run_name,
    )

    # 1.1. Load model
    processor = WhisperProcessor.from_pretrained(f"openai/whisper-{config.model_speech_backbone}", language="English", task="transcribe")
    tokenizer = processor.tokenizer
    base_config = WhisperConfig.from_pretrained(f"openai/whisper-{config.model_speech_backbone}")
    model = WhisperForConditionalGeneration.from_pretrained(f"openai/whisper-{config.model_speech_backbone}")
    
    if not config.decoder_seqclr_ckp:
        logger.info("*** Use Non-Contrastive Encoder ***")
        pass
    else:
        logger.info("*** Use Contrastive Encoder ***")
        # 1.2. load state_dict of seqclr(encoder)
        seqclr_state_dict = safe_open(f'{config.decoder_seqclr_ckp}/model.safetensors', framework='pt')  #['model_state_dict']

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
    if config.decoder_seqclr_freeze:
        model.freeze_encoder()
        model.model.encoder.gradient_checkpointing = False

    if config.model_speech_backbone == 'large-v3':
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
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=base_config.decoder_start_token_id,
    )

    if config.decoder_dataset_type == 'ua':
        # trainset
        ds_train = UASpeechDataset(config.decoder_dataset_mode_ua_train_mode[0], config.model_speech_backbone)
        # evalset
        # run separate evaluations on each dataset
        if len(config.decoder_dataset_mode_ua_test_mode)>1:
            ds_eval = {}
            for eval_mode in config.decoder_dataset_mode_ua_test_mode:
                key = eval_mode
                value = UASpeechDataset(eval_mode, config.model_speech_backbone)
                ds_eval[key] = value
        else:
            ds_eval = UASpeechDataset(config.decoder_dataset_mode_ua_test_mode[0], config.model_speech_backbone)


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
    trainer = Seq2SeqTrainer(
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