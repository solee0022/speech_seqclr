import torch
import torch.nn as nn
import os
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline
import json
import torchaudio
import whisper

import re
import editdistance
from num2words import num2words
from whisper_normalizer.english import EnglishTextNormalizer
import evaluate
from safetensors import safe_open
from peft import PeftModel
import fire

# Inteligibility
VL = ["M04", "F03", "M12", "M01"]
L = ["M07", "F02", "M16"]
M = ["M05", "M11", "F04"]
H = ["M09", "M14", "M10", "M08", "F05"]
speakers = ["M04", "F03", "M12", "M01", "M07", "F02", "M16","M05", "M11", "F04","M09", "M14", "M10", "M08", "F05"]


def load_data_from_json(jsonl_path):
    with open(jsonl_path, "r") as f:
        jsonl_data = list(f)

    data = []
    for pair in jsonl_data:
        sample = json.loads(pair)
        data.append(sample)
        
    return data

def aud_load(x_path):
    aud = whisper.load_audio('/home/solee0022/data/UASpeech/audio/noisereduce/' + x_path)
    return aud
        
def emb(x_path, encoder):
    # mel-spectrogram
    x_mel = aud_load(x_path)
    x_emb = encoder(x_mel).last_hidden_state 
    return x_emb # [1, 1500, 512]

normalizer = EnglishTextNormalizer()
metric = evaluate.load('wer', experiment_id='base')

# inference
def inference(pipe, pred_jsonl_path, speaker):
    jsonl_path = f"../dataset/original/test_{speaker}.jsonl"
    lines = load_data_from_json(jsonl_path)
    
    for line in lines:
        sample = aud_load(line['audio_path'])
        result = pipe(sample)

        prediction = normalizer(result['text'].strip())
        prediction = re.sub(r"[-+]?\d*\.?\d+|\d+%?", lambda m: num2words(m.group()), prediction).replace("%", " percent").upper()
        print(f"prediction: {prediction}")
        
        wer_prediction = metric.compute(
            predictions=[prediction], references=[line['word']]
        )
        line["pred"] = prediction
        line["wer_pred"] = wer_prediction
        
        pred_f = open(pred_jsonl_path, 'a')
        json.dump(line, pred_f, ensure_ascii=False)
        pred_f.write("\n")

def main(
    method = "supervised",
    asr_model = "base",
    ): 
    
    # 1.1. load model
    model = WhisperForConditionalGeneration.from_pretrained(f"openai/whisper-{asr_model}")
    processor = WhisperProcessor.from_pretrained(f"openai/whisper-{asr_model}", language="English", task="transcribe")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # pretrained, FINETUNE(=supervised), de_SEQCLR(=seqclr), de_SEQCLR-HARD(=seqclr-hard)
    pretrained_model_id = 'openai/whisper-{asr_model}'
    ckp_names = {"pretrained": "pretrained", "supervised": "FINETUNE", "seqclr": "de_SEQCLR", "seqclr-hard": "de_SEQCLR-HARD"}
    if method == "pretrained":
        model_id = "openai/whisper-{asr_model}"
    else:
        model_id = f"../checkpoints/{ckp_names[method]}_{asr_model}" # large-v3|base, openai/whisper-base
         
    # load model
    if ('large-v3' in model_id) and ('checkpoint' in model_id):
        model = PeftModel.from_pretrained(WhisperForConditionalGeneration.from_pretrained(pretrained_model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True), model_id)
    else:
        model = WhisperForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
    model.to(device)
    
    pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=1,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
    )
    
    for speaker in ['HEALTHY', 'H', 'M', 'L', 'VL']:
        pred_jsonl_path = f'../dataset/preds/preds_{method}/test_{speaker}_{asr_model}_new.jsonl'
        inference(pipe, pred_jsonl_path, speaker)

if __name__ == "__main__":
    fire.Fire(main)