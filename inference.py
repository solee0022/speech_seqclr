import torch
import torch.nn as nn
import os
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, pipeline
import json
import torchaudio
import soundfile as sf
import numpy as np

import re
import editdistance
from num2words import num2words
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

metric = evaluate.load('wer', experiment_id='base')

# inference
def inference(model, processor, pred_jsonl_path, speaker):
    jsonl_path = f"../dataset/original/test_{speaker}.jsonl"
    lines = load_data_from_json(jsonl_path)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    for line in lines:
        path = "/home/solee0022/data/UASpeech/audio/noisereduce/"
        waveform, sample_rate = sf.read(path + line['audio_path'])
        input_values = processor(waveform, sampling_rate=sample_rate, return_tensors="pt").input_values.to(device)
        
        # inference
        # retrieve logits & take argmax
        with torch.no_grad():
            outputs = model(input_values)
            logits = outputs.logits
        predicted_ids = torch.argmax(logits, dim=-1)
        # print(f'outputs: {outputs.hidden_states[-1].size()}')
        
        # transcribes
        transcription = processor.batch_decode(predicted_ids)

        prediction = transcription[0].strip()
        prediction = re.sub(r"[-+]?\d*\.?\d+|\d+%?", lambda m: num2words(m.group()), prediction).replace("%", " percent").upper()
        
        wer_prediction = metric.compute(
            predictions=[prediction], references=[line['word']]
        )
        line["pred"] = prediction
        line["wer_pred"] = wer_prediction
        
        pred_f = open(pred_jsonl_path, 'a')
        json.dump(line, pred_f, ensure_ascii=False)
        pred_f.write("\n")


def main(
    method = "pretrained",
    asr_model = "base",
    ): 
    # ckp_names
    ckp_names = {"pretrained": "pretrained", "supervised": "FINETUNE", "seqclr": "de_SEQCLR", "seqclr-hard": "de_SEQCLR-HARD"}
    if method == "pretrained":
        model_id = f"facebook/wav2vec2-{asr_model}-960h"
    else:
        model_id = f"../checkpoints/{ckp_names[method]}_{asr_model}-960h" # large|base, facebook/wav2vec2-{asr_model}-960h
         
    # load model
    model = Wav2Vec2ForCTC.from_pretrained(model_id, output_hidden_states=True)
    processor = Wav2Vec2Processor.from_pretrained(f"facebook/wav2vec2-{asr_model}-960h")
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float32 # if torch.cuda.is_available() else torch.float32
    
    model.to(device)
    print(model)
    
    for speaker in ['HEALTHY', 'H', 'M', 'L', 'VL']:
        pred_jsonl_path = f'../dataset/preds_w2v/preds_{method}/test_{speaker}_{asr_model}.jsonl'
        inference(model, processor, pred_jsonl_path, speaker)

if __name__ == "__main__":
    fire.Fire(main)