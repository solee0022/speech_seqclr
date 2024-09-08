import torch
import torch.nn as nn
import os
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import json
import torchaudio
import whisper

import re
import editdistance
from num2words import num2words
from whisper_normalizer.english import EnglishTextNormalizer
import evaluate
from safetensors import safe_open
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

def aud_load(x_path, asr_model):
    aud = whisper.load_audio('/home/solee0022/data/UASpeech/audio/noisereduce/' + x_path)
    aud = whisper.pad_or_trim(aud) # length=self.N_SAMPLES
    if 'large' in asr_model:
        n_mels = 128
    else:
        n_mels = 80
    mel = whisper.log_mel_spectrogram(aud, n_mels=n_mels) # whisper-base, small, medium=80, large=128
    return mel.unsqueeze(0)
        
def emb(x_path, encoder):
    # mel-spectrogram
    x_mel = aud_load(x_path)
    x_emb = encoder(x_mel).last_hidden_state 
    return x_emb # [1, 1500, 512]

normalizer = EnglishTextNormalizer()
metric = evaluate.load('wer', experiment_id='large-v3')

# inference
def inference(processor, model, asr_model, pred_jsonl_path, speaker):
    jsonl_path = f"../dataset/original/test_{speaker}.jsonl"
    lines = load_data_from_json(jsonl_path)
    
    for line in lines:
        input_features = aud_load(line['audio_path'], asr_model)

        with torch.no_grad():
            predicted_ids = model.generate(input_features)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        prediction = normalizer(transcription)
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
    method = "seqclr",
    asr_model = "large-v3",
    ): 
    
    # 1.1. load model
    model = WhisperForConditionalGeneration.from_pretrained(f"openai/whisper-{asr_model}")
    processor = WhisperProcessor.from_pretrained(f"openai/whisper-{asr_model}", language="English", task="transcribe")
    if method == "pretrained":  
        model = WhisperForConditionalGeneration.from_pretrained(f"openai/whisper-{asr_model}")

    elif method == 'supervised':
        model_dict = safe_open('../checkpoints/FINETUNE_{asr_model}/model.safetensors', framework='pt')
        
    elif method == 'seqclr':
        model_dict = safe_open(f'../checkpoints/de_SEQCLR_{asr_model}/model.safetensors', framework='pt')
        
    elif method == 'seqclr-hard':
        model_dict = safe_open('../checkpoints/de_SEQCLR-HARD_{asr_model}/model.safetensors', framework='pt')
        
    # 1.2. seq2seq state_dict
    current_state_dict = model.state_dict()

    # 1.3. apply loaded state_dict to current_state_dict(correspond to seqclr)
    for key in model_dict.keys():
        if key not in current_state_dict.keys():
            pass
        else:
            current_state_dict[key] = model_dict.get_tensor(key)

    # 1.4. apply to current model
    model.load_state_dict(current_state_dict)
        
    for speaker in ['HEALTHY', 'H', 'M', 'L', 'VL']:
        pred_jsonl_path = f'../dataset/preds/preds_{method}/test_{speaker}_{asr_model}.jsonl'
        inference(processor, model, asr_model, pred_jsonl_path, speaker)


if __name__ == "__main__":
    fire.Fire(main)