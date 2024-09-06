import torch
import torch.nn as nn
import os
from transformers import AutoProcessor,AutoModelForSpeechSeq2Seq, WhisperForConditionalGeneration
import json
import torchaudio
import whisper

import torch
import editdistance
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

def aud_load(x_path):
    aud = whisper.load_audio('/home/solee0022/data/UASpeech/audio/noisereduce/' + x_path)
    aud = whisper.pad_or_trim(aud) # length=self.N_SAMPLES
    mel = whisper.log_mel_spectrogram(aud, n_mels=80) # whisper-base, small, medium=80, large=128
    return mel.unsqueeze(0)
        
def emb(x_path, encoder):
    # mel-spectrogram
    x_mel = aud_load(x_path)
    x_emb = encoder(x_mel).last_hidden_state 
    return x_emb # [1, 1500, 512]

normalizer = EnglishTextNormalizer()
metric = evaluate.load('wer')

# word
def inference(processor, model, f, speaker):
    jsonl_path = f"../dataset/original/test_{speaker}.jsonl"
    lines = load_data_from_json(jsonl_path)
    
    for line in lines:
        input_features = aud_load(line['audio_path'])
        with torch.no_grad():
            predicted_ids = model.generate(input_features)[0]
        transcription = processor.decode(predicted_ids)
        prediction = processor.tokenizer._normalize(transcription)
        prediction = normalizer(prediction).upper()
        wer_pred = metric.compute([prediction], [line['word']])

        line["pred"] = prediction
        line["wer_pred"] = wer_pred
        
        json.dump(line, f, ensure_ascii=False)
        f.write("\n")

    
def main(
    method = "seqclr",
    asr_model = "base",
    ): 
    
    # 1.1. load model
    model = WhisperForConditionalGeneration.from_pretrained(f"openai/whisper-{asr_model}")
    processor = AutoProcessor.from_pretrained(f"openai/whisper-{asr_model}")
    if method == "pretrained":  
        model = AutoModelForSpeechSeq2Seq.from_pretrained(f"openai/whisper-{asr_model}")

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
        f = open(f'../dataset/preds/preds_{method}/test_{speaker}_{asr_model}.jsonl', 'w')
        inference(processor, model, f, speaker)

main(method = "seqclr",
    asr_model = "base",
    )