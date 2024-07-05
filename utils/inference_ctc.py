import torch
import os
from seqclr.dataset.ua import UASpeechDataset
from seqclr.dataset.torgo import TORGODataset
from transformers import AutoProcessor, AutoModelForCTC, AutoTokenizer, AutoFeatureExtractor, Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2Model
from torch.utils.data import DataLoader
import IPython
import torchaudio
import matplotlib.pyplot as plt
import torchaudio.functional as F
import json

## 0.0. load dataset
# ds_ua = UASpeechDataset("train_seqclr") # UASpeechDataset("train", cfg) -> cfg.train_roots

def load_data_from_json(jsonl_path):
    with open(jsonl_path, "r") as f:
        jsonl_data = list(f)

    data = []
    for pair in jsonl_data:
        sample = json.loads(pair)
        data.append(sample)
    return data
 
 
def transcribe_wav2vec(audio_path): 
    ## Wav2Vec2.0 model
    processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h") # "facebook/wav2vec2-base-960h"
    model = AutoModelForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    model_enc = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
    # time_tokenizer = AutoTokenizer.from_pretrained("patrickvonplaten/wav2vec2-base-960h-time-stamps", trust_remote_code=True, return_length=True)

    pho_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")
    pho_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")

    # pho_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")
    # pho_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")
    
    # load audio
    audio, sample_rate = torchaudio.load("/home/solee0022/data/" + audio_path)
    input_features = processor(audio, sampling_rate=sample_rate, return_tensors="pt").input_values[0]
    with torch.no_grad():
        emb = model_enc(input_features).last_hidden_state
        outputs = model(input_features)
    logits = outputs.logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    # convert ids to tokens
    tokens = " ".join(processor.tokenizer.convert_ids_to_tokens(predicted_ids[0].tolist())).replace("<pad>", "_")

    # phoneme
    pho_input_features = pho_processor(audio, sampling_rate=sample_rate, return_tensors="pt").input_values[0]
    with torch.no_grad():
        pho_outputs = pho_model(pho_input_features)
    pho_logits = pho_outputs.logits
    predicted_ids = torch.argmax(pho_logits, dim=-1)
    pho_transcription = pho_processor.batch_decode(predicted_ids)[0]
    # convert ids to tokens
    pho_tokens = " ".join(pho_processor.tokenizer.convert_ids_to_tokens(predicted_ids[0].tolist())).replace("<pad>", "_")

    print(f"input_features.shape: {input_features.shape}")
    print(f"emb.shape: {emb.shape}")
    print(f"transcription: {transcription}")
    print(f"tokens: {tokens}")
    print(f"pho_transcription: {pho_transcription}")
    print(f"pho_tokens: {pho_tokens}")
    print("####################################################")
    # transcription_with_timestamps = tokenizer.batch_decode(predicted_ids, output_time_stamps=True, stride=320, sampling_rate=16000)
    # transcription_with_timestamps = tokenizer.batch_decode(predicted_ids, group_tokens=False, clean_up_tokenization_spaces=False, output_char_offsets=True)