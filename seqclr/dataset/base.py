import os
import torch
import numpy as np
import whisper
import soundfile as sf
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, mode, asr_model):
        self.mode = mode
        self.asr_model = asr_model
        self.ys, self.aud_paths, self.I, self.SEGMENTS, self.texts = [], [], [], [], []
        self.model = Wav2Vec2ForCTC.from_pretrained(f"facebook/wav2vec2-{asr_model}-960h")
        self.processor = Wav2Vec2Processor.from_pretrained(f"facebook/wav2vec2-{asr_model}-960h")
        self.SAMPLE_RATE = 16000
        self.N_FFT = 400
        self.HOP_LENGTH = 160
        self.CHUNK_LENGTH = 30
        self.N_SAMPLES = self.CHUNK_LENGTH * self.SAMPLE_RATE
        
    def nb_classes(self):
        assert set(self.ys) == set(self.classes)
        return len(self.classes)

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, index):

        waveform, sample_rate = sf.read('/home/solee0022/data/UASpeech/audio/noisereduce/' + self.aud_paths[index])
        input_values = self.processor(waveform, sampling_rate=sample_rate, return_tensors="pt").input_values[0]
        label = self.ys[index]

        input_ids = self.processor.tokenizer(self.texts[index], truncation=True).input_ids
            
        sample = {"input_values": input_values, "label": label, 'segments': np.array(self.SEGMENTS[index], dtype=np.float16), 'input_ids': input_ids}
        return sample

    def get_label(self, index):
        return self.ys[index]

    def set_subset(self, I):
        self.ys = [self.ys[i] for i in I]
        self.I = [self.I[i] for i in I]
        self.aud_paths = [self.aud_paths[i] for i in I]
        self.text = [self.text[i] for i in I]
        self.SEGMENTS= [self.SEGMENTS[i] for i in I]

