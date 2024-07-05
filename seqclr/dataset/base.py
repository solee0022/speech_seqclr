import os
import torch
import numpy as np
import whisper
import torchaudio
from transformers import WhisperTokenizer 
from transformers import AutoProcessor


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, mode):
        self.mode = mode
        self.ys, self.aud_paths, self.I, self.text, self.SEGMENTS = [], [], [], [], []
        self.model = whisper.load_model("base")
        self.processor = AutoProcessor.from_pretrained("openai/whisper-base", language="English")
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
        def aud_load(index):
            aud = whisper.load_audio('/home/solee0022/data/' + self.aud_paths[index])
            aud = whisper.pad_or_trim(aud) # length=self.N_SAMPLES
            mel = whisper.log_mel_spectrogram(aud, n_mels=80) # whisper-base, small, medium=80, large=128
            return mel

        mel = aud_load(index)
        label = self.ys[index]
        text = self.processor.tokenizer(self.text[index], truncation=True).input_ids
            
        sample = {"input_features": mel, "label": label, "text": text, 'segments': self.SEGMENTS[index]}
        return sample

    def get_label(self, index):
        return self.ys[index]

    def set_subset(self, I):
        self.ys = [self.ys[i] for i in I]
        self.I = [self.I[i] for i in I]
        self.aud_paths = [self.aud_paths[i] for i in I]
        self.text = [self.text[i] for i in I]
        self.SEGMENTS= [self.SEGMENTS[i] for i in I]

