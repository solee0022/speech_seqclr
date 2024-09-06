import os
import torch
import numpy as np
import whisper
import torchaudio
from transformers import WhisperTokenizer 
from transformers import AutoProcessor


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, mode, asr_model):
        self.mode = mode
        self.asr_model = asr_model
        self.ys, self.aud_paths, self.I, self.SEGMENTS, self.texts = [], [], [], [], []
        self.model = whisper.load_model(asr_model)
        self.processor = AutoProcessor.from_pretrained(f"openai/whisper-{asr_model}", language="English")
        self.SAMPLE_RATE = 16000
        self.N_FFT = 400
        self.HOP_LENGTH = 160
        self.CHUNK_LENGTH = 30
        self.N_SAMPLES = self.CHUNK_LENGTH * self.SAMPLE_RATE
        if self.asr_model == 'large-v3':
            self.n_mels = 128
        else:
            self.n_mels = 80
        
    def nb_classes(self):
        assert set(self.ys) == set(self.classes)
        return len(self.classes)

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, index):
        def aud_load(index):
            aud = whisper.load_audio('/home/solee0022/data/UASpeech/audio/noisereduce/' + self.aud_paths[index])
            aud = whisper.pad_or_trim(aud) # length=self.N_SAMPLES
            mel = whisper.log_mel_spectrogram(aud, n_mels=self.n_mels) # whisper-base, small, medium=80, large=128
            return mel

        mel = aud_load(index)
        label = self.ys[index]

        text = self.processor.tokenizer(self.texts[index], truncation=True).input_ids
            
        sample = {"input_features": mel, "label": label, 'segments': np.array(self.SEGMENTS[index], dtype=np.float16), 'text': text}
        return sample

    def get_label(self, index):
        return self.ys[index]

    def set_subset(self, I):
        self.ys = [self.ys[i] for i in I]
        self.I = [self.I[i] for i in I]
        self.aud_paths = [self.aud_paths[i] for i in I]
        # self.text = [self.text[i] for i in I]
        self.SEGMENTS= [self.SEGMENTS[i] for i in I]

