from .base import *
import json
from typing import Any, Dict, List, Optional, Union

class UASpeechDataset(BaseDataset):
    def __init__(self, mode, asr_model) -> None:
        # audio dir:'/home/solee0022/data/' + audio_path
        self.mode = mode
        self.asr_model = asr_model
        self.path = "seqclr/dataset/UASpeech"
        self.jsonl_path = f'{self.path}/{self.mode}.jsonl'
        self.load_data_from_json()
        
        ke = open('seqclr/dataset/wordlist.txt', 'r').readlines()
        di = {}
        for a in ke:
            i,s,k = a.strip().split(' ')
            di[k] = int(i)
        self.di = di
        
        BaseDataset.__init__(self, self.mode, self.asr_model)
        index = 0

        for i in self.data:
            y = self.di[i["word_id"]] #label
            fn = i["audio_path"] #audio_path
            txt = i["word"] # text
            segments = i["segments"]

            self.ys += [y]
            self.I += [index]
            self.aud_paths.append(fn)
            self.texts.append(txt)
            self.SEGMENTS.append(segments)
            index += 1

    def load_data_from_json(self) -> List:
        with open(self.jsonl_path, "r") as f:
            jsonl_data = list(f)

        data = []
        for pair in jsonl_data:
            sample = json.loads(pair)
            data.append(sample)

        # if self.max_sample is not None:
        #     data = data[: self.max_sample]
        self.data = data




#train_ds = UASpeechDataset("train")