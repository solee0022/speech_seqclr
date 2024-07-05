from .base import *
import json
from typing import Any, Dict, List, Optional, Union

class UASpeechDataset(BaseDataset):
    def __init__(self, mode) -> None:
        # audio dir:'/home/solee0022/data/' + audio_path
        self.mode = mode
        self.path = "seqclr/dataset/UASpeech"
        self.jsonl_path = f'{self.path}/UASpeech_{self.mode}.jsonl'
        self.load_data_from_json()

        BaseDataset.__init__(self, self.mode)
        index = 0

        for i in self.data:
            y = i["label"] #label
            fn = i["audio"] #audio_path
            txt = i["text"] # text
            segments = i["segments"]

            self.ys += [y]
            self.I += [index]
            self.aud_paths.append(fn)
            self.text.append(txt)
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