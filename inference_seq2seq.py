import torch
import torch.nn as nn
import os
from transformers import AutoProcessor, AutoModelForCTC, AutoTokenizer, AutoFeatureExtractor, AutoModelForSpeechSeq2Seq, Wav2Vec2Processor, Wav2Vec2ForCTC, WhisperProcessor, WhisperForConditionalGeneration
from torch.utils.data import DataLoader
import whisper_timestamped as whisper
import json
import torchaudio
import whisper
from sklearn.manifold import TSNE

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

import editdistance
from seqclr.modules.normalizers import EnglishTextNormalizer


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
    aud = whisper.load_audio('/home/solee0022/data/' + x_path)
    aud = whisper.pad_or_trim(aud) # length=self.N_SAMPLES
    mel = whisper.log_mel_spectrogram(aud, n_mels=80) # whisper-base, small, medium=80, large=128
    return mel.unsqueeze(0)
        
def emb(x_path, encoder):
    # mel-spectrogram
    x_mel = aud_load(x_path)
    x_emb = encoder(x_mel).last_hidden_state 
    return x_emb # [1, 1500, 512]

jsonl_path = "/home/solee0022/seqclr_exp/speech_seqclr/seqclr/dataset/ua/original/UASpeech_test.jsonl"
lines = load_data_from_json(jsonl_path)

normalizer = EnglishTextNormalizer()
def calculate_wer(pre, ref):
    return editdistance.eval(pre, ref) / len(ref)

# word
def inference(processor, model, f):
    for line in lines:
        speech, sample_rate = torchaudio.load('/home/solee0022/data/'+line['audio'])
        input_features = processor(input_features, sampling_rate=sample_rate, return_tensors="pt").input_features
        with torch.no_grad():
            predicted_ids = model.generate(input_features.to("cuda"))[0]
        transcription = processor.decode(predicted_ids)
        prediction = processor.tokenizer._normalize(transcription)
        prediction = normalizer(prediction)
        cur_wer = calculate_wer(prediction.split(), line['text'].split())
        speaker = line['audio'].split("/")[3]

        new_line = {'audio': line['audio'], 'wer': cur_wer, 'text': line['text'], 'pred': prediction}            
        json.dump(new_line, f, ensure_ascii=False)
        f.write("\n")

    
def example(rank, world_size): 
    method = "pretrained"
    # create default process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    # load model
    processor = AutoProcessor.from_pretrained("openai/whisper-base").to('cpu')  
    if method == "pretrained":  
        model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-base").to(rank)
        f = open('/home/solee0022/seqclr_exp/speech_seqclr/seqclr/dataset/ua/preds_pre/UASpeech_test.jsonl', 'w')

    elif method == 'supervised':
        model = torch.load('/home/solee0022/seqclr_exp/checkpoints/supervised').to(rank)
        f = open('/home/solee0022/seqclr_exp/speech_seqclr/seqclr/dataset/ua/preds_sup/UASpeech_test.jsonl', 'w')
    elif method == 'syll':
        model = torch.load('/home/solee0022/seqclr_exp/checkpoints/run2-UA-window-syllable').to(rank)
        f = open('/home/solee0022/seqclr_exp/speech_seqclr/seqclr/dataset/ua/preds_syll/UASpeech_test.jsonl', 'w')
    elif method == 'char':
        model = torch.load('/home/solee0022/seqclr_exp/checkpoints/run2-UA-window-character').to(rank)
        f = open('/home/solee0022/seqclr_exp/speech_seqclr/seqclr/dataset/ua/preds_char/UASpeech_test.jsonl', 'w')
    elif method == 'dtw':
        model = torch.load('/home/solee0022/seqclr_exp/checkpoints/run2-UA-window-dtw').to(rank)
        f = open('/home/solee0022/seqclr_exp/speech_seqclr/seqclr/dataset/ua/preds_dtw/UASpeech_test.jsonl', 'w')

    print(model)
    # construct DDP model
    ddp_model = DDP(model, device_ids=[rank])
    inference(processor, ddp_model, f)

def main():
    world_size = 1
    mp.spawn(example,
        args=(world_size,),
        nprocs=world_size,
        join=True)
    
# make embedding
if __name__=="__main__":
    # Environment variables which need to be
    # set when using c10d's default "env"
    # initialization mode.
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    main()

