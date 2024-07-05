import torch
import torch.nn as nn
import os
from transformers import AutoProcessor, AutoModelForCTC, AutoTokenizer, AutoFeatureExtractor, AutoModelForSpeechSeq2Seq, Wav2Vec2Processor, Wav2Vec2ForCTC
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


model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-base").to("cpu")
processor = AutoProcessor.from_pretrained("openai/whisper-base")

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

jsonl_path = "/home/solee0022/seqclr_exp/speech_seqclr/seqclr/dataset/UASpeech/UASpeech_scripts_train_UW_alignment.jsonl"
lines = load_data_from_json(jsonl_path)
pool = nn.AdaptiveAvgPool2d((1, 512))
words = ["juliet", "five", "quebec", "she", "control", "not", "him", "call", "ballons", "make"]

# word
def word_emb(method, encoder):
    word_f = open(f"/home/solee0022/seqclr_exp/speech_seqclr/seqclr/dataset/UASpeech/emb/UASpeech_word_{method}.jsonl", "w")
    
    for line in lines:
        if line['text'] in words:
            speaker = line['audio'].split("/")[3]
            if speaker in VL:
                group = 'VL'
            elif speaker in L:
                group = 'L'
            elif speaker in M:
                group = 'M'
            elif speaker in H:
                group = 'H'
            else:
                group = 'Healthy'
            x_emb = emb(line['audio'], encoder)
            x_pooled = pool(x_emb)[0][0].tolist()
            word_line = {'emb': x_pooled, 'text': line['text'], 'speaker': speaker, 'group': group, 'segments': line['segments']}
            json.dump(word_line, word_f, ensure_ascii=False)
            word_f.write("\n")


def char_emb(method, encoder):
    char_f = open(f"/home/solee0022/seqclr_exp/speech_seqclr/seqclr/dataset/UASpeech/emb/UASpeech_char_{method}.jsonl", "w")
    
    for line in lines:
        if line['text'] in words:
            speaker = line['audio'].split("/")[3]
            if speaker in VL:
                group = 'VL'
            elif speaker in L:
                group = 'L'
            elif speaker in M:
                group = 'M'
            elif speaker in H:
                group = 'H'
            else:
                group = 'Healthy'
            x_emb = emb(line['audio'], encoder)
            segments = line['segments']
            text = line['text']
            for seg_idx, seg in enumerate(segments[1:-1]):
                seg_emb = x_emb[0][seg[0]:seg[1]]
                seg_pooled = pool(seg_emb.unsqueeze(0))[0][0].tolist()
                char = text[seg_idx]
                char_line = {'emb': seg_pooled, 'char': char, 'speaker': speaker, 'group': group, 'text': text}
                json.dump(char_line, char_f, ensure_ascii=False)
                char_f.write("\n")


def example(rank, world_size): 
    method = "pretrained"
    # create default process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    # load model
    if method == "pretrained":
        encoder = model.model.encoder.to(rank)
    elif method == 'supervised':
        state_dict = torch.load("/home/solee0022/seqclr_exp/checkpoints/UA-supervised/checkpoint-6000").state_dict()
        
        # 1.3. seq2seq state_dict
        current_state_dict = model.state_dict()

        # 1.4. apply loaded state_dict to current_state_dict(correspond to seqclr)
        for key, value in seqclr_state_dict.items():
            if key not in current_state_dict.keys():
                pass
            else:
                current_state_dict[key] = value

        # 1.5. apply to current model
        model.load_state_dict(current_state_dict)
        encoder = model.model.encoder.to(rank)
        
    else:
        seqclr_state_dict = torch.load(f"/home/solee0022/seqclr_exp/checkpoints/checkpoint-window-{method}/seqclr-200.pt", map_location=torch.device('cpu')).module.state_dict() #['model_state_dict']

        # 1.3. seq2seq state_dict
        current_state_dict = model.state_dict()

        # 1.4. apply loaded state_dict to current_state_dict(correspond to seqclr)
        for key, value in seqclr_state_dict.items():
            if key not in current_state_dict.keys():
                pass
            else:
                current_state_dict[key] = value

        # 1.5. apply to current model
        model.load_state_dict(current_state_dict)
        encoder = model.model.encoder.to(rank)
        
    print(encoder)
    # construct DDP model
    ddp_encoder = DDP(encoder, device_ids=[rank])
    word_emb(method, ddp_encoder)
    char_emb(method, ddp_encoder)


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

