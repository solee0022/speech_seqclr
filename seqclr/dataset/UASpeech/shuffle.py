import json
import numpy as np
from sklearn.model_selection import train_test_split



def load_data_from_json(jsonl_path):
    with open(jsonl_path, "r") as f:
        jsonl_data = list(f)

    data = []
    for pair in jsonl_data:
        sample = json.loads(pair)
        data.append(sample)
    return data

# Inteligibility
VL = ['M04', 'F03', 'M12', 'M01']
L = ['M07', 'F02', 'M16']
M = ['M05', 'M11', 'F04']
H = ['M09', 'M14', 'M10', 'M08', 'F05']

speakers = ['M04', 'F03', 'M12', 'M01','M07', 'F02', 'M16','M05', 'M11', 'F04','M09', 'M14', 'M10', 'M08', 'F05']
unknown = ['CM13', 'CF04', 'CF02', 'CM10', 'CM12', 'CM09', 'CM08', 'CM04', 'CM05', 'CM06', 'CF03']

# shuffled_ds_f = open("/home/solee0022/seqclr_exp/speech_seqclr/seqclr/dataset/TORGO/train_arrayMic_shuffled.jsonl", "w")
# train_ds = load_data_from_json("/home/solee0022/seqclr_exp/speech_seqclr/seqclr/dataset/TORGO/train_arrayMic.jsonl")
# X_train, X_test = train_test_split(train_ds, test_size=0.00000001, random_state=123, shuffle=True)

# make file
f_train = open("/home/solee0022/seqclr_exp/speech_seqclr/seqclr/dataset/UASpeech/UASpeech_train_M5.jsonl", "w")
f_M04 = open("/home/solee0022/seqclr_exp/speech_seqclr/seqclr/dataset/UASpeech/UASpeech_eval_M04.jsonl", "w")
f_F03 = open("/home/solee0022/seqclr_exp/speech_seqclr/seqclr/dataset/UASpeech/UASpeech_eval_F03.jsonl", "w")
f_M12 = open("/home/solee0022/seqclr_exp/speech_seqclr/seqclr/dataset/UASpeech/UASpeech_eval_M12.jsonl", "w")
f_M01 = open("/home/solee0022/seqclr_exp/speech_seqclr/seqclr/dataset/UASpeech/UASpeech_eval_M01.jsonl", "w")
f_M07 = open("/home/solee0022/seqclr_exp/speech_seqclr/seqclr/dataset/UASpeech/UASpeech_eval_M07.jsonl", "w")
f_F02 = open("/home/solee0022/seqclr_exp/speech_seqclr/seqclr/dataset/UASpeech/UASpeech_eval_F02.jsonl", "w")
f_M16 = open("/home/solee0022/seqclr_exp/speech_seqclr/seqclr/dataset/UASpeech/UASpeech_eval_M16.jsonl", "w")
f_M05 = open("/home/solee0022/seqclr_exp/speech_seqclr/seqclr/dataset/UASpeech/UASpeech_eval_M05.jsonl", "w")
f_M11 = open("/home/solee0022/seqclr_exp/speech_seqclr/seqclr/dataset/UASpeech/UASpeech_eval_M11.jsonl", "w")
f_F04 = open("/home/solee0022/seqclr_exp/speech_seqclr/seqclr/dataset/UASpeech/UASpeech_eval_F04.jsonl", "w")
f_M09 = open("/home/solee0022/seqclr_exp/speech_seqclr/seqclr/dataset/UASpeech/UASpeech_eval_M09.jsonl", "w")
f_M14 = open("/home/solee0022/seqclr_exp/speech_seqclr/seqclr/dataset/UASpeech/UASpeech_eval_M14.jsonl", "w")
f_M10 = open("/home/solee0022/seqclr_exp/speech_seqclr/seqclr/dataset/UASpeech/UASpeech_eval_M10.jsonl", "w")
f_M08 = open("/home/solee0022/seqclr_exp/speech_seqclr/seqclr/dataset/UASpeech/UASpeech_eval_M08.jsonl", "w")
f_F05 = open("/home/solee0022/seqclr_exp/speech_seqclr/seqclr/dataset/UASpeech/UASpeech_eval_F05.jsonl", "w")

train_ds = load_data_from_json("/home/solee0022/seqclr_exp/speech_seqclr/seqclr/dataset/UASpeech/UASpeech_seqclr_train_shuffled.jsonl")
eval_ds = load_data_from_json("/home/solee0022/seqclr_exp/speech_seqclr/seqclr/dataset/UASpeech/UASpeech_seqclr_eval.jsonl")

for line in train_ds:
    mic = line['audio'].split("/")[3].split("_")[3].split(".wav")[0]
    if mic == 'M5':
        json.dump(line, f_train, ensure_ascii=False)
        f_train.write("\n")          
    
for line in eval_ds:
    speaker = line['audio'].split("/")[3].split("_")[0]
    mic = line['audio'].split("/")[3].split("_")[3].split(".wav")[0]
    if mic == 'M5':
        if speaker=='M04':
            json.dump(line, f_M04, ensure_ascii=False)
            f_M04.write("\n")   
        elif speaker=='F03':
            json.dump(line, f_F03, ensure_ascii=False)
            f_F03.write("\n")  
        elif speaker=='M12':
            json.dump(line, f_M12, ensure_ascii=False)
            f_M12.write("\n")      
        elif speaker=='M01':
            json.dump(line, f_M01, ensure_ascii=False)
            f_M01.write("\n")      
        elif speaker=='M07':
            json.dump(line, f_M07, ensure_ascii=False)
            f_M07.write("\n")      
        elif speaker=='F02':
            json.dump(line, f_F02, ensure_ascii=False)
            f_F02.write("\n")      
        elif speaker=='M16':
            json.dump(line, f_M16, ensure_ascii=False)
            f_M16.write("\n")          
        elif speaker=='M05':
            json.dump(line, f_M05, ensure_ascii=False)
            f_M05.write("\n")      
        elif speaker=='M11':
            json.dump(line, f_M11, ensure_ascii=False)
            f_M11.write("\n")      
        elif speaker=='F04':
            json.dump(line, f_F04, ensure_ascii=False)
            f_F04.write("\n")      
        elif speaker=='M09':
            json.dump(line, f_M09, ensure_ascii=False)
            f_M09.write("\n")      
        elif speaker=='M14':
            json.dump(line, f_M14, ensure_ascii=False)
            f_M14.write("\n")      
        elif speaker=='M10':        
            json.dump(line, f_M10, ensure_ascii=False)
            f_M10.write("\n")       
        elif speaker=='M08':
            json.dump(line, f_M08, ensure_ascii=False)
            f_M08.write("\n")   
        elif speaker=='F05':
            json.dump(line, f_F05, ensure_ascii=False)
            f_F05.write("\n")
    else:
        pass