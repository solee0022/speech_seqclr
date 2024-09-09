# speech_seqclr


## Dataset
### UASpeech
|model|speaker|baseline|supervised|seqclr|seqclr-hard|
|:---:|:---:|:---:|:---:|:---:|:---:|
||VL|1.6549||||
||L|1.5241||||
|w2v-base-960h|M|1.2693||||
||H|0.6564||||
||HEALTHY|0.4563||||
|:---:|:---:|:---:|:---:|:---:|:---:|
||VL|1.8294||||
||L|1.5267||||
|w2v-large-960h|M|1.3022||||
||H|0.6007||||
||HEALTHY|0.3664||||


## Training
### 1. seqclr-encoder fine-tune
```
>> $ sbatch run_seqclr.sh

N_GPU=4

SCRIPT_PATH=$HOME/seqclr_exp/speech_seqclr #file path
OMP_NUM_THREADS=1 torchrun \
    --nproc_per_node=$N_GPU \
    --nnodes=1 \
    --rdzv_backend c10d \
    --rdzv_endpoint localhost:0 \
$SCRIPT_PATH/run_seqclr.py \
    --c seqclr/configs/seqclr_model.yaml \
```

### 2. ASR fine-tune
```
>> $ sbatch run_asr.sh

N_GPU=4

SCRIPT_PATH=$HOME/seqclr_exp/speech_seqclr #file path
OMP_NUM_THREADS=1 torchrun \
    --nproc_per_node=$N_GPU \
    --nnodes=1 \
    --rdzv_backend c10d \
    --rdzv_endpoint localhost:0 \
$SCRIPT_PATH/run_asr.py \
    --c seqclr/configs/seqclr_model.yaml \
```

## Inference
run ```python inference.py```
```
def main(
    method = "seqclr",
    asr_model = "base",
    ): 
```