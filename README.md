# speech_seqclr


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


### 3. Inference
run '''python inference.py'''
```
def main(
    method = "seqclr",
    asr_model = "base",
    ): 
```