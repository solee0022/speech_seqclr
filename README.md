# speech_seqclr


### seqclr-encoder fine-tune
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


### ASR fine-tune
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