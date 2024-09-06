#!/bin/bash

# sbatch
#SBATCH -J seqclr_base # job name
#SBATCH -o ./out/output_%x.%j.out 
#SBATCH -p A100-pci # queue name or partiton name
#SBATCH -t 72:00:00 # Run time

# gpu 설정
## gpu 개수
#SBATCH   --gres=gpu:4
#SBTACH   --ntasks=1
##SBATCH   --nodelist=n61
#SBATCH   --tasks-per-node=1
#SBATCH   --cpus-per-task=16

N_GPU=4
cd  $SLURM_SUBMIT_DIR
echo "SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR"
echo "CUDA_HOME=$CUDA_HOME"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "CUDA_VERSION=$CUDA_VERSION"

echo "Start"
echo "conda PATH "

echo "source  $HOME/anaconda3/etc/profile.d/conda.sh"
source  $HOME/anaconda3/etc/profile.d/conda.sh

echo "conda activate py39_12.1" 
conda activate py39_12.1  #conda env

# run
SCRIPT_PATH=$HOME/seqclr_exp/speech_seqclr #file path
OMP_NUM_THREADS=1 torchrun \
    --nproc_per_node=$N_GPU \
    --nnodes=1 \
    --rdzv_backend c10d \
    --rdzv_endpoint localhost:0 \
$SCRIPT_PATH/run_seqclr.py \
    --c seqclr/configs/seqclr_model.yaml \

date

echo "conda deactivate py39_12.1"

conda deactivate 

date
squeue --job $SLURM_JOBID

echo "#####END#####"



