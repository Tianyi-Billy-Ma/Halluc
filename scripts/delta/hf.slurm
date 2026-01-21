#!/bin/bash
#SBATCH --job-name="llmhalluc"
#SBATCH --account=bgdn-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --mem=240G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=closest
#SBATCH --no-requeue
#SBATCH -t 04:00:00
#SBATCH -o ./logs/slurm-%j.log
#SBATCH --mail-user=tma2@nd.edu
#SBATCH --mail-type=BEGIN,END


source ~/.bashrc
cd ~/Halluc
source ./scripts/delta/bashrc.sh
conda activate llmhalluc

# Fix WandB and Triton NFS issues
export WANDB_DIR=/tmp/$USER/wandb
mkdir -p $WANDB_DIR
export TRITON_CACHE_DIR=/tmp/$USER/triton
mkdir -p $TRITON_CACHE_DIR
export WANDB_SERVICE_WAIT=300

accelerate launch -m llmhalluc.run_exp \
    --config ./configs/llmhalluc/gsm8k/masked_sft.yaml \
    --model_name_or_path "meta-llama/Llama-3.1-8b"

