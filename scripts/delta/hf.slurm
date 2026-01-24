#!/bin/bash
#SBATCH --job-name="llmhalluc"
#SBATCH --account=bgdn-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --no-requeue
#SBATCH -t 08:00:00
#SBATCH -o ./logs/slurm-%j.log
#SBATCH --mail-user=tma2@nd.edu
#SBATCH --mail-type=BEGIN,END


source ~/.bashrc
cd ~/Halluc
source ./scripts/delta/bashrc.sh
conda activate llmhalluc

accelerate launch -m lm_eval \
    --tasks squadv2 \
    --include_path ./configs/lm_eval/tasks \
    --model_args pretrained=qwen/Qwen3-4B-Base \
    --output_path ./outputs/qwen3-4b/squadv2/vanilla/eval/squadv2/results.json \
    --log_samples
