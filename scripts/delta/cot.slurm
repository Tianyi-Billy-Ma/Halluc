#!/bin/bash
#SBATCH --job-name="llmhalluc"
#SBATCH --account=bgdn-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --mem=240G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
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



accelerate launch -m lm_eval \
    --tasks hendrycks_math500 \
    --model_args pretrained=meta-llama/Llama-3.1-8b \
    --output_path ./outputs/llama3-8b-cot-math500_simple \
    --log_samples 
    