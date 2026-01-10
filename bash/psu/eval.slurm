#!/bin/bash
#SBATCH --job-name="llmhalluc"
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --account=cis250100p
#SBATCH -t 04:00:00
#SBATCH -e ./logs/slurm-%j.err
#SBATCH -o ./logs/slurm-%j.out
#SBATCH --mail-user=tma2@nd.edu
#SBATCH --mail-type=BEGIN,END


# Navigate to working directory
cd /ocean/projects/cis240110p/zyuan2/billy/Halluc
# Activate conda environment

source ./bash/sys/bashrc.sh
module load anaconda3/2024.10-1
conda activate billy

# Run commands
lm_eval --model hf \
    --model_args pretrained=meta-llama/Llama-3.1-8B-instruct \
    --tasks gsm8k_simple \
    --output_path ./outputs/llama-3.1-8b-instruct/gsm8k/vanilla/eval/results.json \
    --include_path ./configs/lm_eval/tasks \
    --log_samples

python ./bash/psu/notify.py
