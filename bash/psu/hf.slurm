#!/bin/bash
#SBATCH --job-name="llmhalluc"
#SBATCH --partition=GPU
#SBATCH --gpus=h100-80:8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --account=cis240110p
#SBATCH -t 04:00:00
#SBATCH -e ./logs/slurm-%j.err
#SBATCH -o ./logs/slurm-%j.out
#SBATCH --mail-user=tma2@nd.edu
#SBATCH --mail-type=BEGIN,END

# Navigate to working directory
cd /ocean/projects/cis240110p/zyuan2/billy/Halluc
# Activate conda environment
conda activate billy

# Run commands
accelerate launch -m llmhalluc.hf_train
python -m llmhalluc.run_eval
./bash/sys/notify.sh
