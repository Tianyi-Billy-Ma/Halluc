#!/bin/bash
#SBATCH --job-name="llmhalluc"
#SBATCH --partition=ghx4
#SBATCH --mem=240G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=closest
#SBATCH --account=bemy-dtai-gh
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH -t 04:00:00
#SBATCH -e ./logs/slurm-%j.err
#SBATCH -o ./logs/slurm-%j.out
#SBATCH --mail-user=tma2@nd.edu
#SBATCH --mail-type=BEGIN,END

source ~/.bashrc

cd ~/Projects/Halluc
conda activate llmhalluc

accelerate launch -m llmhalluc.hf_train
python -m llmhalluc.run_eval
./bash/sys/notify.sh
