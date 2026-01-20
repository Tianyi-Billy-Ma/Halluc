#!/bin/bash
#SBATCH --job-name="llmhalluc"
#SBATCH --output="a.out.%j.%N.out"
#SBATCH --partition=ghx4
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1  # could be 1 for py-torch
#SBATCH --cpus-per-task=16   # spread out to use 1 core per numa, set to 64 if tasks is 1
#SBATCH --constraint="projects"
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=closest   # select a cpu close to gpu on pci bus topology
#SBATCH --account=bemy-dtai-gh    # <- match to a "Project" returned by the "accounts" command
#SBATCH --exclusive  # dedicated node for this job
#SBATCH --no-requeue
#SBATCH -t 24:00:00
#SBATCH -o ./logs/slurm-%j.log
#SBATCH --mail-user=tma2@nd.edu
#SBATCH --mail-type="BEGIN,END"

source ~/.bashrc
cd /projects/bemy/tma3/LLaMA-Factory
source ./bash/sys/activate_env.sh

llamafactory-cli train ./configs/qwen3/qwen3_nothink_bt_squad_v2_train.yaml 