#!/bin/bash
#SBATCH -J llmhalluc               # 作业名称
#SBATCH -p GPU-shared                        # 分区名称
#SBATCH --gres=gpu:h100-80:2          # 请求 8 个 H100 GPU
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH -A cis250100p                 # 账户名称
#SBATCH -t 48:00:00                   # 预计运行时间（48 小时）
#SBATCH -N 1                          # 请求 1 个节点
#SBATCH -e ./logs/slurm-%j.err
#SBATCH -o ./logs/slurm-%j.out
#SBATCH --mail-user=tma2@nd.edu
#SBATCH --mail-type=BEGIN,END

# Navigate to working directory
cd /ocean/projects/cis240110p/zyuan2/billy/Halluc
# Activate conda environment
source ./bash/sys/bashrc.sh

module load anaconda3/2024.10-1
module load cuda/12.6.1

conda activate billy

sleep 48h
