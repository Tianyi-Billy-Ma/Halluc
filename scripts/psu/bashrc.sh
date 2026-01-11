#!/bin/bash

module load anaconda3/2024.10-1
module load cuda
export PYTHONDONTWRITEBYTECODE=1
export WANDB_DIR=/ocean/projects/cis240110p/zyuan2/billy/Halluc/wandb
conda activate billy