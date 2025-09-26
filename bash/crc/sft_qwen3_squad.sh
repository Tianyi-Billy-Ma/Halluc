#!/bin/bash

#$ -M tma2@nd.edu    # Email address for job notification
#$ -m abe            # Send mail when job begins, ends and aborts
#$ -pe smp 16        # Specify parallel environment and legal core size
#$ -q gpu@@yye7_lab  # Run on the GPU cluster
#$ -o ~/Projects/LLaMA-Factory/logs/$JOB_NAME_$JOB_ID.log
#$ -l gpu_card=2     # Run on 1 GPU card
#$ -N LLMHalluc      # Specify job name

source ~/.bashrc
cd ~/Projects/LLaMA-Factory
source ./bash/sys/activate_env.sh

echo $CUDA_VISIBLE_DEVICES

llamafactory-cli train ./configs/qwen3/qwen3_4b_squad_v2_ds0_train.yaml 
