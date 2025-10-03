#!/bin/bash

#$ -M tma2@nd.edu    # Email address for job notification
#$ -m abe            # Send mail when job begins, ends and aborts
#$ -pe smp 32        # Specify parallel environment and legal core size
#$ -q gpu@@yye7_lab  # Run on the GPU cluster
#$ -o ~/Projects/Halluc/logs/$JOB_NAME_$JOB_ID.log
#$ -l gpu_card=4     # Run on 1 GPU card
#$ -N LLMHalluc      # Specify job name

# RUN_CONFIG="./configs/qwen3/0.6b/gsm8k_train.yaml"
RUN_CONFIG="./configs/qwen3/4b/gsm8k_train_ds0.yaml"

source ./bash/sys/init_env.sh llamafactory

./bash/sys/log_yaml.sh $RUN_CONFIG
llamafactory-cli train $RUN_CONFIG
# llamafactory-cli train ./configs/qwen3/0.6B/gsm8k_eval.yaml
