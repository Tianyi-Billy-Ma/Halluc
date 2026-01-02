#!/bin/bash

#$ -M tma2@nd.edu    # Email address for job notification
#$ -m abe            # Send mail when job begins, ends and aborts
#$ -pe smp 16        # Specify parallel environment and legal core size
#$ -q gpu@@yye7_lab  # Run on the GPU cluster
#$ -o ~/Projects/Halluc/logs/$JOB_NAME_$JOB_ID.log
#$ -l gpu_card=2     # Run on 2 GPU cards
#$ -N llmhalluc      # Specify job name


source ./bash/sys/init_env.sh llmhalluc

lm_eval --model hf \
    --model_args pretrained=/users/tma2/Projects/ReVISE/outputs/step-0-sft \
    --tasks gsm8k_comprehensive \
    --output_path /users/tma2/Projects/Halluc/outputs/llama-3.2-1b/gsm8k/sft/eval \
    --log_samples \
    --include_path ./configs/lm_eval/tasks
./bash/sys/notify.sh
