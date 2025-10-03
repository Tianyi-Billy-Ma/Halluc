#!/bin/bash

#$ -M tma2@nd.edu    # Email address for job notification
#$ -m abe            # Send mail when job begins, ends and aborts
#$ -pe smp 8        # Specify parallel environment and legal core size
#$ -q gpu@@yye7_lab  # Run on the GPU cluster
#$ -o ~/Projects/Halluc/logs/$JOB_NAME_$JOB_ID.log
#$ -l gpu_card=1     # Run on 1 GPU card
#$ -N export_model      # Specify job name

RUN_CONFIG="./configs/llama/3b/merge_adapter.yaml"

source ./bash/sys/init_env.sh llamafactory

./bash/sys/log_yaml.sh $RUN_CONFIG
llamafactory-cli export $RUN_CONFIG
