#!/bin/bash

#$ -M tma2@nd.edu    # Email address for job notification
#$ -m abe            # Send mail when job begins, ends and aborts
#$ -pe smp 16        # Specify parallel environment and legal core size
#$ -q gpu@@yye7_lab  # Run on the GPU cluster
#$ -o ~/Projects/Halluc/logs/$JOB_NAME_$JOB_ID.log
#$ -l gpu_card=2     # Run on 2 GPU cards
#$ -N llmhalluc      # Specify job name


source ./bash/sys/init_env.sh llmhalluc
# accelerate launch -m llmhalluc.hf_train
python -m llmhalluc.run_eval
./bash/sys/notify.sh
