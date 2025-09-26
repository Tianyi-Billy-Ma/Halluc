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


WANDB_PROJECT_NAME="llamafactory"
MODEL_DIR="./models"
OUTPUT_DIR="./outputs"
DDP=1

STAGE="sft"
FINETUNING_TYPE="lora"

MODEL_NAME="qwen3-0.6b"
TASK_NAME="gsm8k"
FULL_MODEL_NAME="${MODEL_NAME}-${TASK_NAME}-${FINETUNING_TYPE}"
MODEL_PATH="${MODEL_DIR}/${FULL_MODEL_NAME}"
OUTPUT_PATH="${OUTPUT_DIR}/${MODEL_NAME}/${TASK_NAME}/${STAGE}/${FINETUNING_TYPE}/lm_eval"
WANDB_NAME="${FULL_MODEL_NAME}"
NUM_FEWSHOT=0
SEED=3

# Build the base command
BASE_CMD="lm_eval --model hf \
    --model_args pretrained=${MODEL_PATH} \
    --tasks ${TASK_NAME} \
    --output_path ${OUTPUT_PATH} \
    --num_fewshot ${NUM_FEWSHOT} \
    --seed ${SEED} \
    --wandb_args project=${WANDB_PROJECT_NAME},name=${WANDB_NAME} \
    --log_samples \
    --apply_chat_template"

echo "================================================"
echo "Available GPUs: $CUDA_VISIBLE_DEVICES"
echo "Model Path: ${MODEL_PATH}"
echo "Output Path: ${OUTPUT_PATH}"
echo "Num of Fewshot: ${NUM_FEWSHOT}"
echo "Seed: ${SEED}"
echo "================================================"

# Conditionally prepend accelerate launch for DDP
if [ $DDP -eq 1 ]; then
    CMD="accelerate launch -m ${BASE_CMD}"
else
    CMD="${BASE_CMD}"
fi

# Execute the command
eval $CMD
    
    
