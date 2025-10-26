#!/bin/bash

#$ -M tma2@nd.edu    # Email address for job notification
#$ -m abe            # Send mail when job begins, ends and aborts
#$ -pe smp 16        # Specify parallel environment and legal core size
#$ -q gpu@@yye7_lab  # Run on the GPU cluster
#$ -o ~/Projects/Halluc/logs/$JOB_NAME_$JOB_ID.log
#$ -l gpu_card=2     # Run on 2 GPU card
#$ -N lm_eval      # Specify job name



source ./bash/sys/init_env.sh lm_eval

WANDB_PROJECT_NAME="llamafactory"
MODEL_DIR="./models"
OUTPUT_DIR="./outputs"
DDP=1

STAGE="sft"
FINETUNING_TYPE="lora"
MODEL_NAME="qwen3-4b"
TASK_NAME="gsm8k"

NUM_FEWSHOT=8
SEED=3

# FULL_MODEL_NAME="${MODEL_NAME}-${TASK_NAME}-${FINETUNING_TYPE}"
# MODEL_PATH="${MODEL_DIR}/${FULL_MODEL_NAME}"
# OUTPUT_PATH="${OUTPUT_DIR}/${MODEL_NAME}/${TASK_NAME}/${STAGE}/${FINETUNING_TYPE}/lm_eval/results.json"
# WANDB_NAME="${FULL_MODEL_NAME}"

MODEL_PATH="Qwen/Qwen3-0.6B"
OUTPUT_PATH="${OUTPUT_DIR}/qwen3-0.6b/${TASK_NAME}/vanilla/lm_eval/fewshot_${NUM_FEWSHOT}/results.json"
WANDB_NAME="qwen3-0.6b_gsm8k_vanilla"


# Build the base command
BASE_CMD="lm_eval --model hf \
    --model_args pretrained=${MODEL_PATH},enable_thinking=False\
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
    
    
