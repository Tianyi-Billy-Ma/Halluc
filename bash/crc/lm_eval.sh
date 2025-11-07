#!/bin/bash

#$ -M tma2@nd.edu    # Email address for job notification
#$ -m abe            # Send mail when job begins, ends and aborts
#$ -pe smp 32        # Specify parallel environment and legal core size
#$ -q gpu@@yye7_lab  # Run on the GPU cluster
#$ -o ~/Projects/Halluc/logs/$JOB_NAME_$JOB_ID.log
#$ -l gpu_card=4     # Run on 2 GPU card
#$ -N lm_eval      # Specify job name



source ./bash/sys/init_env.sh lm_eval

WANDB_PROJECT_NAME="llamafactory"
MODEL_DIR="./models"
OUTPUT_DIR="./outputs"
SEED=3
DDP=1

# Method
STAGE="vanilla"
TASK_NAME="gsm8k_bt"
EVAL_MODEL="hf"

# Model Name and Abbr
MODEL_NAME_OR_PATH="Qwen/Qwen3-4B-Instruct-2507"
ENABLE_THINKING=false

MODEL_ABBR=$(basename "$MODEL_NAME_OR_PATH" | tr '[:upper:]' '[:lower:]')
WANDB_NAME="${MODEL_ABBR}_${TASK_NAME}_${STAGE}"


if [ "$STAGE" == "vanilla" ]; then
    MODEL_PATH="${MODEL_NAME_OR_PATH}"
else
    MODEL_PATH="${MODEL_DIR}/${WANDB_NAME}"
fi

EVAL_OUTPUT_PATH="${OUTPUT_DIR}/${MODEL_ABBR}/${TASK_NAME}/${STAGE}/lm_eval/results.json"

# Build the base command
BASE_CMD="lm_eval --model ${EVAL_MODEL} \
    --model_args pretrained=${MODEL_PATH},enable_thinking=${ENABLE_THINKING}\
    --tasks ${TASK_NAME} \
    --output_path ${EVAL_OUTPUT_PATH} \
    --seed ${SEED} \
    --wandb_args project=${WANDB_PROJECT_NAME},name=${WANDB_NAME} \
    --log_samples \
    --apply_chat_template \
    --include_path ./configs/lm_eval/tasks"


echo "================================================"
echo "Available GPUs: $CUDA_VISIBLE_DEVICES"
echo "Model Path: ${MODEL_PATH}"
echo "Output Path: ${EVAL_OUTPUT_PATH}"
echo "Seed: ${SEED}"
echo "================================================"

# Conditionally prepend accelerate launch for DDP
if [ $DDP -eq 1 ]; then
    CMD="accelerate launch -m ${BASE_CMD}"
else
    CMD="${BASE_CMD}"
fi

# Execute the command
eval "$CMD"

./bash/sys/notify.sh
    
