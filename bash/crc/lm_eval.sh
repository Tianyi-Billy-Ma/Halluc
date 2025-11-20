#!/bin/bash

#$ -M tma2@nd.edu    # Email address for job notification
#$ -m abe            # Send mail when job begins, ends and aborts
#$ -pe smp 16        # Specify parallel environment and legal core size
#$ -q gpu@@yye7_lab  # Run on the GPU cluster
#$ -o ~/Projects/Halluc/logs/$JOB_NAME_$JOB_ID.log
#$ -l gpu_card=2     # Run on 2 GPU card
#$ -N lm_eval      # Specify job name


# SOME THING NEED TO CHANGE

EXP_NAME="gsm8k"
METHOD_NAME="vanilla"


WANDB_PROJECT_NAME="llamafactory"
MODEL_DIR="/scratch365/tma2/.cache/halluc/models"
OUTPUT_DIR="/scratch365/tma2/.cache/halluc/outputs"
SEED=3
DDP=1

# Method
STAGE="vanilla"
EVAL_TASK_NAME="gsm8k_bt"
EVAL_MODEL="hf"

# Model Name and Abbr
# MODEL_NAME_OR_PATH="Qwen/Qwen3-0.6B-Base"
# MODEL_NAME_OR_PATH="meta-llama/Llama-3.2-1B"
# MODEL_NAME_OR_PATH="meta-llama/Llama-3.2-1B-Instruct"
# MODEL_NAME_OR_PATH="meta-llama/Llama-3.1-8B"
MODEL_NAME_OR_PATH="meta-llama/Llama-3.1-8B-Instruct"
# MODEL_NAME_OR_PATH="meta-llama/Llama-3.2-3B"
# MODEL_NAME_OR_PATH="meta-llama/Llama-3.2-3B-Instruct"
ENABLE_THINKING=false

MODEL_ABBR=$(basename "$MODEL_NAME_OR_PATH" | tr '[:upper:]' '[:lower:]')
WANDB_NAME="${MODEL_ABBR}_${EXP_NAME}_${METHOD_NAME}"


if [ "$METHOD_NAME" == "vanilla" ]; then
    MODEL_PATH="${MODEL_NAME_OR_PATH}"
else
    MODEL_PATH="${MODEL_DIR}/${WANDB_NAME}"
fi

EVAL_OUTPUT_PATH="${OUTPUT_DIR}/${MODEL_ABBR}/${EXP_NAME}/${METHOD_NAME}/eval/results.json"


source ./bash/sys/init_env.sh lm_eval

# Build the base command
BASE_CMD="lm_eval --model ${EVAL_MODEL} \
    --model_args pretrained=${MODEL_PATH},enable_thinking=${ENABLE_THINKING}\
    --tasks ${EVAL_TASK_NAME} \
    --output_path ${EVAL_OUTPUT_PATH} \
    --seed ${SEED} \
    --wandb_args project=${WANDB_PROJECT_NAME},name=${WANDB_NAME} \
    --log_samples \
    --confirm_run_unsafe_code \
    --include_path ./configs/lm_eval/tasks"
    # --apply_chat_template \


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
    
