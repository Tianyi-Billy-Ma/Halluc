#!/bin/bash

#$ -M tma2@nd.edu    # Email address for job notification
#$ -m abe            # Send mail when job begins, ends and aborts
#$ -pe smp 32        # Specify parallel environment and legal core size
#$ -q gpu@@yye7_lab  # Run on the GPU cluster
#$ -o ~/Projects/Halluc/logs/$JOB_NAME_$JOB_ID.log
#$ -l gpu_card=4     # Run on 1 GPU card
#$ -N llmhalluc_e2e      # Specify job name

TRAIN_CONFIG="./configs/llamafactory/train.yaml"
MERGE_CONFIG="./configs/llamafactory/merge.yaml"

WANDB_PROJECT_NAME="llamafactory"
MODEL_DIR="/scratch365/tma2/.cache/halluc/models"
OUTPUT_DIR="/scratch365/tma2/.cache/halluc/outputs"
SEED=3
HF_USERNAME="mtybilly"
DDP=1

# Method 
STAGE="sft"
FINETUNING_TYPE="lora"

# Model Name and Abbr
# MODEL_NAME_OR_PATH="Qwen/Qwen3-4B-Instruct-2507"
MODEL_NAME_OR_PATH="Qwen/Qwen3-0.6B"
ENABLE_THINKING=false


# Merge Model 
MERGE_TEMPLATE="qwen3"

# Dataset
TRAIN_DATASET_NAME="gsm8k_symbolic_bt_train"
EVAL_DATASET_NAME=""

# lm_eval
TASK_NAME="gsm8k_bt"
EVAL_MODEL="hf-bt"



### Automatic Parts. You can skip.

# Automatically set MODEL_ABBR based on MODEL_NAME_OR_PATH
MODEL_ABBR=$(basename "$MODEL_NAME_OR_PATH" | tr '[:upper:]' '[:lower:]')

if [ -z "$EVAL_DATASET_NAME" ]; then
    EVAL_DATASET_NAME=$TRAIN_DATASET_NAME
fi

WANDB_NAME="${MODEL_ABBR}_${TASK_NAME}_${STAGE}_${FINETUNING_TYPE}"

MODEL_PATH="${MODEL_DIR}/${WANDB_NAME}"
TRAIN_OUTPUT_PATH="${OUTPUT_DIR}/${MODEL_ABBR}/${TASK_NAME}/${STAGE}/${FINETUNING_TYPE}/train"
TRAIN_CONFIG_PATH="${OUTPUT_DIR}/${MODEL_ABBR}/${TASK_NAME}/${STAGE}/${FINETUNING_TYPE}/train_config.yaml"
MERGE_CONFIG_PATH="${OUTPUT_DIR}/${MODEL_ABBR}/${TASK_NAME}/${STAGE}/${FINETUNING_TYPE}/merge_config.yaml"
EVAL_OUTPUT_PATH="${OUTPUT_DIR}/${MODEL_ABBR}/${TASK_NAME}/${STAGE}/${FINETUNING_TYPE}/lm_eval/results.json"



echo "================================================"
echo "Training Phase"
echo "================================================"

source ./bash/sys/init_env.sh llamafactory

python llmhalluc/scripts/update_yaml.py \
    --input_yaml $TRAIN_CONFIG \
    --output_yaml $TRAIN_CONFIG_PATH \
    model_name_or_path=$MODEL_NAME_OR_PATH \
    enable_thinking=$ENABLE_THINKING \
    output_dir=$TRAIN_OUTPUT_PATH \
    stage=$STAGE \
    finetuning_type=$FINETUNING_TYPE \
    run_name=$WANDB_NAME\
    dataset=$TRAIN_DATASET_NAME \
    eval_dataset=$EVAL_DATASET_NAME 
    # add_special_tokens="<|BACKTRACK|>"

./bash/sys/log_yaml.sh $TRAIN_CONFIG_PATH
llamafactory-cli train $TRAIN_CONFIG_PATH

echo "================================================"
echo "Merge Models"
echo "================================================"

python llmhalluc/scripts/update_yaml.py \
    --input_yaml $MERGE_CONFIG \
    --output_yaml $MERGE_CONFIG_PATH \
    model_name_or_path=$MODEL_NAME_OR_PATH \
    adapter_name_or_path=$TRAIN_OUTPUT_PATH \
    template=$MERGE_TEMPLATE \
    export_dir=$MODEL_PATH \
    export_hub_model_id=$HF_USERNAME/$WANDB_NAME 
    # add_special_tokens="<|BACKTRACK|>"


./bash/sys/log_yaml.sh $MERGE_CONFIG_PATH
llamafactory-cli export $MERGE_CONFIG_PATH 

echo "================================================"
echo "Evaluation Phase"
echo "================================================"

source ./bash/sys/init_env.sh lm_eval

# Build the base command
BASE_CMD="lm_eval --model ${EVAL_MODEL} \
    --model_args pretrained=${MODEL_PATH},enable_thinking=False\
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
    
    
./bash/sys/notify.sh
