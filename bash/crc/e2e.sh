#!/bin/bash

#$ -M tma2@nd.edu    # Email address for job notification
#$ -m abe            # Send mail when job begins, ends and aborts
#$ -pe smp 32        # Specify parallel environment and legal core size
#$ -q gpu@@yye7_lab  # Run on the GPU cluster
#$ -o ~/Projects/Halluc/logs/$JOB_NAME_$JOB_ID.log
#$ -l gpu_card=4     # Run on 1 GPU card
#$ -N llmhalluc_e2e      # Specify job name

RUN_CONFIG="./configs/llamafactory/train.yaml"
MERGE_CONFIG="./configs/llamafactory/merge.yaml"

WANDB_PROJECT_NAME="llamafactory"
MODEL_DIR="./models"
OUTPUT_DIR="./outputs"
NUM_FEWSHOT=8
SEED=3
HF_USERNAME="mtybilly"
DDP=1

# Method 
STAGE="sft"
FINETUNING_TYPE="lora"
TRAIN_ABBR="train_bt"
# Model Name and Abbr
MODEL_NAME_OR_PATH="Qwen/Qwen3-4B-Instruct-2507"
ENABLE_THINKING=false

# Automatically set MODEL_ABBR based on MODEL_NAME_OR_PATH
MODEL_ABBR=$(basename "$MODEL_NAME_OR_PATH" | tr '[:upper:]' '[:lower:]')

# Merge Model 
MERGE_TEMPLATE="qwen3"

# Dataset
DATASET_NAME="gsm8k_symbolic_bt_train"
EVAL_DATASET_NAME=""

# lm_eval
TASK_NAME="gsm8k"

WANDB_NAME="${MODEL_ABBR}_${TASK_NAME}_${STAGE}_${FINETUNING_TYPE}"

MODEL_PATH="${MODEL_DIR}/${WANDB_NAME}"
EVAL_OUTPUT_PATH="${OUTPUT_DIR}/${WANDB_NAME}/lm_eval/results.json"
TRAIN_OUTPUT_PATH="$OUTPUT_DIR/$MODEL_ABBR/$TASK_NAME/$STAGE/$FINETUNING_TYPE/$TRAIN_ABBR"

echo "================================================"
echo "Training Phase"
echo "================================================"

source ./bash/sys/init_env.sh llamafactory

./bash/sys/log_yaml.sh $RUN_CONFIG
llamafactory-cli train $RUN_CONFIG \
    model_name_or_path=$MODEL_NAME_OR_PATH \
    enable_thinking=$ENABLE_THINKING \
    output_dir=$TRAIN_OUTPUT_PATH \
    stage=$STAGE \
    finetuning_type=$FINETUNING_TYPE \
    run_name=$WANDB_NAME\
    dataset=$DATASET_NAME \
    eval_dataset=$EVAL_DATASET_NAME \

echo "================================================"
echo "Merge Models"
echo "================================================"


./bash/sys/log_yaml.sh $MERGE_CONFIG
llamafactory-cli merge $MERGE_CONFIG \
    model_name_or_path=$MODEL_NAME_OR_PATH \
    adapter_name_or_path=$TRAIN_OUTPUT_PATH \
    template=$MERGE_TEMPLATE \
    export_dir=$MODEL_PATH \
    export_hub_model_id=$HF_USERNAME/$WANDB_NAME


echo "================================================"
echo "Evaluation Phase"
echo "================================================"

source ./bash/sys/init_env.sh lm_eval


# Build the base command
BASE_CMD="lm_eval --model hf \
    --model_args pretrained=${MODEL_PATH},enable_thinking=False\
    --tasks ${TASK_NAME} \
    --output_path ${EVAL_OUTPUT_PATH} \
    --num_fewshot ${NUM_FEWSHOT} \
    --seed ${SEED} \
    --wandb_args project=${WANDB_PROJECT_NAME},name=${WANDB_NAME} \
    --log_samples \
    --apply_chat_template"

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
    
    
