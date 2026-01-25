source ~/.bashrc
module load cuda/12.1
cd /users/tma2/Projects/Halluc
conda activate llmhalluc

# Qwen3-4B-Base: 4B pretrained model, requires transformers>=4.51.0
MODEL_NAME_OR_PATH="Qwen/Qwen3-4B-Base"
OUTPUT_DIR="./outputs/qwen3-4b/gsm8k/sft"

export HF_ALLOW_CODE_EVAL=1

# ==============================================================================
# Step 1: SFT Training on GSM8K
# ==============================================================================
# accelerate launch -m llmhalluc.run_train \
#     --config ./configs/llmhalluc/gsm8k/sft.yaml \
#     --model_name_or_path ${MODEL_NAME_OR_PATH} \
#     --template qwen3 \
#     --output_dir ${OUTPUT_DIR}/train

# ==============================================================================
# Step 2: Evaluation using lm_eval
# ==============================================================================
accelerate launch -m lm_eval --model hf \
    --model_args pretrained=${MODEL_NAME_OR_PATH},peft=${OUTPUT_DIR}/train/train,trust_remote_code=True \
    --tasks gsm8k_simple \
    --include_path ./configs/lm_eval/tasks \
    --output_path ${OUTPUT_DIR}/eval/results.json \
    --log_samples