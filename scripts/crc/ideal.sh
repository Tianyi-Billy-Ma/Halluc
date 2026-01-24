source ~/.bashrc
module load cuda/12.1
cd /users/tma2/Projects/Halluc
conda activate llmhalluc



# Qwen3-4B-Base: 4B pretrained model, requires transformers>=4.51.0
# Compatible with current environment (no transformers 5.0 needed)
# MODEL_NAME_OR_PATH="Qwen/Qwen3-4B-Base"
MODEL_NAME_OR_PATH="meta-llama/Llama-3.1-8B"
ADAPTER_NAME_OR_PATH="/users/tma2/Projects/Halluc/outputs/qwen3-4b/squad_v2/sft/lora/train"
TOKENIZER_NAME_OR_PATH="Qwen/Qwen3-4B-Base"

export HF_ALLOW_CODE_EVAL=1

# accelerate launch -m llmhalluc.run_eval \
#     --config ./configs/llmhalluc/backtrack_sft.yaml 

# accelerate launch -m lm_eval \
#     --model_args pretrained=${MODEL_NAME_OR_PATH} \
#     --tasks bigbench_strategyqa_generate_until \
#     --include_path ./configs/lm_eval/tasks \
#     --log_samples \
#     --output_path ./outputs/llama-3.2-1b/strategyqa/vanilla/eval/strategyqa/results.json \
#     --wandb_args name=llama-3.2-1b_vanilla_strategyqa,project=Halluc
    
# accelerate launch -m llmhalluc.run_exp \
#     --config ./configs/llmhalluc/gsm8k/masked_sft.yaml \
#     --model_name_or_path ${MODEL_NAME_OR_PATH} 
#

accelerate launch -m lm_eval \
    --tasks mbpp \
    --include_path ./configs/lm_eval/tasks \
    --model_args pretrained=${MODEL_NAME_OR_PATH},trust_remote_code=True \
    --output_path ./outputs/llama-3.1-8b/mbpp/vanilla/eval/results.json \
    --log_samples \
    --confirm_run_unsafe_code