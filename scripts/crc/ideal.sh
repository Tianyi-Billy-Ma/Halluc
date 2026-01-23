source ~/.bashrc
module load cuda/12.1
cd /users/tma2/Projects/Halluc
conda activate llmhalluc



MODEL_NAME_OR_PATH="meta-llama/Llama-3.2-1b"
ADAPTER_NAME_OR_PATH="/users/tma2/Projects/Halluc/outputs/llama-3.2-1b/gsm8k/sft/lora/train"
TOKENIZER_NAME_OR_PATH="meta-llama/Llama-3.2-1b"


# accelerate launch -m llmhalluc.run_eval \
#     --config ./configs/llmhalluc/backtrack_sft.yaml 

# accelerate launch -m lm_eval \
#     --model_args pretrained=${MODEL_NAME_OR_PATH} \
#     --tasks bigbench_strategyqa_generate_until \
#     --include_path ./configs/lm_eval/tasks \
#     --log_samples \
#     --output_path ./outputs/llama-3.2-1b/strategyqa/vanilla/eval/strategyqa/results.json \
#     --wandb_args name=llama-3.2-1b_vanilla_strategyqa,project=Halluc
    
accelerate launch -m llmhalluc.run_exp \
    --config ./configs/llmhalluc/gsm8k/masked_sft.yaml \
    --model_name_or_path ${MODEL_NAME_OR_PATH} 
    