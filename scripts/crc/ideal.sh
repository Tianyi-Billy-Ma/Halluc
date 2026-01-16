source ~/.bashrc
module load cuda/12.1
cd /users/tma2/Projects/Halluc
conda activate llmhalluc



MODEL_NAME_OR_PATH="meta-llama/Llama-3.2-1b"
ADAPTER_NAME_OR_PATH="/users/tma2/Projects/Halluc/outputs/llama-3.2-1b/gsm8k/sft/lora/train"
TOKENIZER_NAME_OR_PATH="meta-llama/Llama-3.2-1b"


accelerate launch -m llmhalluc.run_eval \
    --config ./configs/llmhalluc/backtrack_sft.yaml 