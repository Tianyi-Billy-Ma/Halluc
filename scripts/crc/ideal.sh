source ~/.bashrc
module load cuda/12.1
cd /users/tma2/Projects/Halluc
conda activate llmhalluc



MODEL_NAME_OR_PATH="meta-llama/Llama-3.2-1b"
ADAPTER_NAME_OR_PATH="/users/tma2/Projects/Halluc/outputs/llama-3.2-1b/gsm8k/sft/lora/train"
TOKENIZER_NAME_OR_PATH="meta-llama/Llama-3.2-1b"


python -m llmhalluc.scripts.chat_completion \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --adapter_name_or_path $ADAPTER_NAME_OR_PATH \
    --tokenizer_name_or_path $TOKENIZER_NAME_OR_PATH \
    --message "Rania saw a 210-foot whale with 7 72-inch remoras attached to it. What percentage of the whale's body length is the combined length of the remoras? Let's think step by step. Put your final answer at the end with 'The answer is: .'"
