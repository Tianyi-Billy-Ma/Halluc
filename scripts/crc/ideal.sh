source ~/.bashrc
module load cuda/12.1
cd /users/tma2/Projects/Halluc
conda activate llmhalluc

python -m llmhalluc.scripts.chat_completion --model_name_or_path ./outputs/llama-3.2-1b/gsm8k_masked_backtrack/sft/lora --message "What does <|reserved_special_token_0|> means?"
