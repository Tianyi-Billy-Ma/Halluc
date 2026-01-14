source ~/.bashrc
module load cuda/12.1
cd /users/tma2/Projects/Halluc
conda activate llmhalluc

# accelerate launch -m lm_eval --model hf --model_args pretrained=meta-llama/Llama-3.2-3B --tasks gsm8k_cot_simple --output_path outputs/llama-3.2-3b/gsm8k/vanilla/eval/gsm8k_cot_simple/results.json --include_path ./configs/lm_eval/tasks --log_samples --wandb_args name=llama-3.2-3b_gsm8k_cot_simple_vanilla,project=Halluc

accelerate launch -m llmhalluc.run_exp --config configs/llmhalluc/sft.yaml

