source ~/.bashrc
module load cuda/12.1
cd /users/tma2/Projects/Halluc
conda activate llmhalluc

accelerate launch -m lm_eval --task gsm8k_cot_simple --model_path meta-llama/Llama-3.2-1b --batch_s