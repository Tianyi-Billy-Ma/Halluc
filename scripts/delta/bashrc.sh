module load pytorch-conda/2.8
module load aws-ofi-nccl/1.14.2
conda activate llmhalluc

# Weights & Biases Configuration for Delta
export WANDB_MODE=offline
# export WANDB_SERVICE_WAIT=300
export WANDB_DIR=/work/hdd/bgdn/tma3/.cache/wandb
mkdir -p $WANDB_DIR

# HuggingFace Cache
export HF_HOME=/work/hdd/bgdn/tma3/.cache/huggingface
mkdir -p $HF_HOME