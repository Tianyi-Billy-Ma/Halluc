#!/bin/bash
# Interactive session script for NCSA Delta
# Usage: ./scripts/delta/interactive.sh [time_in_minutes] [num_gpus]

TIME=${1:-30}     # Default 30 minutes
GPUS=${2:-1}      # Default 1 GPU

echo "Requesting interactive session with $GPUS GPU(s) for $TIME minutes..."

srun \
    --account=bgdn-delta-gpu \
    --partition=gpuA100x4 \
    --nodes=1 \
    --gpus-per-node=$GPUS \
    --ntasks-per-node=1 \
    --cpus-per-task=$((GPUS * 4)) \
    --mem=$((GPUS * 60))G \
    --time=00:$TIME:00 \
    --pty /bin/bash
