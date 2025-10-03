#!/bin/bash

# Get environment name from first argument, default to "llamafactory"
ENV_NAME=${1:-llamafactory}

# Source bashrc to ensure conda/venv commands are available
source ~/.bashrc

# Change to working directory
cd $WORK_DIR

if command -v conda &> /dev/null; then
    # Try conda first
    if conda env list | grep -q "^$ENV_NAME "; then
        conda activate $ENV_NAME
    else
        echo "Warning: Conda environment '$ENV_NAME' not found"
        exit 1
    fi
else 
    echo "Warning: Conda not found"
    exit 1
fi

echo "================================================"
echo "Environment: $ENV_NAME"
echo "Available GPUs: $CUDA_VISIBLE_DEVICES"
echo "Working Directory: $WORK_DIR"
echo "Python Path: $(which python)"
echo "================================================"