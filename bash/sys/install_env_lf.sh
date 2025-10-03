#!/bin/bash

conda create -n llamafactory python=3.11 -y
conda activate llamafactory
pip install -r ./LLaMA-Factory/requirements.txt
pip install -e ./LLaMA-Factory/[torch,metrics,deepspeed] --no-build-isolation
pip install flash-attn unsloth
# uv pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
