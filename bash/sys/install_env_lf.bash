pip install -r requirements.txt
pip install -e .[torch,metrics,deepspeed] --no-build-isolation
pip install flash-attn unsloth
# uv pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
