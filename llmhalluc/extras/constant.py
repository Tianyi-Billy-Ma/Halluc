import os

BACKTRACK_TOKEN = "<|BACKTRACK|>"
HF_USER_ID = "mtybilly"


SPECIAL_TOKEN_MAPPING = {
    "llama3": {
        "<|reserved_special_token_0|>": "This token is used to delete the previous token in the response."
    },
    "qwen3": {
        "<|BACKTRACK|>": "This token is used to delete the previous token in the response."
    },
}


POSSIBLE_CACHE_DIR = [
    "/scratch365/tma2/.cache/",
    "/work/nvme/bemy/tma3/.cache/",
]

CACHE_DIR = None
CACHE_PATH = None
MODEL_PATH = "/scratch365/tma2/.cache/halluc/models"
OUTPUT_PATH = "/scratch365/tma2/.cache/halluc/outputs"
