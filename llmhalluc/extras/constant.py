BACKTRACK_TOKEN = "<|BACKTRACK|>"
HF_USER_ID = "mtybilly"


# Default semantic words for backtrack-like tokens
BACKTRACK_SEMANTIC_WORDS = [
    "delete",
    "remove",
    "undo",
    "erase",
    "back",
    "cancel",
    "retry",
    "revert",
    "reset",
    "clear",
    "backspace",
]


# Special token configuration mapping
# Config format:
#   "<|TOKEN|>": {
#       "description": "Description of the token",
#       "semantic_words": ["word1", "word2", ...],
#       "strategy": "combined" | "description" | "semantic" | "mean",
#       "description_weight": 0.5
#   }
SPECIAL_TOKEN_MAPPING = {
    "llama3": {
        "<|reserved_special_token_0|>": {
            "description": "This token is used to delete the previous token in the response.",
            "semantic_words": BACKTRACK_SEMANTIC_WORDS,
            "strategy": "combined",
            "description_weight": 0.5,
        }
    },
    "llama": {
        "<|reserved_special_token_0|>": {
            "description": "This token is used to delete the previous token in the response.",
            "semantic_words": BACKTRACK_SEMANTIC_WORDS,
            "strategy": "combined",
            "description_weight": 0.5,
        }
    },
    "qwen3": {
        "<|BACKTRACK|>": {
            "description": "This token is used to delete the previous token in the response.",
            "semantic_words": BACKTRACK_SEMANTIC_WORDS,
            "strategy": "combined",
            "description_weight": 0.5,
        }
    },
}


POSSIBLE_CACHE_DIR = [
    "/scratch365/tma2/.cache/halluc/",
    "/work/nvme/bemy/tma3/.cache/halluc/",
]

CACHE_DIR = None
CACHE_PATH = None
# MODEL_PATH = "/scratch365/tma2/.cache/halluc/models"
# OUTPUT_PATH = "/scratch365/tma2/.cache/halluc/outputs"
# MODEL_PATH = "/work/nvme/bemy/tma3/.cache/halluc/models"
# OUTPUT_PATH = "/work/nvme/bemy/tma3/.cache/halluc/outputs"
MODEL_PATH = "./models"
OUTPUT_PATH = "./outputs"
