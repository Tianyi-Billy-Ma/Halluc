import os

from .constant import POSSIBLE_CACHE_DIR, CACHE_DIR, CACHE_PATH, MODEL_PATH, OUTPUT_PATH


def _init_cache_dir():
    global CACHE_DIR, CACHE_PATH, MODEL_PATH, OUTPUT_PATH
    for path in POSSIBLE_CACHE_DIR:
        if os.path.exists(path):
            CACHE_DIR = path
            break

    CACHE_DIR = CACHE_DIR or "./"  # default to current directory
    CACHE_PATH = os.path.join(CACHE_DIR, "halluc")
    MODEL_PATH = os.path.join(CACHE_PATH, "models")
    OUTPUT_PATH = os.path.join(CACHE_PATH, "outputs")
    os.makedirs(CACHE_PATH, exist_ok=True)
    os.makedirs(MODEL_PATH, exist_ok=True)
    os.makedirs(OUTPUT_PATH, exist_ok=True)


if not CACHE_PATH or not os.exists(CACHE_DIR):
    _init_cache_dir()


__all__ = ["CACHE_DIR", "CACHE_PATH", "MODEL_PATH", "OUTPUT_PATH"]
