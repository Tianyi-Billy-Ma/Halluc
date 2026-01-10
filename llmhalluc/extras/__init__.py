import os

from .constant import CACHE_DIR, CACHE_PATH, MODEL_PATH, OUTPUT_PATH, POSSIBLE_CACHE_DIR


def _init_cache_dir():
    global CACHE_DIR, CACHE_PATH, MODEL_PATH, OUTPUT_PATH

    # Check for environment variable first
    env_cache = os.environ.get("HALLUC_CACHE_DIR")
    if env_cache and os.path.exists(env_cache):
        CACHE_DIR = env_cache
    else:
        for path in POSSIBLE_CACHE_DIR:
            if os.path.exists(path):
                CACHE_DIR = path
                break

    CACHE_DIR = CACHE_DIR or "./"  # default to current directory
    OUTPUT_PATH = os.path.join(CACHE_DIR, "outputs")
    os.makedirs(OUTPUT_PATH, exist_ok=True)


if not CACHE_PATH or not os.path.exists(CACHE_DIR):
    _init_cache_dir()


__all__ = ["CACHE_DIR", "CACHE_PATH", "MODEL_PATH", "OUTPUT_PATH"]
