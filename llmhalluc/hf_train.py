"""HuggingFace training entry point.

Usage:
    accelerate launch -m llmhalluc.hf_train
    accelerate launch -m llmhalluc.hf_train --num_train_epochs 1
    accelerate launch -m llmhalluc.hf_train --learning_rate 5e-5 --per_device_train_batch_size 8
"""

import sys
from pathlib import Path

from llmhalluc.utils import setup_logging, hf_cfg_setup
from llmhalluc.train import run_train

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "llmhalluc" / "e2e.yaml"


def main():
    setup_logging(verbose=False)

    # Pass CLI args for config overrides
    # sys.argv[1:] contains user-provided overrides after accelerate consumes its args
    setup_dict = hf_cfg_setup(DEFAULT_CONFIG_PATH, cli_args=sys.argv[1:])

    run_train(setup_dict.args.hf_args)


if __name__ == "__main__":
    main()
