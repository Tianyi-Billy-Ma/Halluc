"""HuggingFace training entry point.

Usage:
    accelerate launch -m llmhalluc.run_train
    accelerate launch -m llmhalluc.run_train --config configs/llmhalluc/dpo.yaml
    accelerate launch -m llmhalluc.run_train --num_train_epochs 1
    accelerate launch -m llmhalluc.run_train --config configs/llmhalluc/dpo.yaml --learning_rate 5e-5
"""

import os
import sys
from pathlib import Path

from llmhalluc.hparams import hf_cfg_setup, parse_config_arg
from llmhalluc.train import run_train
from llmhalluc.utils import setup_logging

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "llmhalluc" / "e2e.yaml"


def main():
    setup_logging(verbose=False)

    # Parse --config argument, rest are config overrides
    config_path, cli_args = parse_config_arg(sys.argv[1:], DEFAULT_CONFIG_PATH)

    setup_dict = hf_cfg_setup(config_path, cli_args=cli_args)

if __name__ == "__main__":
    main()
