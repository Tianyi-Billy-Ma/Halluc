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

from llmhalluc.train import run_train
from llmhalluc.utils import hf_cfg_setup, setup_logging

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "llmhalluc" / "e2e.yaml"


def parse_config_arg(args: list[str]) -> tuple[Path, list[str]]:
    """Extract --config argument from CLI args.

    Args:
        args: List of CLI arguments

    Returns:
        Tuple of (config_path, remaining_args)
    """
    config_path = DEFAULT_CONFIG_PATH
    remaining_args = []

    i = 0
    while i < len(args):
        if args[i] == "--config":
            if i + 1 < len(args):
                config_path = Path(args[i + 1])
                # Handle relative paths
                if not config_path.is_absolute():
                    config_path = REPO_ROOT / config_path
                i += 2
            else:
                raise ValueError("--config requires a path argument")
        else:
            remaining_args.append(args[i])
            i += 1

    return config_path, remaining_args


def main():
    setup_logging(verbose=False)

    # Parse --config argument, rest are config overrides
    config_path, cli_args = parse_config_arg(sys.argv[1:])

    setup_dict = hf_cfg_setup(config_path, cli_args=cli_args)

    # Set WANDB_PROJECT environment variable if specified in config
    # HuggingFace Trainer uses this env var to determine the WandB project
    hf_args = setup_dict.args.hf_args
    wandb_project = getattr(hf_args, "wandb_project", None)
    if wandb_project:
        os.environ["WANDB_PROJECT"] = wandb_project

    run_train(hf_args)


if __name__ == "__main__":
    main()
