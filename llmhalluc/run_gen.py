"""Batch generation entry point using vLLM.

Usage:
    python -m llmhalluc.run_gen --config configs/llmhalluc/generation.yaml
    python -m llmhalluc.run_gen --config configs/llmhalluc/grpo.yaml
"""

import sys
from pathlib import Path

from llmhalluc.gen import run_gen
from llmhalluc.hparams import hf_cfg_setup, parse_config_arg
from llmhalluc.utils import setup_logging

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "llmhalluc" / "e2e.yaml"


def main():
    setup_logging(verbose=False)

    # Parse --config argument, rest are config overrides
    config_path, cli_args = parse_config_arg(sys.argv[1:], DEFAULT_CONFIG_PATH)

    setup_dict = hf_cfg_setup(config_path, cli_args=cli_args)

    gen_args = setup_dict.args.gen_args

    run_gen(gen_args)


if __name__ == "__main__":
    main()
