"""HuggingFace training entry point.

Usage:
    accelerate launch -m llmhalluc.hf_train
"""

from pathlib import Path

from llmhalluc.utils import setup_logging, hf_cfg_setup
from llmhalluc.train import run_train

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "llmhalluc" / "e2e.yaml"


def main():
    setup_logging(verbose=True)
    setup_dict = hf_cfg_setup(DEFAULT_CONFIG_PATH)
    run_train(setup_dict.args.hf_args)


if __name__ == "__main__":
    main()
