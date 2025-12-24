"""Evaluation entry point.

Usage:
    python -m llmhalluc.run_eval
"""

from pathlib import Path
from llmhalluc.utils import e2e_cfg_setup
from llmhalluc.eval import run_eval

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "llmhalluc" / "e2e.yaml"


def main():
    setup_dict = e2e_cfg_setup(DEFAULT_CONFIG_PATH, save_cfg=True)
    run_eval(setup_dict.paths["EVAL_CONFIG_PATH"])


if __name__ == "__main__":
    main()
