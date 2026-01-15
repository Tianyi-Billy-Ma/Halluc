"""Evaluation entry point.

Usage:
    python -m llmhalluc.run_eval
"""

from pathlib import Path

from llmhalluc.eval import run_eval
from llmhalluc.hparams import e2e_cfg_setup, parse_config_args
from llmhalluc.utils import setup_logging

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "llmhalluc" / "e2e.yaml"


def main():
    setup_logging(verbose=False)
    
    config_path, cli_args = parse_config_args(sys.args[1:],DEFAULT_CONFIG_PATH)

    setup_dict = e2e_cfg_setup(DEFAULT_CONFIG_PATH, save_cfg=True)

    wandb_project = getattr(hf_args, "wandb_project", None)
    if wandb_project:
        os.environ["WANDB_PROJECT"] = wandb_project

    run_eval(setup_dict.paths["EVAL_CONFIG_PATH"])


if __name__ == "__main__":
    main()
