from pathlib import Path

from llmhalluc.utils import setup_logging, hf_cfg_setup
from llmhalluc.train import run_train

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "llmhalluc" / "e2e.yaml"


def main():
    setup_logging(verbose=True)
    hf_args = hf_cfg_setup()
    run_train(hf_args)
