from pathlib import Path
from easydict import EasyDict

from llmhalluc.utils import setup_logging, hf_cfg_setup
from llmhalluc.train import run_train

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "llmhalluc" / "e2e.yaml"


def main():
    args = EasyDict(
        config=DEFAULT_CONFIG_PATH,
    )
    setup_logging(verbose=True)
    hf_args = hf_cfg_setup(args)
    run_train(hf_args)
