import sys
from pathlib import Path

from llmhalluc.gen import run_gen
from llmhalluc.hparams import gen_cfg_setup, parse_config_arg
from llmhalluc.utils import setup_logging

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "llmhalluc" / "gen.yaml"


def main():
    setup_logging(verbose=False)

    config_path, cli_args = parse_config_arg(sys.argv[1:], DEFAULT_CONFIG_PATH)

    gen_args = gen_cfg_setup(config_path, cli_args=cli_args)

    run_gen(gen_args)


if __name__ == "__main__":
    main()
