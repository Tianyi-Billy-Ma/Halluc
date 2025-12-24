import argparse
import os
from copy import deepcopy
import subprocess
from pathlib import Path

from llmhalluc.utils import setup_logging, e2e_cfg_setup
from llmhalluc.eval import run_eval

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "llmhalluc" / "e2e.yaml"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare train/merge configs")
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to the high-level e2e YAML (default: %(default)s)",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a config entry (can be used multiple times)",
    )
    parser.add_argument(
        "--format",
        choices=("json", "shell", "else"),
        default="json",
        help="Output format for resolved values (default: %(default)s)",
    )
    parser.add_argument(
        "--do-train",
        type=bool,
        default=True,
        metavar="BOOL",
        help="Generate the train config (default: true)",
    )
    parser.add_argument(
        "--do-merge",
        type=bool,
        default=True,
        metavar="BOOL",
        help="Generate the merge config (default: true)",
    )
    parser.add_argument(
        "--do-eval",
        type=bool,
        default=True,
        metavar="BOOL",
        help="Generate the eval config (default: true)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args(argv)


def run_llamafactory(mode, config_path, additional: list[str] | None = None):
    assert mode in ["train", "export"]

    cmd = ["llamafactory-cli", mode, config_path] + (additional if additional else [])
    subprocess.run(cmd, env=deepcopy(os.environ), check=True)


def main():
    argv = ["--format", "else"]

    args = parse_args(argv)
    setup_logging(verbose=args.verbose)
    setup_dict = e2e_cfg_setup(args)
    print(setup_dict)
    run_llamafactory("train", setup_dict["TRAIN_CONFIG_PATH"])
    run_llamafactory("export", setup_dict["MERGE_CONFIG_PATH"])
    run_eval(setup_dict["EVAL_CONFIG_PATH"])


if __name__ == "__main__":
    main()
