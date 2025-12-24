import argparse
import os
from copy import deepcopy
import subprocess
from pathlib import Path

from llmhalluc.utils import load_config, setup_logging, e2e_cfg_setup

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


def run_eval(config_path):
    eval_config = load_config(config_path)

    args = []

    for key, val in eval_config.items():
        if isinstance(val, bool) and val:
            args.append(f"--{key}")
        elif isinstance(val, str):
            args.append(f"--{key}")
            args.append(val)
        elif isinstance(val, int):
            args.append(f"--{key}")
            args.append(str(val))
        elif isinstance(val, dict):
            sub_args = ",".join([f"{k}={v}" for k, v in val.items()])
            args.append(f"--{key}")
            args.append(sub_args)
        else:
            raise ValueError(f"Invalid value type: {type(val)}")

    cmd = ["accelerate", "launch", "-m", "lm_eval"] + args
    subprocess.run(cmd, env=deepcopy(os.environ), check=True)


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
