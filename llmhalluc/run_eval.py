"""Run lm_eval evaluations with YAML configuration support.

This script provides a YAML-driven interface to run lm_eval evaluations,
while maintaining full backwards compatibility with the original CLI interface.

Usage:
    # Using YAML config
    python llmhalluc/run_eval.py --config configs/lm_eval/run/default.yaml

    # Using CLI arguments (original style)
    python llmhalluc/run_eval.py --tasks gsm8k --model hf --model_args pretrained=model_name

    # Mixing both (CLI args override YAML)
    python llmhalluc/run_eval.py --config default.yaml --tasks gsm8k --num_fewshot 5
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

# Import the lm_eval CLI infrastructure
from lm_eval.__main__ import cli_evaluate, setup_parser


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Dictionary containing configuration parameters

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is malformed
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    return config if config is not None else {}


def merge_configs(yaml_config: Dict[str, Any], cli_args: argparse.Namespace) -> argparse.Namespace:
    """Merge YAML config with CLI arguments, with CLI taking priority.

    Args:
        yaml_config: Configuration loaded from YAML file
        cli_args: Parsed command-line arguments

    Returns:
        Merged configuration as argparse.Namespace
    """
    # Start with YAML config
    merged = yaml_config.copy()

    # Override with CLI args that were explicitly provided
    # We need to identify which CLI args were explicitly set vs. using defaults
    parser = setup_parser()
    defaults = {action.dest: action.default for action in parser._actions}

    for key, value in vars(cli_args).items():
        # Skip the 'config' argument itself
        if key == "config":
            continue

        # If the CLI value differs from the default, it was explicitly set
        # Or if the key is not in yaml_config, use the CLI value
        if key not in defaults or value != defaults.get(key) or key not in merged:
            merged[key] = value

    # Ensure include_path includes our custom task directories
    # The lm_eval __main__.py adds llmhalluc/tasks, we add configs/lm_eval/tasks
    include_paths = []

    # Start with any existing include_path from config or CLI
    if "include_path" in merged and merged["include_path"] is not None:
        if isinstance(merged["include_path"], list):
            include_paths.extend(merged["include_path"])
        else:
            include_paths.append(merged["include_path"])

    # Add our default custom task directory if not already there
    custom_task_dir = "configs/lm_eval/tasks"
    if custom_task_dir not in include_paths:
        include_paths.insert(0, custom_task_dir)

    # Set include_path: if we have multiple paths, keep as list; if single, use string
    # Note: lm_eval __main__.py will also add llmhalluc/tasks separately
    if len(include_paths) > 1:
        merged["include_path"] = include_paths
    elif len(include_paths) == 1:
        merged["include_path"] = include_paths[0]
    else:
        merged["include_path"] = custom_task_dir

    return argparse.Namespace(**merged)


def setup_extended_parser() -> argparse.ArgumentParser:
    """Setup argument parser with additional --config option.

    Returns:
        ArgumentParser with all lm_eval args plus --config
    """
    # Get the base parser from lm_eval
    parser = setup_parser()

    # Add our custom --config argument
    parser.add_argument(
        "--config",
        "-cfg",
        type=str,
        default=None,
        help="Path to YAML configuration file. CLI arguments override config file values.",
    )

    return parser


def main() -> None:
    """Main entry point for running lm_eval with YAML config support."""
    # Parse arguments
    parser = setup_extended_parser()
    args = parser.parse_args()

    # Load YAML config if provided
    if args.config is not None:
        try:
            yaml_config = load_yaml_config(args.config)
            # Merge configs (CLI args override YAML)
            merged_args = merge_configs(yaml_config, args)
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        except yaml.YAMLError as e:
            print(f"Error parsing YAML config: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        # No config file, merge empty config to ensure include_path is set
        merged_args = merge_configs({}, args)

    # Call the original lm_eval CLI function
    cli_evaluate(merged_args)


if __name__ == "__main__":
    main()
