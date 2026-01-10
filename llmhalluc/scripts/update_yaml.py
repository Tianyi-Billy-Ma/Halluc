#!/usr/bin/env python3
"""
YAML Configuration Updater Script

This script updates YAML configuration files with additional arguments,
similar to how command-line arguments override YAML values in LLaMA-Factory.

Usage:
    python update_yaml.py <input_yaml> <output_yaml> [key=value ...]

Examples:
    # Update train.yaml with model parameters
    python update_yaml.py configs/llamafactory/train.yaml configs/llamafactory/train_updated.yaml \
        model_name_or_path=Qwen/Qwen3-4B-Instruct-2507 \
        enable_thinking=false \
        output_dir=./outputs/test \
        stage=sft \
        finetuning_type=lora

    # Update merge.yaml with merge parameters
    python update_yaml.py configs/llamafactory/merge.yaml configs/llamafactory/merge_updated.yaml \
        model_name_or_path=Qwen/Qwen3-4B-Instruct-2507 \
        adapter_name_or_path=./outputs/test \
        template=qwen3 \
        export_dir=./models/test_model
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Union

import yaml


def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration.

    Args:
        verbose: If True, set logging level to DEBUG, otherwise INFO.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_key_value_pairs(args: List[str]) -> Dict[str, Any]:
    """Parse key=value pairs from command line arguments.

    Args:
        args: List of key=value strings.

    Returns:
        Dictionary with parsed key-value pairs.

    Raises:
        ValueError: If any argument is not in key=value format.
    """
    parsed_args = {}

    for arg in args:
        if "=" not in arg:
            raise ValueError(f"Argument '{arg}' is not in key=value format")

        key, value = arg.split("=", 1)  # Split only on first '='

        # Try to convert value to appropriate type
        parsed_value = convert_value_type(value)
        parsed_args[key] = parsed_value

        logging.debug(
            f"Parsed argument: {key} = {parsed_value} (type: {type(parsed_value).__name__})"
        )

    return parsed_args


def convert_value_type(value: str) -> Union[str, int, float, bool, None]:
    """Convert string value to appropriate Python type.

    Args:
        value: String value to convert.

    Returns:
        Converted value with appropriate type.
    """
    # Handle None/null values
    if value.lower() in ("null", "none"):
        return None

    # Handle boolean values
    if value.lower() in ("true", "false"):
        return value.lower() == "true"

    # Try to convert to int
    try:
        return int(value)
    except ValueError:
        pass

    # Try to convert to float
    try:
        return float(value)
    except ValueError:
        pass

    # Return as string if no other conversion works
    return value


def load_yaml_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Load YAML file and return its contents.

    Args:
        file_path: Path to the YAML file.

    Returns:
        Dictionary containing YAML contents.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        yaml.YAMLError: If the file contains invalid YAML.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"YAML file not found: {file_path}")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = yaml.safe_load(f)
            logging.info(f"Successfully loaded YAML file: {file_path}")
            return content or {}  # Return empty dict if file is empty
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Invalid YAML in file {file_path}: {e}")


def save_yaml_file(data: Dict[str, Any], file_path: Union[str, Path]) -> None:
    """Save dictionary to YAML file.

    Args:
        data: Dictionary to save.
        file_path: Path where to save the YAML file.
    """
    file_path = Path(file_path)

    # Create parent directories if they don't exist
    file_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False, indent=2)
        logging.info(f"Successfully saved YAML file: {file_path}")
    except Exception as e:
        raise IOError(f"Failed to save YAML file {file_path}: {e}")


def update_yaml_config(
    input_file: Union[str, Path],
    output_file: Union[str, Path],
    updates: Dict[str, Any],
    preserve_comments: bool = True,
) -> None:
    """Update YAML configuration file with new values.

    Args:
        input_file: Path to input YAML file.
        output_file: Path to output YAML file.
        updates: Dictionary of key-value pairs to update.
        preserve_comments: Whether to preserve comments (currently not implemented).
    """
    # Load existing configuration
    config = load_yaml_file(input_file)

    # Apply updates
    for key, value in updates.items():
        old_value = config.get(key)
        config[key] = value

        if old_value is not None:
            logging.info(f"Updated '{key}': {old_value} -> {value}")
        else:
            logging.info(f"Added '{key}': {value}")

    save_yaml_file(config, output_file)

    logging.info(f"Configuration updated successfully. Output saved to: {output_file}")


def main() -> None:
    """Main function to handle command-line interface."""
    parser = argparse.ArgumentParser(
        description="Update YAML configuration files with additional arguments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--input_yaml", type=str, help="Path to input YAML file")

    parser.add_argument("--output_yaml", type=str, help="Path to output YAML file")

    parser.add_argument(
        "updates", nargs="*", help="Key=value pairs to update in the YAML file"
    )

    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be updated without making changes",
    )

    args = parser.parse_args()

    # Set up logging
    setup_logging(args.verbose)

    try:
        # Parse key-value pairs
        updates = parse_key_value_pairs(args.updates)

        if not updates:
            logging.warning(
                "No updates specified. Output file will be identical to input file."
            )

        if args.dry_run:
            # Load and display what would be changed
            config = load_yaml_file(args.input_yaml)
            print(f"\nDry run - would update {args.input_yaml} -> {args.output_yaml}")
            print("Changes:")
            for key, value in updates.items():
                old_value = config.get(key)
                if old_value is not None:
                    print(f"  {key}: {old_value} -> {value}")
                else:
                    print(f"  {key}: (new) -> {value}")
            return

        # Perform the update
        update_yaml_config(args.input_yaml, args.output_yaml, updates)

    except (FileNotFoundError, yaml.YAMLError, ValueError, IOError) as e:
        logging.error(f"Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logging.info("Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
