#!/usr/bin/env python3
"""
Utility to materialize per-run train/merge configs from a single experiment YAML.

Typical usage inside bash scripts:

    eval "$(
        python -m llmhalluc.scripts.e2e_setup \
            --format shell \
            --config configs/llmhalluc/e2e.yaml
    )"

This prints shell-safe KEY=VALUE exports (TRAIN_CONFIG_PATH, MERGE_CONFIG_PATH,
EVAL_CONFIG_PATH) and writes the concrete config files so callers can invoke:

    llamafactory-cli train "$TRAIN_CONFIG_PATH"
    llamafactory-cli export "$MERGE_CONFIG_PATH"

The script also supports inline overrides: --override exp_name=myexp.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

from transformers import HfArgumentParser

from llmhalluc.extras.constant import SPECIAL_TOKEN_MAPPING
from llmhalluc.hparams.eval_args import EvaluationArguments
from llmhalluc.hparams.merge_args import MergeArguments
from llmhalluc.hparams.train_args import TrainArguments
from llmhalluc.utils.log_utils import setup_logging
from llmhalluc.utils.sys_utils import (
    apply_overrides,
    load_config,
    save_config,
)
from llmhalluc.utils.type_utils import str2bool

REPO_ROOT = Path(__file__).resolve().parents[2]
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
        type=str2bool,
        default=True,
        metavar="BOOL",
        help="Generate the train config (default: true)",
    )
    parser.add_argument(
        "--do-merge",
        type=str2bool,
        default=True,
        metavar="BOOL",
        help="Generate the merge config (default: true)",
    )
    parser.add_argument(
        "--do-eval",
        type=str2bool,
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


def patch_train_config(
    args: TrainArguments,
    plan: dict[str, bool] | None = None,
) -> TrainArguments:
    """Hook for experiment-specific train config tweaks.

    Auto-sets special token config based on template when init_special_tokens=True
    but new_special_tokens_config is not provided.

    Args:
        args: TrainArguments to patch
        plan: Optional plan dict (unused, kept for compatibility)

    Returns:
        Patched TrainArguments
    """
    if args.init_special_tokens:
        template = args.template.lower() if args.template else ""

        # Auto-set new_special_tokens_config if not provided
        if not args.new_special_tokens_config:
            if "llama3" in template:
                args.new_special_tokens_config = SPECIAL_TOKEN_MAPPING["llama3"]
            elif "qwen3" in template:
                args.new_special_tokens_config = SPECIAL_TOKEN_MAPPING["qwen3"]
            else:
                raise ValueError(
                    f"init_special_tokens=True but template '{args.template}' not supported. "
                    "Please provide new_special_tokens_config manually or use llama3/qwen3 template."
                )

        # Auto-set replace_text if not provided (for dataset preprocessing)
        # LLaMA3 uses reserved token, so we need to map <|BACKTRACK|> to it
        if not args.replace_text:
            if "llama3" in template:
                args.replace_text = {"<|BACKTRACK|>": "<|reserved_special_token_0|>"}
            # qwen3 can use <|BACKTRACK|> directly, no replace needed

    return args


def patch_merge_config(
    args: MergeArguments,
    additional_args: TrainArguments | None = None,
    plan: dict[str, bool] | None = None,
) -> MergeArguments:
    """Hook for experiment-specific merge config tweaks.

    Args:
        args: MergeArguments to patch
        additional_args: TrainArguments for referencing training config
        plan: Optional plan dict (unused, kept for compatibility)

    Returns:
        Patched MergeArguments
    """
    if additional_args:
        if additional_args.init_special_tokens:
            args.init_special_tokens = additional_args.init_special_tokens
            args.new_special_tokens_config = additional_args.new_special_tokens_config
            args.replace_text = additional_args.replace_text

        args.adapter_name_or_path = additional_args.output_dir
    return args


def patch_eval_config(
    args: EvaluationArguments,
    additional_args: TrainArguments | None = None,
    plan: dict[str, bool] | None = None,
) -> EvaluationArguments:
    """Hook for experiment-specific eval config tweaks.

    Args:
        args: EvaluationArguments to patch
        additional_args: TrainArguments for referencing training config
        plan: Optional plan dict (unused, kept for compatibility)

    Returns:
        Patched EvaluationArguments
    """
    return args


def format_shell(env_map: dict[str, any]) -> str:
    lines: list[str] = []
    for key in sorted(env_map.keys()):
        value = env_map[key]
        if isinstance(value, bool):
            rendered = "true" if value else "false"
        else:
            rendered = "" if value is None else str(value)
        lines.append(f"{key}={shlex.quote(rendered)}")
    return "\n".join(lines)


def build_configs(config: dict[str, any], plan: dict[str, bool]) -> tuple:
    train_args, *_ = HfArgumentParser([TrainArguments]).parse_dict(
        config, allow_extra_keys=True
    )
    train_args = patch_train_config(args=train_args, plan=plan)
    merge_args, *_ = HfArgumentParser((MergeArguments,)).parse_dict(
        config,
        allow_extra_keys=True,
    )
    merge_args = patch_merge_config(
        args=merge_args,
        additional_args=train_args,
        plan=plan,
    )
    eval_args, *_ = HfArgumentParser((EvaluationArguments,)).parse_dict(
        {
            "model_path": merge_args.export_dir
            if train_args.finetuning_type in ["lora"]
            else train_args.output_dir,
            "run_name": train_args.run_name,
            "exp_path": train_args.exp_path,
            **config,
        },
        allow_extra_keys=True,
    )

    eval_args = patch_eval_config(
        args=eval_args,
        additional_args=train_args,
        plan=plan,
    )
    return train_args, merge_args, eval_args


def e2e_setup(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.verbose)

    config = load_config(args.config)
    config = apply_overrides(config, args.override)

    plan = {"train": args.do_train, "merge": args.do_merge, "eval": args.do_eval}

    if not any(plan.values()):
        raise ValueError("At least one stage must be enabled to generate configs.")

    train_args, merge_args, eval_args = build_configs(config, plan)

    save_config(
        train_args.to_yaml(),
        train_args.config_path,
    )
    save_config(
        merge_args.to_yaml(),
        merge_args.config_path,
    )
    save_config(
        eval_args.to_yaml(),
        eval_args.config_path,
    )

    output = {
        "TRAIN_CONFIG_PATH": str(train_args.config_path),
        "MERGE_CONFIG_PATH": str(merge_args.config_path),
        "EVAL_CONFIG_PATH": str(eval_args.config_path),
    }

    if args.format == "json":
        print(json.dumps(output, indent=2))

    elif args.format == "shell":
        print(format_shell(output))
    else:
        return output
    return 0


if __name__ == "__main__":
    sys.exit(e2e_setup())
