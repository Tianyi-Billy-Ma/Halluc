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

from llmhalluc.utils.type_utils import str2bool
from llmhalluc.utils.log_utils import setup_logging
from llmhalluc.utils.sys_utils import (
    load_config,
    apply_overrides,
    save_config,
)
from llmhalluc.hparams.base_args import BaseArguments
from llmhalluc.hparams.train_args import TrainArguments
from llmhalluc.hparams.merge_args import MergeArguments
from llmhalluc.hparams.eval_args import EvaluationArguments
from llmhalluc.extras.constant import SPECIAL_TOKEN_MAPPING

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
        choices=("json", "shell"),
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
    additional_args: BaseArguments | None = None,
    plan: dict[str, bool] = {},
) -> dict[str, any]:
    """Hook for experiment-specific train config tweaks."""

    template = args.template
    if args.init_special_tokens:
        save_config(SPECIAL_TOKEN_MAPPING[template], args.new_special_tokens_config)

    if "llama3" in template:
        args.replace_text = {"<|BACKTRACK|>": "<|reserved_special_token_0|>"}
        args.force_init_embeddings = True
    elif "qwen3" in template:
        args.force_init_embeddings = True
    return args


def patch_merge_config(
    args: TrainArguments,
    additional_args: BaseArguments | None = None,
    plan: dict[str, bool] = {},
) -> dict[str, any]:
    """Hook for experiment-specific merge config tweaks."""
    if additional_args.init_special_tokens:
        args.new_special_tokens_config = additional_args.new_special_tokens_config
        args.init_special_tokens = additional_args.init_special_tokens
        args.force_init_embeddings = additional_args.force_init_embeddings

    args.adapter_name_or_path = additional_args.output_dir
    return args


def patch_eval_config(
    args: EvaluationArguments,
    additional_args: BaseArguments | None = None,
    plan: dict[str, bool] = {},
) -> dict[str, any]:
    """Hook for experiment-specific eval config tweaks."""
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


def build_configs(config: dict[str, any], plan: dict[str, bool]) -> dict[str, any]:
    train_args, *_ = HfArgumentParser([TrainArguments]).parse_dict(
        config, allow_extra_keys=True
    )
    train_args = patch_train_config(args=train_args, additional_args=None, plan=plan)
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
            "model_path": merge_args.export_dir,
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


def main(argv: list[str] | None = None) -> int:
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
        "SPECIAL_TOKEN_CONFIG_PATH": str(train_args.new_special_tokens_config or ""),
    }

    if args.format == "json":
        print(json.dumps(output, indent=2))
    else:
        print(format_shell(output))

    return 0


if __name__ == "__main__":
    sys.exit(main())
