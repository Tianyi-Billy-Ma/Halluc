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
import logging
import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional

from omegaconf import OmegaConf

from llmhalluc.utils.type_utils import str2bool

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "llmhalluc" / "e2e.yaml"


@dataclass
class ExperimentContext:
    """Holds resolved experiment metadata and derived paths."""

    exp_name: str
    method_name: str
    model_name_or_path: str
    enable_thinking: bool
    lf_template: str
    stage: str
    finetuning_type: str
    train_dataset: str
    eval_dataset: str
    eval_task_name: str
    eval_model: str
    wandb_project: str
    seed: int
    ddp: int
    hf_username: Optional[str]
    train_config_template: Path
    merge_config_template: Path
    eval_config_template: Path
    special_tokens_template: Path
    model_dir: Path
    output_dir: Path
    model_abbr: str = field(init=False)
    wandb_name: str = field(init=False)
    model_path: Path = field(init=False)
    train_output_path: Path = field(init=False)
    train_config_path: Path = field(init=False)
    merge_config_path: Path = field(init=False)
    eval_output_path: Path = field(init=False)
    eval_config_path: Path = field(init=False)
    special_tokens_path: Path = field(init=False)

    ### DPO
    pref_loss: str = "sigmoid"
    pref_beta: float = 0.1

    def __post_init__(self) -> None:
        model_basename = Path(self.model_name_or_path).name.lower()
        self.model_abbr = model_basename
        method = self.method_name or self.stage.lower()
        self.method_name = method
        self.wandb_name = f"{model_basename}_{self.exp_name}_{method}"

        self.train_output_path = (
            self.output_dir / self.model_abbr / self.exp_name / method / "train"
        )
        base_dir = self.output_dir / self.model_abbr / self.exp_name / method
        self.train_config_path = base_dir / "train_config.yaml"
        self.merge_config_path = base_dir / "merge_config.yaml"
        eval_dir = base_dir / "eval"
        self.eval_output_path = eval_dir / "results.json"
        self.eval_config_path = base_dir / "eval_config.yaml"
        self.special_tokens_path = base_dir / "special_token_config.yaml"

        if method == "vanilla" or not Path(self.model_name_or_path).exists():
            self.model_path = Path(self.model_name_or_path)
        else:
            self.model_path = self.model_dir / self.wandb_name


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
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
        "--dry-run",
        action="store_true",
        help="Resolve values without writing train/merge configs",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args(argv)


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")


def load_config(path: str) -> Dict[str, Any]:
    cfg_path = resolve_path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    cfg = OmegaConf.load(cfg_path)
    return OmegaConf.to_container(cfg, resolve=True) or {}


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = REPO_ROOT / path_str
    return path.expanduser().resolve()


def apply_overrides(
    config: MutableMapping[str, Any], overrides: List[str]
) -> Dict[str, Any]:
    if not overrides:
        return dict(config)

    base_conf = OmegaConf.create(config)
    override_conf = OmegaConf.from_dotlist(overrides)
    merged = OmegaConf.merge(base_conf, override_conf)
    return OmegaConf.to_container(merged, resolve=True)


def ensure_required(config: Mapping[str, Any], key: str) -> Any:
    if key not in config or config[key] is None:
        raise ValueError(f"Missing required config key: '{key}'")
    return config[key]


def bool_from(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() == "true"
    return bool(value)


def int_from(value: Any, key: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"Config key '{key}' must be an integer, got {value!r}"
        ) from None


def build_context(config: Mapping[str, Any]) -> ExperimentContext:
    eval_dataset = config.get("eval_dataset") or config.get("train_dataset")
    context = ExperimentContext(
        exp_name=str(ensure_required(config, "exp_name")),
        method_name=str(config.get("method_name") or ""),
        model_name_or_path=str(ensure_required(config, "model_name_or_path")),
        enable_thinking=bool_from(config.get("enable_thinking", False)),
        lf_template=str(ensure_required(config, "lf_template")),
        stage=str(ensure_required(config, "stage")),
        finetuning_type=str(ensure_required(config, "finetuning_type")),
        train_dataset=str(ensure_required(config, "train_dataset")),
        eval_dataset=str(eval_dataset),
        eval_task_name=str(ensure_required(config, "eval_task_name")),
        eval_model=str(ensure_required(config, "eval_model")),
        wandb_project=str(ensure_required(config, "wandb_project")),
        seed=int_from(ensure_required(config, "seed"), "seed"),
        ddp=int_from(ensure_required(config, "ddp"), "ddp"),
        hf_username=(str(config["hf_username"]) if config.get("hf_username") else None),
        train_config_template=resolve_path(
            str(ensure_required(config, "train_config"))
        ),
        merge_config_template=resolve_path(
            str(ensure_required(config, "merge_config"))
        ),
        eval_config_template=resolve_path(str(ensure_required(config, "eval_config"))),
        special_tokens_template=resolve_path(
            str(ensure_required(config, "new_special_tokens_config"))
        ),
        model_dir=resolve_path(str(ensure_required(config, "model_dir"))),
        output_dir=resolve_path(str(ensure_required(config, "output_dir"))),
    )
    return context


def ensure_parent_dirs(context: ExperimentContext, plan: Dict[str, bool]) -> None:
    if plan.get("train"):
        context.train_output_path.mkdir(parents=True, exist_ok=True)
        context.train_config_path.parent.mkdir(parents=True, exist_ok=True)
    if plan.get("merge"):
        context.merge_config_path.parent.mkdir(parents=True, exist_ok=True)
    if plan.get("eval"):
        context.eval_output_path.parent.mkdir(parents=True, exist_ok=True)
        context.eval_config_path.parent.mkdir(parents=True, exist_ok=True)
    if plan.get("train") or plan.get("merge"):
        context.special_tokens_path.parent.mkdir(parents=True, exist_ok=True)


def validate_templates(context: ExperimentContext, plan: Dict[str, bool]) -> None:
    checks = {
        "train": ("train_config", context.train_config_template),
        "merge": ("merge_config", context.merge_config_template),
        "eval": ("eval_config", context.eval_config_template),
    }
    for stage, (label, path) in checks.items():
        if plan.get(stage) and not path.exists():
            raise FileNotFoundError(f"{label} template not found: {path}")


def write_train_config(context: ExperimentContext) -> None:
    updates = {
        "model_name_or_path": context.model_name_or_path,
        "enable_thinking": context.enable_thinking,
        "template": context.lf_template,
        "output_dir": str(context.train_output_path),
        "stage": context.stage,
        "finetuning_type": context.finetuning_type,
        "run_name": context.wandb_name,
        "dataset": context.train_dataset,
        "eval_dataset": context.eval_dataset,
    }
    updates = patch_train_config(updates, context)
    logging.info("Writing train config -> %s", context.train_config_path)
    write_yaml_with_updates(
        context.train_config_template, context.train_config_path, updates
    )


def write_merge_config(context: ExperimentContext) -> None:
    updates = {
        "model_name_or_path": context.model_name_or_path,
        "adapter_name_or_path": str(context.train_output_path),
        "template": context.lf_template,
        "export_dir": str(context.model_path),
    }
    if context.hf_username:
        updates["export_hub_model_id"] = f"{context.hf_username}/{context.wandb_name}"

    updates = patch_merge_config(updates, context)
    logging.info("Writing merge config -> %s", context.merge_config_path)
    write_yaml_with_updates(
        context.merge_config_template, context.merge_config_path, updates
    )


def write_eval_config(context: ExperimentContext) -> None:
    enable_thinking = "true" if context.enable_thinking else "false"
    model_args = f"pretrained={context.model_path},enable_thinking={enable_thinking}"
    updates = {
        "model": context.eval_model,
        "model_args": model_args,
        "tasks": context.eval_task_name,
        "output_path": str(context.eval_output_path),
        "seed": context.seed,
        "wandb_args": f"project={context.wandb_project},name={context.wandb_name}",
    }
    updates = patch_eval_config(updates, context)
    logging.info("Writing eval config -> %s", context.eval_config_path)
    write_yaml_with_updates(
        context.eval_config_template, context.eval_config_path, updates
    )


def patch_train_config(
    updates: Dict[str, Any], context: ExperimentContext
) -> Dict[str, Any]:
    """Hook for experiment-specific train config tweaks."""
    ensure_special_token_config(context)
    updates.setdefault("new_special_tokens_config", str(context.special_tokens_path))

    template_name = updates.get("template", "").lower()
    if "llama3" in template_name:
        updates["replace_text"] = {"<|BACKTRACK|>": "<|reserved_special_token_0|>"}
        updates["force_init_embeddings"] = False
    elif "qwen3" in template_name:
        updates["force_init_embeddings"] = True

    return updates


def patch_merge_config(
    updates: Dict[str, Any], context: ExperimentContext
) -> Dict[str, Any]:
    """Hook for experiment-specific merge config tweaks."""
    ensure_special_token_config(context)
    updates.setdefault("new_special_tokens_config", str(context.special_tokens_path))
    return updates


def patch_eval_config(
    updates: Dict[str, Any], context: ExperimentContext
) -> Dict[str, Any]:
    """Hook for experiment-specific eval config tweaks."""
    return updates


def ensure_special_token_config(context: ExperimentContext) -> None:
    if getattr(context, "_special_tokens_written", False):
        return
    write_special_token_config(context)
    context._special_tokens_written = True


def write_special_token_config(context: ExperimentContext) -> None:
    cfg = OmegaConf.load(context.special_tokens_template)
    data = OmegaConf.to_container(cfg, resolve=True)

    if not isinstance(data, dict):
        raise ValueError(
            f"Special token config must map template names to token sets: {context.special_tokens_template}"
        )

    profile = resolve_token_profile(context.lf_template, data)
    tokens = data.get(profile)

    if not isinstance(tokens, dict) or not tokens:
        raise ValueError(
            f"Special token profile '{profile}' must contain token:description pairs."
        )

    logging.info(
        "Writing special token config (%s tokens) -> %s",
        len(tokens),
        context.special_tokens_path,
    )
    OmegaConf.save(OmegaConf.create(tokens), context.special_tokens_path)


def resolve_token_profile(template_name: str, profiles: Mapping[str, Any]) -> str:
    name = template_name.lower()
    candidates = [name]

    for sep in ("_", "-"):
        candidates.append(name.split(sep)[0])

    if "llama" in name:
        candidates.append("llama")
    if "qwen" in name:
        candidates.append("qwen")

    candidates.append("default")

    for key in candidates:
        if key and key in profiles:
            return key

    raise ValueError(
        f"Unable to find special token profile for template '{template_name}'. "
        f"Available profiles: {', '.join(profiles.keys())}"
    )


def write_yaml_with_updates(
    template_path: Path, output_path: Path, updates: Mapping[str, Any]
) -> None:
    conf = OmegaConf.load(template_path)
    for key, value in updates.items():
        conf[key] = value
    output_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(conf, output_path)


def build_output(context: ExperimentContext, plan: Dict[str, bool]) -> Dict[str, Any]:
    return {
        "TRAIN_CONFIG_PATH": str(context.train_config_path)
        if plan.get("train")
        else "",
        "MERGE_CONFIG_PATH": str(context.merge_config_path)
        if plan.get("merge")
        else "",
        "EVAL_CONFIG_PATH": str(context.eval_config_path) if plan.get("eval") else "",
        "SPECIAL_TOKEN_CONFIG_PATH": str(context.special_tokens_path)
        if plan.get("train") or plan.get("merge")
        else "",
    }


def format_shell(env_map: Mapping[str, Any]) -> str:
    lines: List[str] = []
    for key in sorted(env_map.keys()):
        value = env_map[key]
        if isinstance(value, bool):
            rendered = "true" if value else "false"
        else:
            rendered = "" if value is None else str(value)
        lines.append(f"{key}={shlex.quote(rendered)}")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    setup_logging(args.verbose)

    config = load_config(args.config)
    config = apply_overrides(config, args.override)

    context = build_context(config)
    plan = {"train": args.do_train, "merge": args.do_merge, "eval": args.do_eval}

    if not any(plan.values()):
        raise ValueError("At least one stage must be enabled to generate configs.")

    validate_templates(context, plan)
    ensure_parent_dirs(context, plan)

    if not args.dry_run:
        if plan["train"]:
            write_train_config(context)
        if plan["merge"]:
            write_merge_config(context)
        if plan["eval"]:
            write_eval_config(context)
    else:
        logging.info("Dry-run mode: skipping file writes")

    output = build_output(context, plan)

    if args.format == "json":
        print(json.dumps(output, indent=2))
    else:
        print(format_shell(output))

    return 0


if __name__ == "__main__":
    sys.exit(main())
