import json
import logging
from pathlib import Path

from easydict import EasyDict
from omegaconf import OmegaConf
from transformers import HfArgumentParser

from llmhalluc.utils.sys_utils import resolve_path

from .eval_args import EvaluationArguments
from .ft_args import DPOArguments, GRPOArguments, SFTArguments
from .gen_args import GenerationArguments
from .patcher import (
    patch_configs,
    patch_dpo_config,
    patch_grpo_config,
    patch_sft_config,
)
from .train_args import TrainArguments

logger = logging.getLogger(__name__)


def parse_config_arg(
    args: list[str],
    default_config_path: Path,
) -> tuple[Path, list[str]]:
    """Extract --config argument from CLI args.

    Args:
        args: List of CLI arguments
        default_config_path: Default config path to use if --config is not specified

    Returns:
        Tuple of (config_path, remaining_args)
    """
    config_path = default_config_path
    remaining_args = []

    i = 0
    while i < len(args):
        if args[i] == "--config":
            if i + 1 < len(args):
                config_path = Path(args[i + 1])
                # Handle relative paths - resolve based on parent of default config
                if not config_path.is_absolute():
                    config_path = default_config_path.parent.parent.parent / config_path
                i += 2
            else:
                raise ValueError("--config requires a path argument")
        else:
            remaining_args.append(args[i])
            i += 1

    return config_path, remaining_args


def load_config(path: str | Path) -> dict[str, any]:
    cfg_path = resolve_path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    if cfg_path.suffix == ".yaml":
        cfg = OmegaConf.load(cfg_path)
        return OmegaConf.to_container(cfg, resolve=True) or {}
    elif cfg_path.suffix == ".json":
        with open(cfg_path, "r") as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported config file type: {cfg_path.suffix}")

    logger.info(f"Loaded config from {cfg_path}")


def apply_overrides(config: dict[str, any], overrides: list[str]) -> dict[str, any]:
    if not overrides:
        return dict(config)

    base_conf = OmegaConf.create(config)
    override_conf = OmegaConf.from_dotlist(overrides)
    merged = OmegaConf.merge(base_conf, override_conf)
    return OmegaConf.to_container(merged, resolve=True)


def parse_cli_to_dotlist(args: list[str]) -> list[str]:
    """
    Convert CLI args like --num_train_epochs 1 to OmegaConf dotlist format.

    Args:
        args: CLI arguments like ['--num_train_epochs', '1', '--lr', '1e-4']

    Returns:
        Dotlist format like ['num_train_epochs=1', 'lr=1e-4']

    Example:
        >>> parse_cli_to_dotlist(['--num_train_epochs', '1', '--bf16'])
        ['num_train_epochs=1', 'bf16=true']
    """
    dotlist = []
    i = 0
    while i < len(args):
        arg = args[i]
        if arg.startswith("--"):
            key = arg[2:]  # Remove '--'
            # Check if next arg is a value or another flag
            if i + 1 < len(args) and not args[i + 1].startswith("--"):
                value = args[i + 1]
                dotlist.append(f"{key}={value}")
                i += 2
            else:
                # Boolean flag (--flag means True)
                dotlist.append(f"{key}=true")
                i += 1
        else:
            i += 1
    return dotlist


def save_config(args: dict[str, any], path: str | Path) -> None:
    cfg_path = resolve_path(path)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(args), cfg_path)
    logger.info(f"Saved config to {str(cfg_path)}")


def save_eval_cmd(args: EvaluationArguments, path: str | Path) -> None:
    cmd_path = resolve_path(path)
    cmd_path.parent.mkdir(parents=True, exist_ok=True)

    cmd_parts = [
        "lm_eval",
        "--model",
        args.model,
        "--model_args",
        args.model_args,
        "--tasks",
        args.tasks,
        "--output_path",
        args.output_path,
        "--include_path",
        args.include_path,
    ]

    if args.log_samples:
        cmd_parts.append("--log_samples")

    if args.apply_chat_template:
        cmd_parts.append("--apply_chat_template")

    if not args.disable_wandb and args.wandb_args:
        cmd_parts.append("--wandb_args")
        cmd_parts.append(args.wandb_args)

    full_cmd = " ".join(cmd_parts)

    with open(cmd_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write(full_cmd + "\n")


def save_gen_cmd(args: GenerationArguments, path: str | Path) -> None:
    """Save generation command script with detailed arguments."""
    cmd_path = resolve_path(path)
    cmd_path.parent.mkdir(parents=True, exist_ok=True)

    cmd_parts = [
        "python -m llmhalluc.run_gen",
        f"--model_name_or_path {args.model_name_or_path}",
        f"--dataset {args.dataset}",
        f"--output_path {args.output_path}",
        f"--output_filename {args.output_filename}",
        f"--max_new_tokens {args.max_new_tokens}",
        f"--temperature {args.temperature}",
        f"--top_p {args.top_p}",
        f"--batch_size {args.batch_size}",
        f"--seed {args.seed}",
    ]

    if args.adapter_name_or_path:
        cmd_parts.append(f"--adapter_name_or_path {args.adapter_name_or_path}")

    if args.tokenizer_name_or_path:
        cmd_parts.append(f"--tokenizer_name_or_path {args.tokenizer_name_or_path}")

    if args.max_samples is not None:
        cmd_parts.append(f"--max_samples {args.max_samples}")

    if args.tensor_parallel_size is not None:
        cmd_parts.append(f"--tensor_parallel_size {args.tensor_parallel_size}")

    if args.max_model_len is not None:
        cmd_parts.append(f"--max_model_len {args.max_model_len}")

    if args.do_sample:
        cmd_parts.append("--do_sample")

    if args.top_k > 0:
        cmd_parts.append(f"--top_k {args.top_k}")

    if args.num_return_sequences > 1:
        cmd_parts.append(f"--num_return_sequences {args.num_return_sequences}")

    full_cmd = " \\\n    ".join(cmd_parts)

    with open(cmd_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write(full_cmd + "\n")


def verify_train_args(args: TrainArguments) -> None:
    """Verify training arguments and check for compatibility issues."""
    # 1. Model path check
    if not args.model_name_or_path:
        raise ValueError("model_name_or_path must be specified.")

    # 2. ZeRO-3 Compatibility with LoRA/QLoRA
    finetuning_type = getattr(args, "finetuning_type", "full")
    deepspeed = getattr(args, "deepspeed", None)

    if deepspeed and finetuning_type in ["lora", "qlora"]:
        # Try to load deepspeed config
        ds_config = {}
        if isinstance(deepspeed, str):
            try:
                ds_config = load_config(deepspeed)
            except FileNotFoundError:
                pass  # Ignore if not found, explicit check later if needed
        elif isinstance(deepspeed, dict):
            ds_config = deepspeed

        # Check if zero_optimization stage is 3
        if ds_config.get("zero_optimization", {}).get("stage") == 3:
            raise ValueError(
                "DeepSpeed ZeRO-3 is currently not supported with LoRA/QLoRA in this pipeline. "
                "Please use ZeRO-2 or full fine-tuning."
            )

    # 3. QLoRA Check (load_in_4bit should align)
    load_in_4bit = getattr(args, "load_in_4bit", False)
    if finetuning_type == "qlora" and not load_in_4bit:
        args.load_in_4bit = True

    # 4. Early Stopping Check (requires eval_dataset)
    early_stopping = getattr(args, "early_stopping", False)
    eval_dataset = getattr(args, "eval_dataset", None)
    if early_stopping and not eval_dataset:
        raise ValueError(
            "early_stopping requires eval_dataset to be set. "
            "Please provide an eval_dataset or disable early_stopping."
        )

    # 5. Position IDs Check
    if getattr(args, "reset_position_ids", False) and not getattr(
        args, "train_backtrack", False
    ):
        raise ValueError(
            "reset_position_ids=True requires train_backtrack=True. "
            "Please enable train_backtrack or disable reset_position_ids."
        )


def e2e_cfg_setup(
    config_path: str,
    save_cfg: bool = True,
    cli_args: list[str] | None = None,
) -> EasyDict:
    """Setup end-to-end training config with optional CLI overrides.

    Args:
        config_path: Path to YAML config file
        save_cfg: Whether to save resolved configs
        cli_args: CLI arguments for overrides (e.g., ['--num_train_epochs', '1'])

    Returns:
        EasyDict with paths and parsed argument objects
    """
    config = load_config(config_path)

    # Apply CLI overrides to config dict BEFORE parsing into dataclasses
    if cli_args:
        overrides = parse_cli_to_dotlist(cli_args)
        config = apply_overrides(config, overrides)

    arg_dict = patch_configs(config)
    train_args = arg_dict.train_args
    merge_args = arg_dict.merge_args
    eval_args = arg_dict.eval_args
    gen_args = arg_dict.gen_args

    # Verify arguments
    verify_train_args(train_args)

    if save_cfg:
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
        save_config(
            gen_args.to_yaml(),
            gen_args.config_path,
        )

        # Save evaluation command script
        save_eval_cmd(eval_args, eval_args.config_path.parent / "eval.sh")

        # Save generation command script
        save_gen_cmd(gen_args, gen_args.config_path.parent / "gen.sh")

    output = {
        "TRAIN_CONFIG_PATH": str(train_args.config_path),
        "MERGE_CONFIG_PATH": str(merge_args.config_path),
        "EVAL_CONFIG_PATH": str(eval_args.config_path),
        "GEN_CONFIG_PATH": str(gen_args.config_path),
    }

    return EasyDict(paths=output, args=arg_dict)


def save_ft_config(args: SFTArguments | DPOArguments, path: str | Path) -> None:
    """Save fine-tuning arguments (SFT or DPO) to a YAML file."""
    cfg_path = resolve_path(path)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)

    # Use to_yaml mechanism if available (inherited from BaseArguments)
    if hasattr(args, "to_yaml"):
        # We can dump directly via save_config since to_yaml usually returns a dict
        data = args.to_yaml()
        save_config(data, cfg_path)
    else:
        # Fallback to asdict -> json-like behavior but attempting to save as yaml
        from dataclasses import asdict

        data = asdict(args)
        save_config(data, cfg_path)


def hf_cfg_setup(
    config_path: str | Path,
    save_cfg: bool = True,
    cli_args: list[str] | None = None,
) -> EasyDict:
    """Setup HuggingFace training config with optional CLI overrides.

    Args:
        config_path: Path to YAML config file
        save_cfg: Whether to save resolved configs
        cli_args: CLI arguments for overrides (e.g., ['--num_train_epochs', '1'])

    Returns:
        EasyDict with paths and parsed argument objects including hf_args
    """
    # Store original config to preserve stage-specific fields
    original_config = load_config(config_path)
    if cli_args:
        overrides = parse_cli_to_dotlist(cli_args)
        original_config = apply_overrides(original_config, overrides)

    setup_dict = e2e_cfg_setup(config_path, save_cfg=save_cfg, cli_args=cli_args)
    train_args = setup_dict.args.train_args

    hf_args = None
    stage = getattr(train_args, "stage", "sft")

    if stage == "sft":
        raw_args: dict[str, any] = patch_sft_config(train_args)
        # Merge original config to preserve SFT-specific fields
        raw_args = {**original_config, **raw_args}
        hf_args, *_ = HfArgumentParser(SFTArguments).parse_dict(
            raw_args, allow_extra_keys=True
        )

        if save_cfg:
            save_ft_config(hf_args, hf_args.config_path)

    elif stage == "dpo":
        raw_args: dict[str, any] = patch_dpo_config(train_args)
        # Merge original config to preserve DPO-specific fields
        raw_args = {**original_config, **raw_args}
        hf_args, *_ = HfArgumentParser(DPOArguments).parse_dict(
            raw_args, allow_extra_keys=True
        )

        if save_cfg:
            save_ft_config(hf_args, hf_args.config_path)

    elif stage == "grpo":
        raw_args: dict[str, any] = patch_grpo_config(train_args)
        # Merge original config to preserve GRPO-specific fields like num_generations
        raw_args = {**original_config, **raw_args}
        hf_args, *_ = HfArgumentParser(GRPOArguments).parse_dict(
            raw_args, allow_extra_keys=True
        )

        if save_cfg:
            save_ft_config(hf_args, hf_args.config_path)

    else:
        raise ValueError(f"Unsupported stage: {stage}")

    setup_dict.args.hf_args = hf_args
    return setup_dict


def gen_cfg_setup(
    config_path: str | Path,
    cli_args: list[str] | None = None,
    save_cfg: bool = True,
) -> GenerationArguments:
    config = load_config(config_path)

    if cli_args:
        overrides = parse_cli_to_dotlist(cli_args)
        config = apply_overrides(config, overrides)

    gen_args, *_ = HfArgumentParser(GenerationArguments).parse_dict(
        config, allow_extra_keys=True
    )

    if save_cfg:
        save_config(gen_args.to_yaml(), gen_args.config_path)

    return gen_args
