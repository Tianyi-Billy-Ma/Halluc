"""Config patching utilities for training arguments."""

from pathlib import Path

from easydict import EasyDict
from transformers import HfArgumentParser

from llmhalluc.extras.constant import SPECIAL_TOKEN_MAPPING

from .base_args import BaseArguments
from .eval_args import EvaluationArguments
from .merge_args import MergeArguments
from .train_args import TrainArguments


def patch_train_config(args: TrainArguments) -> TrainArguments:
    """Hook for experiment-specific train config tweaks.

    Auto-sets special token config based on template when init_special_tokens=True
    but new_special_tokens_config is not provided.

    Args:
        args: TrainArguments to patch

    Returns:
        Patched TrainArguments (same object, modified in place)
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
) -> MergeArguments:
    """Hook for experiment-specific merge config tweaks.

    Args:
        args: MergeArguments to patch
        additional_args: TrainArguments for referencing training config

    Returns:
        Patched MergeArguments
    """
    if additional_args:
        # Copy special token config if enabled
        if additional_args.init_special_tokens:
            args.init_special_tokens = additional_args.init_special_tokens
            args.new_special_tokens_config = additional_args.new_special_tokens_config
            args.replace_text = additional_args.replace_text

        args.adapter_name_or_path = additional_args.output_dir

    return args


def patch_eval_config(
    args: EvaluationArguments,
    additional_args: TrainArguments | None = None,
) -> EvaluationArguments:
    """Hook for experiment-specific eval config tweaks.

    Args:
        args: EvaluationArguments to patch
        additional_args: TrainArguments for referencing training config

    Returns:
        Patched EvaluationArguments
    """
    if additional_args:
        init_special_tokens = getattr(additional_args, "init_special_tokens", False)

        if init_special_tokens:
            # Here, we use the trained model's tokenizer
            args.tokenizer_name_or_path = additional_args.output_dir
            args._update_model_args()

    return args


def patch_sft_config(args) -> dict[str, any]:
    """Patch SFT config for HuggingFace training.

    Args:
        args: TrainArguments or BaseArguments instance

    Returns:
        Dictionary with resolved config values
    """
    if isinstance(args, BaseArguments):
        arg_dict = args.to_yaml(exclude=False)
    else:
        arg_dict = dict(args)

    # Resolve tokenizer path (default to model path if not specified)
    if not arg_dict.get("tokenizer_name_or_path"):
        arg_dict["tokenizer_name_or_path"] = arg_dict.get("model_name_or_path")

    # Handle config_path (may be Path or str from to_yaml() serialization)
    if arg_dict.get("config_path"):
        config_path = Path(arg_dict["config_path"])
        arg_dict["config_path"] = str(config_path.parent / "sft_config.yaml")
    return arg_dict


def patch_dpo_config(args) -> dict[str, any]:
    """Patch DPO config for HuggingFace training.

    Args:
        args: TrainArguments or BaseArguments instance

    Returns:
        Dictionary with resolved config values
    """
    if isinstance(args, BaseArguments):
        arg_dict = args.to_yaml(exclude=False)
    else:
        arg_dict = dict(args)

    # Resolve tokenizer path (default to model path if not specified)
    if not arg_dict.get("tokenizer_name_or_path"):
        arg_dict["tokenizer_name_or_path"] = arg_dict.get("model_name_or_path")

    # Handle config_path (may be Path or str from to_yaml() serialization)
    if arg_dict.get("config_path"):
        config_path = Path(arg_dict["config_path"])
        arg_dict["config_path"] = str(config_path.parent / "dpo_config.yaml")

    if arg_dict.get("pref_loss"):
        arg_dict["loss_type"] = arg_dict.get("pref_loss")
    if arg_dict.get("pref_beta"):
        arg_dict["beta"] = arg_dict.get("pref_beta")
    return arg_dict


def patch_grpo_config(args) -> dict[str, any]:
    """Patch GRPO config for HuggingFace training.

    Args:
        args: TrainArguments or BaseArguments instance

    Returns:
        Dictionary with resolved config values
    """
    if isinstance(args, BaseArguments):
        arg_dict = args.to_yaml(exclude=False)
    else:
        arg_dict = dict(args)

    # Resolve tokenizer path (default to model path if not specified)
    if not arg_dict.get("tokenizer_name_or_path"):
        arg_dict["tokenizer_name_or_path"] = arg_dict.get("model_name_or_path")

    # Handle config_path (may be Path or str from to_yaml() serialization)
    if arg_dict.get("config_path"):
        config_path = Path(arg_dict["config_path"])
        arg_dict["config_path"] = str(config_path.parent / "grpo_config.yaml")
    return arg_dict


def patch_configs(config: dict[str, any]) -> EasyDict:
    """Parse and patch all config types from a single config dict.

    Args:
        config: Raw config dictionary

    Returns:
        EasyDict with train_args, merge_args, eval_args
    """
    train_args, *_ = HfArgumentParser([TrainArguments]).parse_dict(
        config, allow_extra_keys=True
    )
    train_args = patch_train_config(args=train_args)

    merge_args, *_ = HfArgumentParser((MergeArguments,)).parse_dict(
        config,
        allow_extra_keys=True,
    )
    merge_args = patch_merge_config(
        args=merge_args,
        additional_args=train_args,
    )

    eval_dict = {
        "exp_path": train_args.exp_path,
        "run_name": train_args.run_name,
        **config,
    }
    if train_args.finetuning_type in ["lora", "qlora"]:
        eval_dict["model_name_or_path"] = train_args.model_name_or_path
        eval_dict["adapter_name_or_path"] = train_args.output_dir
    else:
        eval_dict["model_name_or_path"] = train_args.output_dir
    if train_args.report_to == "wandb":
        eval_dict["wandb_project"] = train_args.wandb_project

    # Use output_dir as default model_path (will be fixed in patch_eval_config if LoRA)
    eval_args, *_ = HfArgumentParser((EvaluationArguments,)).parse_dict(
        eval_dict,
        allow_extra_keys=True,
    )

    eval_args = patch_eval_config(
        args=eval_args,
        additional_args=train_args,
    )

    return EasyDict(
        train_args=train_args,
        merge_args=merge_args,
        eval_args=eval_args,
    )
