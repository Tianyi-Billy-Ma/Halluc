"""Config patching utilities for training arguments."""

import logging
from pathlib import Path

from easydict import EasyDict
from transformers import HfArgumentParser
from transformers.trainer_utils import get_last_checkpoint

from llmhalluc.extras.constant import SPECIAL_TOKEN_MAPPING

from .base_args import BaseArguments
from .eval_args import EvaluationArguments
from .gen_args import GenerationArguments
from .merge_args import MergeArguments
from .train_args import TrainArguments

logger = logging.getLogger(__name__)


def patch_train_config(args: TrainArguments) -> TrainArguments:
    """Hook for experiment-specific train config tweaks.

    Auto-sets special token config based on template when init_special_tokens=True
    but new_special_tokens_config is not provided.

    Args:
        args: TrainArguments to patch

    Returns:
        Patched TrainArguments (same object, modified in place)
    """
    train_backtrack = args.train_backtrack
    init_special_tokens = args.init_special_tokens

    if train_backtrack and not init_special_tokens:
        logger.warning(
            "train_backtrack is True but init_special_tokens is False. "
            "Forcing init_special_tokens to True."
        )
        args.init_special_tokens = True

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
        if "llama3" in template:
            args.replace_text = args.replace_text or {}
            args.replace_text["<|BACKTRACK|>"] = "<|reserved_special_token_0|>"
            args.backtrack_token = "<|reserved_special_token_0|>"
            # qwen3 can use <|BACKTRACK|> directly, no replace needed
    else:
        args.replace_text = None
        args.new_special_tokens_config = None
        args.backtrack_token = ""

    # Patch reset_position_ids
    if getattr(args, "reset_position_ids", False) and not getattr(
        args, "train_backtrack", False
    ):
        logger.warning(
            "reset_position_ids is True but train_backtrack is False. "
            "Forcing reset_position_ids to False."
        )
        args.reset_position_ids = False

    # Patch resume_from_checkpoint
    if args.resume:
        if args.resume_from_checkpoint:
            logger.info(
                f"Resume requested with explicit path: {args.resume_from_checkpoint}"
            )
        elif args.output_dir:
            last_checkpoint = get_last_checkpoint(args.output_dir)
            if last_checkpoint:
                logger.info(f"Auto-detected checkpoint: {last_checkpoint}")
                args.resume_from_checkpoint = last_checkpoint
            else:
                raise ValueError(
                    f"Resume requested but no checkpoint found in {args.output_dir}. Cannot resume training."
                )
        else:
            raise ValueError("resume=True but no output_dir specified. Cannot resume.")

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

        if additional_args.finetuning_type in ["lora", "qlora"]:
            args.model_name_or_path = additional_args.model_name_or_path
            args.adapter_name_or_path = additional_args.output_dir
        else:
            args.model_name_or_path = additional_args.output_dir
        args._update_model_args()

        if additional_args.report_to == "wandb":
            args.report_to = "wandb"
            args.wandb_project = additional_args.wandb_project
            args._update_wandb_args()

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


def patch_gen_config(
    args: GenerationArguments,
    additional_args: TrainArguments | None = None,
) -> GenerationArguments:
    """Hook for experiment-specific generation config tweaks.

    Args:
        args: GenerationArguments to patch
        additional_args: TrainArguments for referencing training config

    Returns:
        Patched GenerationArguments
    """
    if additional_args:
        # Use trained model/adapter paths
        if additional_args.finetuning_type in ["lora", "qlora"]:
            args.model_name_or_path = additional_args.model_name_or_path
            args.adapter_name_or_path = additional_args.output_dir
        else:
            args.model_name_or_path = additional_args.output_dir

        # Use trained tokenizer
        args.tokenizer_name_or_path = additional_args.output_dir

        # Copy dataset info if not explicitly set
        if not args.dataset and additional_args.dataset:
            args.dataset = additional_args.dataset

    return args


def patch_configs(config: dict[str, any]) -> EasyDict:
    """Parse and patch all config types from a single config dict.

    Args:
        config: Raw config dictionary

    Returns:
        EasyDict with train_args, merge_args, eval_args, gen_args
    """
    train_args, *_ = HfArgumentParser([TrainArguments]).parse_dict(
        config, allow_extra_keys=True
    )
    train_args = patch_train_config(args=train_args)

    merge_dict = {
        **config,
        "exp_path": train_args.exp_path,
        "run_name": train_args.run_name,
    }

    merge_args, *_ = HfArgumentParser((MergeArguments,)).parse_dict(
        merge_dict,
        allow_extra_keys=True,
    )
    merge_args = patch_merge_config(
        args=merge_args,
        additional_args=train_args,
    )

    eval_dict = {
        **config,
        "exp_path": train_args.exp_path,
        "run_name": train_args.run_name,
    }

    # Use output_dir as default model_path (will be fixed in patch_eval_config if LoRA)
    eval_args, *_ = HfArgumentParser((EvaluationArguments,)).parse_dict(
        eval_dict,
        allow_extra_keys=True,
    )

    eval_args = patch_eval_config(
        args=eval_args,
        additional_args=train_args,
    )

    # Parse and patch generation args
    gen_dict = {
        **config,
        "exp_path": train_args.exp_path,
        "run_name": train_args.run_name,
    }

    gen_args, *_ = HfArgumentParser((GenerationArguments,)).parse_dict(
        gen_dict,
        allow_extra_keys=True,
    )

    gen_args = patch_gen_config(
        args=gen_args,
        additional_args=train_args,
    )

    return EasyDict(
        train_args=train_args,
        merge_args=merge_args,
        eval_args=eval_args,
        gen_args=gen_args,
    )
