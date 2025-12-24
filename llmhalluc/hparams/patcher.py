from transformers import HfArgumentParser
from easydict import EasyDict

from llmhalluc.extras.constant import SPECIAL_TOKEN_MAPPING

from .base_args import BaseArguments
from .train_args import TrainArguments
from .eval_args import EvaluationArguments
from .merge_args import MergeArguments
from .sft_args import SFTArguments


def patch_train_config(
    args: TrainArguments,
    additional_args: BaseArguments | None = None,
) -> dict[str, any]:
    """Hook for experiment-specific train config tweaks."""

    template = args.template
    additional_args = None
    if args.init_special_tokens:
        if "llama3" in template:
            args.replace_text = {"<|BACKTRACK|>": "<|reserved_special_token_0|>"}
            args.force_init_embeddings = True
            model_name = "llama3"
            args.backtrack_token = "<|reserved_special_token_0|>"
        elif "qwen3" in template:
            args.force_init_embeddings = True
            args.backtrack_token = "<|BACKTRACK|>"
            model_name = "qwen3"
        else:
            raise ValueError(f"Unsupported template: {template}")
        additional_args = SPECIAL_TOKEN_MAPPING[model_name]
    return args, additional_args


def patch_merge_config(
    args: MergeArguments,
    additional_args: BaseArguments | None = None,
) -> dict[str, any]:
    """Hook for experiment-specific merge config tweaks."""
    if additional_args.init_special_tokens:
        args.new_special_tokens_config = additional_args.new_special_tokens_config
        args.init_special_tokens = additional_args.init_special_tokens
        args.force_init_embeddings = additional_args.force_init_embeddings
        args.backtrack_token = additional_args.backtrack_token

    args.adapter_name_or_path = additional_args.output_dir
    return args


def patch_eval_config(
    args: EvaluationArguments,
    additional_args: BaseArguments | None = None,
) -> dict[str, any]:
    """Hook for experiment-specific eval config tweaks."""
    return args


def patch_sft_config(args) -> dict[str, any]:
    """Patch SFT config for HuggingFace training.

    Args:
        args: TrainArguments or BaseArguments instance

    Returns:
        Dictionary with resolved config values
    """
    if isinstance(args, BaseArguments):
        arg_dict = args.to_yaml()
    else:
        arg_dict = dict(args)

    # Resolve tokenizer path (default to model path if not specified)
    if not arg_dict.get("tokenizer_name_or_path"):
        arg_dict["tokenizer_name_or_path"] = arg_dict.get("model_name_or_path")

    return arg_dict


def patch_configs(config: dict[str, any]) -> dict[str, any]:
    train_args, *_ = HfArgumentParser([TrainArguments]).parse_dict(
        config, allow_extra_keys=True
    )
    train_args, extra_args = patch_train_config(args=train_args, additional_args=None)
    merge_args, *_ = HfArgumentParser((MergeArguments,)).parse_dict(
        config,
        allow_extra_keys=True,
    )
    merge_args = patch_merge_config(
        args=merge_args,
        additional_args=train_args,
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
    )

    return EasyDict(
        train_args=train_args,
        merge_args=merge_args,
        eval_args=eval_args,
        extra_args=extra_args,
    )
