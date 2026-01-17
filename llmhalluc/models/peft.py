"""PEFT (Parameter-Efficient Fine-Tuning) utilities.

Provides LoRA and QLoRA configuration builders for use with any trainer.
"""

from typing import Any

import torch
from peft import LoraConfig, TaskType
from transformers import BitsAndBytesConfig


def get_target_modules(args: Any) -> str | list[str]:
    """Parse lora_target into list of module names or 'all-linear'.

    Args:
        args: Training arguments object.

    Returns:
        Target modules for LoRA.
    """
    lora_target = getattr(args, "lora_target", "all")

    if lora_target == "all":
        return "all-linear"
    else:
        # Parse comma-separated module names
        return [m.strip() for m in lora_target.split(",")]


def get_peft_config(args: Any):
    """Build LoraConfig if finetuning_type is 'lora' or 'qlora'.

    Args:
        args: Training arguments object with PEFT-related fields.

    Returns:
        LoraConfig if using PEFT, None otherwise.
    """
    finetuning_type = getattr(args, "finetuning_type", "full")

    if finetuning_type not in ["lora", "qlora"]:
        return None

    target_modules = get_target_modules(args)

    return LoraConfig(
        r=getattr(args, "lora_rank", 8),
        lora_alpha=getattr(args, "lora_alpha", 16),
        lora_dropout=getattr(args, "lora_dropout", 0.05),
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def get_quantization_config(args: Any):
    """Build BitsAndBytesConfig for QLoRA quantization.

    Args:
        args: Training arguments object with quantization-related fields.

    Returns:
        BitsAndBytesConfig if using QLoRA/4-bit, None otherwise.
    """
    finetuning_type = getattr(args, "finetuning_type", "full")
    load_in_4bit = getattr(args, "load_in_4bit", False)

    if finetuning_type != "qlora" and not load_in_4bit:
        return None

    compute_dtype_str = getattr(args, "bnb_4bit_compute_dtype", "bfloat16")
    compute_dtype = getattr(torch, compute_dtype_str, torch.bfloat16)

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type=getattr(args, "bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=getattr(args, "bnb_4bit_use_double_quant", True),
    )
