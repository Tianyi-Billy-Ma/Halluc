#!/usr/bin/env python3
"""Test script to verify GRPO argument passing."""

from llmhalluc.hparams.train_args import TrainArguments
from llmhalluc.hparams.ft_args import GRPOArguments
from llmhalluc.hparams.patcher import patch_grpo_config
from transformers import HfArgumentParser

# Simulate a config with GRPO-specific fields
test_config = {
    "run_name": "test",
    "model_name_or_path": "meta-llama/Llama-3.1-8B",
    "template": "llama3",
    "stage": "grpo",
    "finetuning_type": "lora",
    "dataset": "gsm8k_bt_grpo_train",
    "converter": "grpo",
    "reward_funcs": "backtrack",
    # GRPO-specific from BaseGRPOConfig
    "num_generations": 8,  # Should this be passed through?
    "max_completion_length": 512,  # Should this be passed through?
}

print("=" * 80)
print("Step 1: Parse into TrainArguments")
print("=" * 80)
train_args, *_ = HfArgumentParser([TrainArguments]).parse_dict(
    test_config, allow_extra_keys=True
)

print(f"Parsed TrainArguments successfully")
print(f"Has num_generations attr: {hasattr(train_args, 'num_generations')}")
print(f"Has max_completion_length attr: {hasattr(train_args, 'max_completion_length')}")

print("\n" + "=" * 80)
print("Step 2: Convert TrainArguments to dict via to_yaml()")
print("=" * 80)
train_dict = train_args.to_yaml(exclude=False)
print(f"num_generations in dict: {'num_generations' in train_dict}")
print(f"max_completion_length in dict: {'max_completion_length' in train_dict}")

print("\n" + "=" * 80)
print("Step 3: Patch config for GRPO")
print("=" * 80)
patched_dict = patch_grpo_config(train_args)
print(f"num_generations in patched dict: {'num_generations' in patched_dict}")
print(
    f"max_completion_length in patched dict: {'max_completion_length' in patched_dict}"
)

print("\n" + "=" * 80)
print("Step 4: Parse into GRPOArguments")
print("=" * 80)
grpo_args, *_ = HfArgumentParser(GRPOArguments).parse_dict(
    patched_dict, allow_extra_keys=True
)

print(f"Parsed GRPOArguments successfully")
print(f"Has num_generations attr: {hasattr(grpo_args, 'num_generations')}")
print(f"Has max_completion_length attr: {hasattr(grpo_args, 'max_completion_length')}")

if hasattr(grpo_args, "num_generations"):
    print(f"num_generations value: {grpo_args.num_generations}")
if hasattr(grpo_args, "max_completion_length"):
    print(f"max_completion_length value: {grpo_args.max_completion_length}")

print("\n" + "=" * 80)
print("CONCLUSION:")
print("=" * 80)
if hasattr(grpo_args, "num_generations"):
    if grpo_args.num_generations == 8:
        print("✅ GRPO-specific fields ARE being passed through correctly!")
    else:
        print(
            f"⚠️ GRPO-specific fields exist but have default value: {grpo_args.num_generations}"
        )
else:
    print("❌ GRPO-specific fields are NOT being passed through!")
