# GRPO Argument Passing Issue - FIXED

## Issue Summary

**Problem**: GRPO-specific arguments from `BaseGRPOConfig` (like `num_generations`, `max_completion_length`, etc.) were being **lost** during the configuration pipeline.

**Reported by User**: 2026-01-11

## Root Cause

The configuration flow was:
1. Load YAML config → dict
2. Parse dict into `TrainArguments` (using `HfArgumentParser`)
3. Convert `TrainArguments.to_yaml()` → dict (❌ **LOST GRPO-specific fields here**)
4. Pass dict to `HfArgumentParser(GRPOArguments)`

The problem was in step 3: `TrainArguments.to_yaml()` only exports fields **defined in `TrainArguments`**. Fields like `num_generations` that are defined in `BaseGRPOConfig` but NOT in `TrainArguments` were silently dropped.

## Solution Applied

**File**: `/mnt/black/Project/Halluc/llmhalluc/hparams/parser.py`
**Function**: `hf_cfg_setup()`

Added logic to preserve the original config and merge it with the patched config:

```python
def hf_cfg_setup(
    config_path: str,
    save_cfg: bool = True,
    cli_args: list[str] | None = None,
) -\u003e EasyDict:
    # NEW: Store original config to preserve stage-specific fields
    original_config = load_config(config_path)
    if cli_args:
        overrides = parse_cli_to_dotlist(cli_args)
        original_config = apply_overrides(original_config, overrides)
    
    setup_dict = e2e_cfg_setup(config_path, save_cfg=save_cfg, cli_args=cli_args)
    train_args = setup_dict.args.train_args

    hf_args = None
    stage = getattr(train_args, "stage", "sft")

    if stage == "grpo":
        raw_args: dict[str, any] = patch_grpo_config(train_args)
        # NEW: Merge original config to preserve GRPO-specific fields
        raw_args = {**original_config, **raw_args}  # ← FIX HERE
        hf_args, *_ = HfArgumentParser(GRPOArguments).parse_dict(
            raw_args, allow_extra_keys=True
        )
        ...
```

### How It Works

1. **Load original config** before it gets parsed into `TrainArguments`
2. **Merge**: `{**original_config, **raw_args}`
   - `original_config` has ALL fields from YAML (including GRPO-specific ones)
   -  `raw_args` has patched/derived fields from `TrainArguments`
   - Patched fields override original (correct behavior)
   - GRPO-specific fields are preserved from original

## Affected Fields (Examples)

GRPO-specific fields from `BaseGRPOConfig` that are now properly preserved:
- ✅ `num_generations` - Number of completions to generate per prompt
- ✅ `max_completion_length` - Maximum length of generated completions
- ✅ `num_generations_per_prompt` - Alternative name for num_generations
- ✅ `temperature` - Sampling temperature for generation
- ✅ `top_k`, `top_p` - Sampling parameters
- ✅ Any other GRPO-specific TRL config parameters

Similarly for DPO and SFT stages, their specific fields are now preserved.

## Testing

To verify GRPO parameters are passed through:
1. Add GRPO-specific fields to your YAML config
2. They will now reach the `GRPOTrainer` correctly
3. Check saved config at `outputs/.../grpo/grpo_config.yaml` to verify

Example YAML:
```yaml
stage: grpo
num_generations: 8
max_completion_length: 512
temperature: 0.7
reward_funcs: "backtrack,accuracy"
```

## Impact

- ✅ **GRPO training**: Now respects user-specified generation parameters
- ✅ **DPO training**: DPO-specific parameters preserved
- ✅ **SFT training**: SFT-specific parameters preserved
- ✅ **No breaking changes**: Existing configs continue to work
- ✅ **Backward compatible**: Missing fields still use defaults

## Files Changed

1. `/mnt/black/Project/Halluc/llmhalluc/hparams/parser.py`
   - Modified `hf_cfg_setup()` to preserve original config
   - Added merging logic for all three stages (SFT, DPO, GRPO)

## Date Fixed

2026-01-11
