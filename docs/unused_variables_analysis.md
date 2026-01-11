# Unused Variables Analysis in Argument Classes

This document lists all variables defined in argument classes that appear to be unused in the codebase.

## Analysis Date
Generated on: 2026-01-11

## Methodology
- Searched all Python files in the llmhalluc package
- Identified fields defined in argument classes
- Checked whether each field is accessed anywhere in the codebase
- Fields are considered "unused" if they are only defined but never accessed (except in yaml_exclude)

---

## TrainArguments (`llmhalluc/hparams/train_args.py`)

### Unused Fields

1. **`flash_attn: str = "fa2"`** (Line 24)
   - **Status**: ❌ UNUSED
   - **Usage**: Only defined, never accessed
   - **Suggestion**: Remove if not needed, or integrate into model loading

2. **`dataloader_num_workers: int = 4`** (Line 56)
   - **Status**: ❌ UNUSED
   - **Usage**: Only defined, never accessed
   - **Suggestion**: Likely should be used in DataLoader configuration but currently isn't
   - **Note**: TRL trainers may have their own dataloader configuration that this doesn't affect

3. **`plot_loss: bool = True`** (Line 62)
   - **Status**: ❌ UNUSED
   - **Usage**: Only defined, never accessed
   - **Suggestion**: Remove or implement loss plotting functionality

4. **`save_only_model: bool = False`** (Line 64)
   - **Status**: ❌ UNUSED
   - **Usage**: Only defined, never accessed
   - **Suggestion**: This might have been intended for LLaMA-Factory compatibility but is not used
   - **Note**: TRL trainers handle save behavior differently

5. **`compute_only_loss: bool = False`** (Line 114)
   - **Status**: ❌ UNUSED
   - **Usage**: Only defined, never accessed (SFT specific)
   - **Suggestion**: Remove if not needed for SFT

6. **`cutoff_len: int = 2048`** (Line 51)
   - **Status**: ❌ UNUSED
   - **Usage**: Only defined, never accessed
   - **Suggestion**: Likely should be used for max_seq_length in TRL configs
   - **Note**: TRL trainers use `max_seq_length` parameter instead

7. **`overwrite_cache: bool = True`** (Line 52)
   - **Status**: ❌ UNUSED
   - **Usage**: Only defined, never accessed
   - **Suggestion**: Should be passed to `process_dataset` if dataset caching is desired
   - **Note**: Currently `load_from_cache_file` is used instead

8. **`preprocessing_num_workers: int = 8`** (Line 55)
   - **Status**: ❌ UNUSED
   - **Usage**: Only defined, never accessed
   - **Suggestion**: Should be passed to `dataset.map()` calls in preprocessing
   - **Note**: Currently not used in dataset processing pipeline

9. **`tags: list[str] = field(default_factory=list)`** (Line 15)
   - **Status**: ❌ UNUSED
   - **Usage**: Only defined, never accessed
   - **Suggestion**: Could be used for wandb run tags, but currently not implemented

10. **`reward_model_type: str = ""`** (Line 111)
    - **Status**: ❌ UNUSED (PPO specific)
    - **Usage**: Only defined and conditionally excluded from yaml_exclude
    - **Suggestion**: Remove if PPO is not being used or implement PPO support

### Conditionally Used Fields

1. **`enable_thinking: bool = False`** (Line 19)
   - **Status**: ⚠️ PARTIALLY USED
   - **Usage**: Used in `EvaluationArguments._update_model_args()` for lm-eval-harness
   - **Primary Use**: Only for evaluation, not training
   - **Note**: This is for the lm-evaluation-harness integration

2. **`do_predict: bool = False`** (Line 30)
   - **Status**: ⚠️ MINIMALLY USED
   - **Usage**: Only used in `e2e_setup.py` (legacy script)
   - **Suggestion**: May not be needed for HuggingFace training pipeline

---

## MergeArguments (`llmhalluc/hparams/merge_args.py`)

### Unused Fields

1. **`export_hub_model_id: str | None = None`** (Line 18)
   - **Status**: ❌ UNUSED
   - **Usage**: Only defined, never accessed
   - **Suggestion**: Remove if HuggingFace Hub export is not needed

2. **`export_size: int = 2`** (Line 19)
   - **Status**: ❌ UNUSED
   - **Usage**: Only defined, never accessed
   - **Suggestion**: Remove if not needed

3. **`export_device: str = "auto"`** (Line 20)
   - **Status**: ❌ UNUSED
   - **Usage**: Only defined, never accessed
   - **Suggestion**: Remove if not needed

4. **`export_legacy_format: bool = False`** (Line 21)
   - **Status**: ❌ UNUSED
   - **Usage**: Only defined, never accessed
   - **Suggestion**: Remove if not needed

**Note**: MergeArguments appears to be created but there's no actual merge/export script that uses these fields. The entire merge functionality may need to be implemented or these arguments should be removed.

---

## EvaluationArguments (`llmhalluc/hparams/eval_args.py`)

### Unused Fields

All fields in EvaluationArguments appear to be used, particularly:
- Fields are used in `_update_model_args()` and `_update_wandb_args()`
- Fields are used to construct lm-eval command in `save_eval_cmd()`
- No unused fields identified

**Status**: ✅ All fields appear to be used

---

## FTArguments (Base class for SFT/DPO/GRPO)

### Assessment

All fields in `FTArguments` and its subclasses (`SFTArguments`, `DPOArguments`, `GRPOArguments`) appear to be properly used because:

1. **Inherited from TRL Config classes**: Most fields come from TRL's `SFTConfig`, `DPOConfig`, `GRPOConfig`
2. **Used in training**: The argument objects are passed directly to TRL trainers
3. **Converter field**: Used in `setup_dataset()` to get the appropriate dataset converter
4. **Special token fields**: Used in `setup_tokenizer()` and model initialization
5. **Early stopping fields**: Used in `get_callbacks()` to set up early stopping callback

**Status**: ✅ All fields appear to be used

---

## Summary

### High Priority Removals (Definitely Unused)

From **TrainArguments**:
- `flash_attn`
- `dataloader_num_workers`
- `plot_loss`
- `save_only_model`
- `compute_only_loss`
- `cutoff_len`
- `overwrite_cache`
- `preprocessing_num_workers`
- `tags`
- `reward_model_type`

From **MergeArguments** (entire merge functionality appears unimplemented):
- `export_hub_model_id`
- `export_size`
- `export_device`
- `export_legacy_format`

### Total Count
- **14 unused fields** identified across argument classes
- **10 from TrainArguments**
- **4 from MergeArguments**

### Recommendations

1. **Remove unused fields** to reduce confusion and maintenance burden
2. **Implement missing functionality** if these fields are intended to be used:
   - Flash attention configuration
   - Loss plotting
   - Dataset preprocessing workers
   - WandB tags
   - Merge/export functionality
3. **Document** any fields that are used but the usage is not obvious
4. **Consider** whether PPO support is needed (reward_model, reward_model_type)

---

## Notes

- This analysis is based on grep searches across the codebase
- Some fields might be used dynamically via `getattr()` which could be missed
- Fields passed to TRL trainers are not marked as unused even if not explicitly accessed
- The search included all `.py` files in the llmhalluc package
- External packages like lm-evaluation-harness were included in the search

---

## Cleanup Status (Updated 2026-01-11)

### Fields Successfully Removed from TrainArguments

The following fields have been **removed** from `TrainArguments`:
- ✅ **`flash_attn`** - Removed (not needed)
- ✅ **`cutoff_len`** - Removed (TRL uses max_seq_length)
- ✅ **`overwrite_cache`** - Removed (load_from_cache_file is sufficient)
- ✅ **`preprocessing_num_workers`** - Removed (not integrated)
- ✅ **`tags`** - Removed (not implemented)
- ✅ **`reward_model`** - Removed (PPO not supported)
- ✅ **`reward_model_type`** - Removed (PPO not supported)

**Total removed**: 7 fields

### Fields Intentionally Kept (Despite Current Non-Use)

The following fields were **kept** in `TrainArguments` for potential future use:
- ⚠️ **`dataloader_num_workers`** - May be integrated with TRL trainers in future
- ⚠️ **`plot_loss`** - May implement loss plotting functionality
- ⚠️ **`save_only_model`** - May be used for custom save behavior
- ⚠️ **`compute_only_loss`** - SFT-specific, potential future use

**Total kept**: 4 fields

### MergeArguments - Pending Review

The following fields in `MergeArguments` were **not addressed** in this cleanup:
- ❓ **`export_hub_model_id`**
- ❓ **`export_size`**
- ❓ **`export_device`**
- ❓ **`export_legacy_format`**

**Status**: These fields remain in the code and should be reviewed in a future cleanup if merge/export functionality is not implemented.
