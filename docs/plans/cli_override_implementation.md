# Implementation Plan: CLI Argument Override Support

## Objective
Enable command-line argument overrides when running `accelerate launch -m llmhalluc.hf_train --num_train_epochs 1`, so CLI arguments take precedence over YAML config values.

## Current Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   e2e.yaml      │ ──▶ │  load_config()   │ ──▶ │ patch_configs() │ ──▶ │ TrainArguments  │
│   (config)      │     │  (cfg_utils.py)  │     │  (patcher.py)   │     │  (dataclass)    │
└─────────────────┘     └──────────────────┘     └─────────────────┘     └─────────────────┘
                                                                                  │
                                                                                  ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   run_train()   │ ◀── │   hf_cfg_setup() │ ◀── │ patch_sft_cfg() │ ◀── │  SFTArguments   │
│   (train.py)    │     │   (cfg_utils.py) │     │  (patcher.py)   │     │  (dataclass)    │
└─────────────────┘     └──────────────────┘     └─────────────────┘     └─────────────────┘
```

**Problem:** No CLI parsing happens anywhere. `HfArgumentParser.parse_dict()` is used instead of `parse_args_into_dataclasses()`.

## Proposed Solution

Use a **two-phase parsing** approach:
1. **Phase 1:** Load YAML config as defaults
2. **Phase 2:** Parse CLI args and override the defaults

### Strategy: Hybrid Dict + CLI Parsing

HuggingFace's `HfArgumentParser` supports parsing CLI arguments with defaults from a namespace. We'll:
1. Load YAML config into a dict
2. Use `parse_args_into_dataclasses()` with remaining unknown args

---

## Implementation Steps

### Step 1: Create a CLI Argument Parser Utility

**File:** `llmhalluc/utils/cli_utils.py` (new file)

```python
"""CLI argument utilities for config override support."""


def parse_cli_to_dotlist(args: list[str]) -> list[str]:
    """
    Convert CLI args like --num_train_epochs 1 to OmegaConf dotlist format.
    
    Args:
        args: CLI arguments like ['--num_train_epochs', '1', '--lr', '1e-4']
    
    Returns:
        Dotlist format like ['num_train_epochs=1', 'lr=1e-4']
    """
    dotlist = []
    i = 0
    while i < len(args):
        arg = args[i]
        if arg.startswith('--'):
            key = arg[2:]  # Remove '--'
            # Check if next arg is a value or another flag
            if i + 1 < len(args) and not args[i + 1].startswith('--'):
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
```

### Step 2: Modify `hf_cfg_setup()` in `cfg_utils.py`

**File:** `llmhalluc/utils/cfg_utils.py`

**Changes:**
1. Add optional `cli_args` parameter
2. Use `parse_cli_to_dotlist()` + `apply_overrides()` to merge CLI args into config before parsing

```python
def hf_cfg_setup(
    config_path: str, 
    save_cfg: bool = True,
    cli_args: list[str] | None = None,  # NEW: Accept CLI args
) -> EasyDict:
    """Setup HF training config with optional CLI overrides.
    
    Args:
        config_path: Path to YAML config file
        save_cfg: Whether to save resolved configs
        cli_args: CLI arguments for overrides (defaults to sys.argv[1:])
    """
    # Load base config from YAML
    config = load_config(config_path)
    
    # Apply CLI overrides to config dict BEFORE parsing into dataclasses
    if cli_args is None:
        import sys
        cli_args = sys.argv[1:]
    
    # Use OmegaConf to apply CLI overrides in dotlist format
    config = apply_overrides(config, parse_cli_to_dotlist(cli_args))
    
    # Continue with existing logic...
    arg_dict = patch_configs(config)
    # ... rest of function
```

### Step 3: Update `hf_train.py` Entry Point

**File:** `llmhalluc/hf_train.py`

```python
"""HuggingFace training entry point.

Usage:
    accelerate launch -m llmhalluc.hf_train
    accelerate launch -m llmhalluc.hf_train --num_train_epochs 1
    accelerate launch -m llmhalluc.hf_train --learning_rate 5e-5 --per_device_train_batch_size 8
"""
import sys
from pathlib import Path

from llmhalluc.utils import setup_logging, hf_cfg_setup
from llmhalluc.train import run_train

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "llmhalluc" / "e2e.yaml"


def main():
    setup_logging(verbose=False)
    
    # Pass CLI args (excluding accelerate's args which are already consumed)
    # sys.argv[1:] contains user-provided overrides
    setup_dict = hf_cfg_setup(DEFAULT_CONFIG_PATH, cli_args=sys.argv[1:])
    
    run_train(setup_dict.args.hf_args)


if __name__ == "__main__":
    main()
```

### Step 4: Add `--config` Flag Support (Optional Enhancement)

Allow users to specify a different config file:

```python
def main():
    setup_logging(verbose=False)
    
    # Check for --config flag
    config_path = DEFAULT_CONFIG_PATH
    args = sys.argv[1:]
    
    if '--config' in args:
        idx = args.index('--config')
        config_path = Path(args[idx + 1])
        args = args[:idx] + args[idx + 2:]  # Remove --config and its value
    
    setup_dict = hf_cfg_setup(config_path, cli_args=args)
    run_train(setup_dict.args.hf_args)
```

---

## File Changes Summary

| File | Action | Description |
|------|--------|-------------|
| `llmhalluc/utils/cli_utils.py` | **CREATE** | New file with `parse_cli_to_dotlist()` function |
| `llmhalluc/utils/__init__.py` | **MODIFY** | Export new CLI utilities |
| `llmhalluc/utils/cfg_utils.py` | **MODIFY** | Add `cli_args` parameter to `hf_cfg_setup()`, integrate CLI override logic |
| `llmhalluc/hf_train.py` | **MODIFY** | Pass `sys.argv[1:]` to `hf_cfg_setup()` |

---

## Usage Examples

After implementation:

```bash
# Use YAML config only (current behavior preserved)
accelerate launch -m llmhalluc.hf_train

# Override specific arguments
accelerate launch -m llmhalluc.hf_train --num_train_epochs 1

# Override multiple arguments
accelerate launch -m llmhalluc.hf_train --num_train_epochs 1 --learning_rate 5e-5

# Use different config file (optional enhancement)
accelerate launch -m llmhalluc.hf_train --config ./configs/custom.yaml

# Combine custom config with overrides
accelerate launch -m llmhalluc.hf_train --config ./configs/custom.yaml --num_train_epochs 2
```

---

## Testing Plan

1. **Unit Tests:**
   - Test `parse_cli_to_dotlist()` with various arg formats
   - Test `apply_overrides()` with CLI dotlist

2. **Integration Tests:**
   - Run with no CLI args → should use YAML defaults
   - Run with `--num_train_epochs 1` → should override
   - Run with invalid arg → should error gracefully

3. **Manual Validation:**
   - Check saved config files reflect CLI overrides
   - Verify training actually uses overridden values

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Accelerate passes its own args to the script | Filter out accelerate-specific args OR rely on accelerate consuming them before they reach our script |
| Type conversion issues (string → int/float) | OmegaConf handles type inference; add explicit type hints in dataclasses |
| Breaking existing workflows | All changes are additive; no-arg invocation works exactly as before |

---

## Implementation Order

1. ✅ Create `cli_utils.py` with `parse_cli_to_dotlist()`
2. ✅ Modify `cfg_utils.py` to accept and apply CLI overrides  
3. ✅ Update `hf_train.py` to pass CLI args
4. ✅ Update `__init__.py` exports
5. ✅ Test with sample overrides
6. (Optional) Add `--config` flag support
