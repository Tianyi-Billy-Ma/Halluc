# HuggingFace Training Implementation Plan

## Executive Summary

This document outlines the implementation plan for completing the HuggingFace-based training pipeline (`hf.py`). The main focus is on fixing critical issues in the configuration setup, data loading logic under `SFTExecutor`, and ensuring the pipeline works end-to-end.

---

## 1. Identified Issues

### 1.1 Critical Issues

#### ~~Issue 1: `hf_cfg_setup()` Called Without Arguments~~ [RESOLVED]
**Status:** Resolved by user

The user has introduced a workaround using `EasyDict` with minimal required arguments:
```python
def main():
    args = EasyDict(
        config=DEFAULT_CONFIG_PATH,
    )
    setup_logging(verbose=True)
    hf_args = hf_cfg_setup(args)
    run_train(hf_args)
```

**Discussion Point:** The current approach avoids conflicts with `HfArgumentParser` by using `EasyDict` instead of `argparse`. However, this approach requires hardcoding defaults in `hf.py`. Alternative approaches to consider:
1. Use environment variables for configuration override
2. Create a separate argument namespace that doesn't conflict with HF's parser
3. Use a YAML-based configuration override system (load base config, then apply overrides from CLI)

**Current Required Args by `hf_cfg_setup`:**
- `args.config` - path to config file
- `args.override` - list of config overrides (accessed in `apply_overrides`)
- `args.do_train`, `args.do_merge`, `args.do_eval` - plan flags (accessed in `e2e_cfg_setup`)
- `args.stage` - training stage ("sft", "dpo", "ppo")

**Recommendation:** Update `EasyDict` to include all required fields with sensible defaults:
```python
args = EasyDict(
    config=DEFAULT_CONFIG_PATH,
    override=[],
    do_train=True,
    do_merge=False,
    do_eval=False,
    stage="sft",  # Read from config if not specified
)
```

---

#### Issue 2: `SFTExecutor.setup_dataset()` is Empty
**Location:** [sft.py:11-12](llmhalluc/train/sft.py#L11-L12)

```python
class SFTExecutor(BaseExecutor):
    def setup_dataset(self):
        pass  # Does nothing!
```

**Problem:** The `setup_dataset()` method is a no-op, meaning the dataset is never loaded. However, `setup_trainer()` at line 18 tries to access `self.dataset["train"]`, which will fail since `self.dataset` is `None`.

**Impact:** `AttributeError` or `TypeError` when accessing `self.dataset["train"]` in `setup_trainer()`.

---

#### ~~Issue 3: Missing `get_model` and `get_tokenizer` Functions~~ [RESOLVED]
**Status:** Resolved by user

The `llmhalluc/models/` module now exists with:
- [models/__init__.py](llmhalluc/models/__init__.py) - exports `get_model`, `get_tokenizer`
- [models/manager.py](llmhalluc/models/manager.py) - implements `get_model`, `get_tokenizer`
- [models/patcher.py](llmhalluc/models/patcher.py) - `patch_model`, `patch_tokenizer`

**Note:** There's a minor issue in `get_model()` - it doesn't return the model:
```python
def get_model(model_name_or_path: str, **kwargs):
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
    model = patch_model(model)
    # Missing: return model
```

---

#### Issue 4: `converter` Attribute Not Initialized
**Location:** [base.py:72-74](llmhalluc/train/base.py#L72-L74)

**Problem:** `self.converter` is referenced in `setup_dataset()` but may not be stored from the constructor argument.

**Resolution:** Will be addressed if necessary during SFT data loading implementation.

---

#### Issue 5: Dataset Config Path Does Not Exist
**Location:** [data/utils.py:4-6](llmhalluc/data/utils.py#L4-L6)

```python
DEFAULT_DATASET_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "llmhalluc" / "dataset_config.yaml"
)
```

**Problem:** Points to non-existent path. The actual dataset config is at [data/dataset_info.json](data/dataset_info.json).

**Design Decision:** Config files should stay outside `llmhalluc/` package. Update path to reference the existing JSON file.

---

#### Issue 6: `e2e_cfg_setup` Has `save_config` Name Collision
**Location:** [cfg_utils.py:43, 58-60](llmhalluc/utils/cfg_utils.py#L43)

```python
def e2e_cfg_setup(args, save_config: bool = True) -> int:
    # ...
    if save_config:
        save_config(...)  # TypeError: 'bool' object is not callable
```

**Problem:** Parameter `save_config` shadows the function `save_config()`.

**Resolution:** Rename parameter to `save_cfg` to avoid collision.

---

### 1.2 Design Issues

#### Issue 7: Inconsistent Data Format Requirements
The `SFTTrainer` from TRL expects data in specific formats (e.g., `text` column for completion-only training, or `messages` for chat format). The current converters produce `{prompt, query, response}` format which is not directly compatible.

#### Issue 8: Missing Chat Template Application
For SFT training, the data typically needs to be formatted using the model's chat template. This step is missing in the current data loading pipeline.

---

## 2. Implementation Plan

### Phase 1: Fix Critical Bugs

#### Task 1.1: Complete Argument Setup in `hf.py`
**File:** `llmhalluc/hf.py`

**Changes:**
Add missing fields to `EasyDict` that are required by `hf_cfg_setup`:

```python
from pathlib import Path
from easydict import EasyDict

from llmhalluc.utils import setup_logging, hf_cfg_setup, load_config
from llmhalluc.train import run_train

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "llmhalluc" / "e2e.yaml"


def main():
    # Load config to get stage from YAML
    config = load_config(DEFAULT_CONFIG_PATH)

    args = EasyDict(
        config=DEFAULT_CONFIG_PATH,
        override=[],
        do_train=True,
        do_merge=False,
        do_eval=False,
        stage=config.get("stage", "sft"),
    )
    setup_logging(verbose=True)
    hf_args = hf_cfg_setup(args)
    run_train(hf_args)


if __name__ == "__main__":
    main()
```

---

#### Task 1.2: Fix `save_config` Name Collision
**File:** `llmhalluc/utils/cfg_utils.py`

**Changes:**
Rename parameter from `save_config` to `save_cfg`:

```python
def e2e_cfg_setup(args, save_cfg: bool = True) -> int:
    config = load_config(args.config)
    config = apply_overrides(config, getattr(args, 'override', []))

    plan = {
        "train": getattr(args, 'do_train', True),
        "merge": getattr(args, 'do_merge', False),
        "eval": getattr(args, 'do_eval', False)
    }

    if not any(plan.values()):
        raise ValueError("At least one stage must be enabled to generate configs.")

    arg_dict = patch_configs(config, plan)
    train_args = arg_dict.train_args
    merge_args = arg_dict.merge_args
    eval_args = arg_dict.eval_args  # Note: was incorrectly merge_args
    extra_args = arg_dict.extra_args

    if save_cfg:
        save_config(train_args.to_yaml(), train_args.config_path)
        save_config(merge_args.to_yaml(), merge_args.config_path)
        save_config(eval_args.to_yaml(), eval_args.config_path)

        if extra_args:
            save_config(extra_args, train_args.new_special_tokens_config)

    # ... rest of function
```

---

#### Task 1.3: Fix `get_model` Return Statement
**File:** `llmhalluc/models/manager.py`

**Changes:**
Add missing return statement:

```python
def get_model(model_name_or_path: str, **kwargs):
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
    model = patch_model(model)
    return model  # Add this line
```

---

#### Task 1.4: Fix Dataset Config Path
**File:** `llmhalluc/data/utils.py`

**Changes:**
Update to point to the existing JSON file in `data/`:

```python
from pathlib import Path
import json

DEFAULT_DATASET_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "dataset_info.json"
)


def load_data_config(
    path: str | Path = DEFAULT_DATASET_CONFIG_PATH,
) -> dict[str, any]:
    """Load dataset configuration from JSON file.

    Args:
        path: Path to dataset config file (JSON format)

    Returns:
        Dictionary containing dataset configurations
    """
    path = Path(path)
    with open(path, "r") as f:
        return json.load(f)
```

---

### Phase 2: Implement SFT Data Loading

#### Task 2.1: Create `SFTDatasetConverter`
**File:** `llmhalluc/data/sft.py` (new file)

**Design Approach:**
Create a new converter following the existing pattern that transforms `{prompt, query, response}` format to TRL-compatible `{messages}` format.

**Implementation:**
```python
"""SFT dataset converter for TRL SFTTrainer."""

from dataclasses import dataclass
from typing import Any

from .base import DatasetConverter


@dataclass
class SFTDatasetConverter(DatasetConverter):
    """Converter that transforms {prompt, query, response} to TRL messages format.

    This converter takes the standard format produced by other converters
    and converts it to the chat messages format expected by TRL's SFTTrainer.

    Args:
        prompt_key: Key for the system prompt in input examples.
        query_key: Key for the user query in input examples.
        response_key: Key for the assistant response in input examples.
    """

    prompt_key: str = "prompt"
    query_key: str = "query"
    response_key: str = "response"

    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        """Convert an example to TRL messages format.

        Args:
            example: Input example with prompt, query, and response fields.

        Returns:
            Example with 'messages' field containing list of role/content dicts.
        """
        messages = []

        # Add system message if prompt exists
        prompt_content = example.get(self.prompt_key, "")
        if prompt_content:
            messages.append({
                "role": "system",
                "content": prompt_content
            })

        # Add user message
        query_content = example.get(self.query_key, "")
        if query_content:
            messages.append({
                "role": "user",
                "content": query_content
            })

        # Add assistant response
        response_content = example.get(self.response_key, "")
        messages.append({
            "role": "assistant",
            "content": response_content
        })

        return {"messages": messages}
```

**Update `llmhalluc/data/__init__.py`:**
```python
from .sft import SFTDatasetConverter

__all__ = [
    # ... existing exports ...
    "SFTDatasetConverter",
]
```

**Update `llmhalluc/data/manager.py`:**
```python
from .sft import SFTDatasetConverter

DATASET_CONVERTERS = {
    # ... existing converters ...
    "sft": SFTDatasetConverter,
}
```

---

#### Task 2.2: Implement `SFTExecutor.setup_dataset()`
**File:** `llmhalluc/train/sft.py`

**Design Approach:**
1. Use `load_data_config()` to get dataset configuration from `dataset_info.json`
2. Load train and eval datasets separately (aligning with LlamaFactory)
3. Apply `SFTDatasetConverter` to transform to TRL format
4. Use existing `process_dataset()` utility for transformations

**Implementation:**
```python
from trl import SFTTrainer
from datasets import DatasetDict, load_dataset

from .base import BaseExecutor
from llmhalluc.hparams import SFTArguments
from llmhalluc.data import load_data_config, SFTDatasetConverter
from llmhalluc.utils import process_dataset


class SFTExecutor(BaseExecutor):
    def __init__(self, args: SFTArguments):
        super().__init__(args)

    def setup_dataset(self):
        """Load and prepare dataset for SFT training.

        Loads train and eval datasets separately from their configured
        HuggingFace Hub URLs and applies SFTDatasetConverter.
        """
        data_config = load_data_config()

        # Load and process train dataset
        train_info = data_config.get(self.args.dataset, {})
        train_dataset = load_dataset(
            self.args.dataset_path,
            name=train_info.get("subset"),
            split=train_info.get("split", "train"),
        )

        # Get column mapping and create converter
        column_mapping = train_info.get("columns", {})
        sft_converter = SFTDatasetConverter(
            prompt_key=column_mapping.get("prompt", "prompt"),
            query_key=column_mapping.get("query", "query"),
            response_key=column_mapping.get("response", "response"),
        )

        # Apply converter using existing utility
        train_dataset = process_dataset(
            dataset=train_dataset,
            processor=sft_converter,
        )

        # Build DatasetDict
        self.dataset = DatasetDict({"train": train_dataset})

        # Load and process eval dataset if specified
        if hasattr(self.args, 'eval_dataset') and self.args.eval_dataset:
            eval_info = data_config.get(self.args.eval_dataset, {})
            eval_dataset = load_dataset(
                self.args.eval_dataset_path,
                name=eval_info.get("subset"),
                split=eval_info.get("split", "test"),
            )

            # Get eval column mapping (may differ from train)
            eval_column_mapping = eval_info.get("columns", {})
            eval_converter = SFTDatasetConverter(
                prompt_key=eval_column_mapping.get("prompt", "prompt"),
                query_key=eval_column_mapping.get("query", "query"),
                response_key=eval_column_mapping.get("response", "response"),
            )

            eval_dataset = process_dataset(
                dataset=eval_dataset,
                processor=eval_converter,
            )
            self.dataset["eval"] = eval_dataset

    def setup_trainer(self):
        """Setup SFTTrainer with the loaded model, tokenizer, and dataset."""
        train_dataset = self.dataset.get("train")
        eval_dataset = self.dataset.get("eval")

        self.trainer = SFTTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=self.args,
        )


def run_sft(args: SFTArguments):
    executor = SFTExecutor(args=args)
    executor.on_train_start()
    executor.fit()
    executor.on_train_end()
```

---

#### Task 2.3: Update `BaseExecutor` to Store Converter (If Necessary)
**File:** `llmhalluc/train/base.py`

**Changes:**
Ensure converter is stored if passed:

```python
class BaseExecutor(ABC):
    def __init__(
        self,
        args,
        model: PreTrainedModel = None,
        tokenizer: PreTrainedTokenizer | None = None,
        dataset: DatasetDict | Dataset | None = None,
        converter: DatasetConverter | None = None,
        trainer: Trainer | None = None,
        save_model: bool = True,
    ):
        self.args = args
        self.save_model = save_model
        self.converter = converter  # Store the converter

        # ... rest of init
```

---

#### Task 2.3: Add Eval Dataset Support to SFTArguments
**File:** `llmhalluc/hparams/sft_args.py`

**Changes:**
Uncomment and implement eval dataset fields:

```python
@dataclass
class SFTArguments(BaseSFTConfig):
    # ... existing fields ...

    eval_dataset_path: str | None = field(
        default=None,
        metadata={"help": "The eval dataset path (HF Hub URL or local path)"}
    )
    eval_dataset_name: str | None = field(
        default=None,
        metadata={"help": "The eval dataset name/config"}
    )
    eval_dataset: str | None = field(
        default=None,
        metadata={"help": "The eval dataset key in dataset_info.json"}
    )
    dataset: str | None = field(
        default=None,
        metadata={"help": "The train dataset key in dataset_info.json"}
    )
```

---

### Phase 3: Integration

#### Task 3.1: Update `patch_sft_config` to Handle All Fields
**File:** `llmhalluc/hparams/patcher.py`

**Changes:**
Ensure eval dataset paths are resolved from config:

```python
def patch_sft_config(args) -> dict[str, any]:
    if isinstance(args, BaseArguments):
        arg_dict = args.to_yaml()
    else:
        arg_dict = dict(args)

    data_config = load_data_config()

    # Resolve tokenizer path
    if "tokenizer_name_or_path" not in arg_dict or not arg_dict.get("tokenizer_name_or_path"):
        arg_dict["tokenizer_name_or_path"] = arg_dict.get("model_name_or_path")

    # Resolve train dataset
    dataset_name = arg_dict.get("dataset")
    if dataset_name and not arg_dict.get("dataset_path"):
        dataset_info = data_config.get(dataset_name)
        if dataset_info is None:
            raise ValueError(f"Dataset '{dataset_name}' not found in data config")
        arg_dict["dataset_path"] = dataset_info.get("hf_hub_url")
        arg_dict["dataset_name"] = dataset_info.get("subset")

    # Resolve eval dataset
    eval_dataset_name = arg_dict.get("eval_dataset")
    if eval_dataset_name and not arg_dict.get("eval_dataset_path"):
        eval_dataset_info = data_config.get(eval_dataset_name)
        if eval_dataset_info:
            arg_dict["eval_dataset_path"] = eval_dataset_info.get("hf_hub_url")
            arg_dict["eval_dataset_name"] = eval_dataset_info.get("subset")

    return arg_dict
```

---

#### Task 3.2: Fix `hf_cfg_setup` to Access Correct Args
**File:** `llmhalluc/utils/cfg_utils.py`

**Issue:** Line 88 accesses `setup_dict.args.train_args` but `e2e_cfg_setup` returns `EasyDict(paths=output, args=args)` where `args` is the input args, not `arg_dict`.

**Changes:**
```python
def hf_cfg_setup(args) -> SFTArguments:
    setup_dict = e2e_cfg_setup(args, save_cfg=False)

    # Fix: Access train_args from the correct location
    # e2e_cfg_setup returns EasyDict(paths=output, args=arg_dict)
    # where arg_dict contains train_args, merge_args, etc.
    train_args = setup_dict.args.train_args  # This needs fixing in e2e_cfg_setup

    hf_args = None
    stage = getattr(args, 'stage', 'sft')
    if stage == "sft":
        raw_args: dict[str, any] = patch_sft_config(train_args)
        hf_args, *_ = HfArgumentParser(SFTArguments).parse_dict(
            raw_args, allow_extra_keys=True
        )
    return hf_args
```

**Note:** Also fix `e2e_cfg_setup` to return `arg_dict` properly:
```python
return EasyDict(paths=output, args=arg_dict)  # Pass arg_dict, not input args
```

---

## 3. File Changes Summary

| File | Action | Description |
|------|--------|-------------|
| `llmhalluc/hf.py` | Modify | Add missing EasyDict fields (override, do_train, etc.) |
| `llmhalluc/utils/cfg_utils.py` | Modify | Fix `save_config` -> `save_cfg`, fix return value |
| `llmhalluc/models/manager.py` | Modify | Add missing `return model` statement |
| `llmhalluc/data/utils.py` | Modify | Update path to `data/dataset_info.json` |
| `llmhalluc/data/sft.py` | Create | New `SFTDatasetConverter` class |
| `llmhalluc/data/__init__.py` | Modify | Export `SFTDatasetConverter` |
| `llmhalluc/data/manager.py` | Modify | Register `SFTDatasetConverter` in `DATASET_CONVERTERS` |
| `llmhalluc/train/sft.py` | Modify | Implement `setup_dataset()` using `SFTDatasetConverter` |
| `llmhalluc/train/base.py` | Modify | Store converter in `__init__` (if needed) |
| `llmhalluc/hparams/sft_args.py` | Modify | Add eval dataset fields and dataset key field |
| `llmhalluc/hparams/patcher.py` | Modify | Handle eval dataset resolution |

---

## 4. Implementation Order

1. **Phase 1 - Critical Fixes** (Must be done first)
   1. Task 1.2: Fix `save_config` -> `save_cfg` name collision
   2. Task 1.3: Fix `get_model` return statement
   3. Task 1.4: Fix dataset config path to use `data/dataset_info.json`
   4. Task 1.1: Complete argument setup in `hf.py`
   5. Task 3.2: Fix `hf_cfg_setup` and `e2e_cfg_setup` return values

2. **Phase 2 - Core Implementation**
   1. Task 2.4: Add eval dataset support to SFTArguments
   2. Task 3.1: Update `patch_sft_config` to handle all fields
   3. Task 2.1: Create `SFTDatasetConverter` in `llmhalluc/data/sft.py`
   4. Task 2.3: Update `BaseExecutor` to store converter (if needed)
   5. Task 2.2: Implement `SFTExecutor.setup_dataset()`

---

## 5. Design Decisions (Resolved)

1. **Chat Template**: Rely on TRL's internal handling with the `messages` format. No need to manually call `tokenizer.apply_chat_template()`.

2. **Converter Integration**: Create a new `SFTDatasetConverter` class that follows the naming style of other converters (e.g., `GSM8KDatasetConverter`, `SquadDatasetConverter`). This converter will transform the `{prompt, query, response}` format to TRL-compatible `{messages}` format.

3. **Eval Dataset Handling**: Handle train and eval datasets separately with different HuggingFace Hub URLs, aligning with the LlamaFactory setting where each dataset entry in `dataset_info.json` specifies its own path and split.

---

## 6. Existing Utilities to Leverage

The following existing utilities should be used in the implementation:

| Utility | Location | Purpose |
|---------|----------|---------|
| `get_dataset()` | `llmhalluc/data/manager.py` | Load datasets from HF Hub |
| `process_dataset()` | `llmhalluc/utils/data_utils.py` | Apply transformations with batching |
| `load_data_config()` | `llmhalluc/data/utils.py` | Load dataset configuration |
| `DatasetConverter` classes | `llmhalluc/data/` | Transform dataset examples |
| `get_model()`, `get_tokenizer()` | `llmhalluc/models/manager.py` | Load models with patching |
| `patch_model()`, `patch_tokenizer()` | `llmhalluc/models/patcher.py` | Apply model/tokenizer patches |

---

## 7. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| TRL version incompatibility | Training fails | Pin TRL version in requirements, test with specific version |
| Dataset format changes | Data loading fails | Add validation in converters, use existing dataset_info.json schema |
| Memory issues with large models | OOM errors | Add gradient checkpointing, support for quantization |
| Chat template variations | Incorrect formatting | Test with multiple model families (Llama, Qwen, etc.) |

---

## Approval Checklist

- [ ] Argument handling approach approved (EasyDict + config file)
- [ ] Data format handling approved (use existing converters + SFT formatter)
- [ ] Dataset config path change approved (`data/dataset_info.json`)
- [ ] Implementation order approved

Please review this plan and provide feedback before implementation begins.
