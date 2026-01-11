# GRPO Implementation Plan

## Executive Summary

This document outlines the implementation plan for integrating GRPO (Group Relative Policy Optimization) training into the Halluc codebase. GRPO is an online learning algorithm introduced in [DeepSeekMath](https://huggingface.co/papers/2402.03300) that enhances mathematical reasoning abilities while optimizing memory usage compared to PPO.

The implementation follows existing patterns from `SFTTrainer` and `DPOTrainer` and includes a modular reward function system similar to the data converter registry.

---

## 1. Overview

### 1.1 What is GRPO?

GRPO is a variant of PPO (Proximal Policy Optimization) that:
- Generates multiple completions for each prompt
- Computes advantages based on group-relative rewards
- Uses KL divergence to keep the model close to a reference policy
- Optimizes with a clipped policy gradient loss

Key features from TRL's `GRPOTrainer`:
- Support for custom reward functions (not just reward models)
- Multiple reward functions can be combined
- PEFT/LoRA support
- vLLM integration for faster generation

### 1.2 Implementation Goals

1. **GRPOArguments**: New arguments class inheriting from `GRPOConfig` and `BaseArguments`
2. **GRPOExecutor**: New executor class following the `SFTExecutor`/`DPOExecutor` pattern
3. **Reward Function Registry**: Similar to `DATASET_CONVERTERS`, a registry for reward functions
4. **Base Reward Function**: Abstract base class for implementing custom rewards
5. **PEFT Support**: Full LoRA/QLoRA compatibility
6. **vLLM Integration**: vLLM-powered generation for efficiency (default: colocate mode)

---

## 2. Architecture Overview

### 2.1 Directory Structure

```
llmhalluc/
├── data/
│   ├── __init__.py          # Export GRPODatasetConverter
│   ├── grpo.py              # NEW: GRPO dataset converter
│   └── ...
├── hparams/
│   ├── __init__.py          # Export GRPOArguments, patch_grpo_config
│   ├── ft_args.py           # ADD: GRPOArguments class
│   ├── patcher.py           # ADD: patch_grpo_config function
│   └── parser.py            # UPDATE: Add GRPO stage handling
├── reward/                   # NEW: Reward function module
│   ├── __init__.py
│   ├── base.py              # Base reward function class
│   ├── manager.py           # Reward function registry
│   └── ...                  # Future reward implementations
└── train/
    ├── __init__.py          # Export run_grpo
    ├── base.py              # UPDATE: Add GRPO case in run_train()
    └── grpo.py              # NEW: GRPOExecutor class
```

### 2.2 Key Components

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           GRPO Training Flow                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────────────────┐ │
│  │ GRPOArguments│────▶│ GRPOExecutor │────▶│ trl.GRPOTrainer         │ │
│  │              │     │              │     │                          │ │
│  │ - reward_funcs     │ - setup_model│     │ - model                  │ │
│  │ - num_generations  │ - setup_data │     │ - reward_funcs           │ │
│  │ - beta             │ - setup_trainer    │ - reward_processing_cls  │ │
│  │ - loss_type        │              │     │ - peft_config            │ │
│  └──────────────┘     └──────────────┘     └──────────────────────────┘ │
│                              │                                           │
│                              ▼                                           │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                     Reward Function Registry                      │   │
│  ├──────────────────────────────────────────────────────────────────┤   │
│  │  REWARD_FUNCTIONS = {                                            │   │
│  │      "accuracy": AccuracyReward,                                 │   │
│  │      "format": FormatReward,                                     │   │
│  │      "length": LengthReward,                                     │   │
│  │      ...                                                         │   │
│  │  }                                                               │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Detailed Implementation

### 3.1 Phase 1: Reward Function System

#### 3.1.1 Create Base Reward Function (`llmhalluc/reward/base.py`)

```python
"""Base reward function for GRPO training."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from transformers import TrainerState


@dataclass
class BaseRewardFunction(ABC):
    """Base class for reward functions.

    Reward functions must implement __call__ to compute rewards for completions.
    The reward function receives prompts, completions, and any additional dataset
    columns as keyword arguments.

    Following TRL's reward function interface:
    https://huggingface.co/docs/trl/main/en/grpo_trainer#using-a-custom-reward-function

    Requirements:
    1. Accept prompts, completions, completions_ids, trainer_state as kwargs
    2. Accept any additional dataset columns as kwargs (use **kwargs)
    3. Return a list of floats (one reward per completion)
    4. Can return None if reward is not applicable (for multi-task training)

    Attributes:
        name: Identifier for the reward function (used in logging)
        weight: Weight for combining with other reward functions (default: 1.0)
    """

    name: str = "base_reward"
    weight: float = 1.0

    @abstractmethod
    def __call__(
        self,
        prompts: list[str] | list[list[dict[str, str]]],
        completions: list[str] | list[list[dict[str, str]]],
        completions_ids: list[list[int]] | None = None,
        trainer_state: TrainerState | None = None,
        **kwargs: Any,
    ) -> list[float] | None:
        """Compute rewards for the given completions.

        Args:
            prompts: List of prompts (either plain text or conversational format)
            completions: List of completions generated for each prompt
            completions_ids: Tokenized completion IDs (for token-level rewards)
            trainer_state: Current trainer state (for curriculum learning)
            **kwargs: Additional dataset columns (e.g., ground_truth, labels)

        Returns:
            List of float rewards, one per completion.
            Return None if reward is not applicable to these samples.
        """
        pass

    @property
    def processing_class(self) -> Any | None:
        """Return the processing class for this reward function.

        Only needed when reward function is a model-based reward.
        For custom reward functions, this should return None.

        Returns:
            PreTrainedTokenizer or None
        """
        return None
```

#### 3.1.2 Create Reward Manager (`llmhalluc/reward/manager.py`)

```python
"""Reward function registry and manager for GRPO training."""

from typing import Any

from .base import BaseRewardFunction

# Registry of available reward functions
# Keys are string identifiers used in config files
# Values are reward function classes
REWARD_FUNCTIONS: dict[str, type[BaseRewardFunction]] = {}


def register_reward(name: str):
    """Decorator to register a reward function.

    Usage:
        @register_reward("accuracy")
        class AccuracyReward(BaseRewardFunction):
            ...

    Args:
        name: Unique identifier for the reward function

    Returns:
        Decorator function
    """
    def decorator(cls: type[BaseRewardFunction]) -> type[BaseRewardFunction]:
        if name in REWARD_FUNCTIONS:
            raise ValueError(f"Reward function '{name}' is already registered.")
        REWARD_FUNCTIONS[name] = cls
        return cls
    return decorator


def get_reward_function(name: str, **kwargs: Any) -> BaseRewardFunction:
    """Get a reward function instance by name.

    Args:
        name: Name of the registered reward function
        **kwargs: Arguments to pass to the reward function constructor

    Returns:
        Instantiated reward function

    Raises:
        ValueError: If reward function name not found
    """
    if name not in REWARD_FUNCTIONS:
        available = ", ".join(REWARD_FUNCTIONS.keys()) or "none"
        raise ValueError(
            f"Reward function '{name}' not found. "
            f"Available reward functions: {available}"
        )
    return REWARD_FUNCTIONS[name](**kwargs)


def get_reward_functions(
    names: str | list[str],
    **kwargs: Any,
) -> list[BaseRewardFunction]:
    """Get multiple reward function instances from comma-separated names.

    Args:
        names: Comma-separated string of reward function names, or list of names
        **kwargs: Arguments to pass to all reward function constructors

    Returns:
        List of instantiated reward functions

    Example:
        >>> get_reward_functions("accuracy,format")
        [AccuracyReward(...), FormatReward(...)]
    """
    if isinstance(names, str):
        names = [n.strip() for n in names.split(",") if n.strip()]

    return [get_reward_function(name, **kwargs) for name in names]


def list_reward_functions() -> list[str]:
    """List all registered reward function names.

    Returns:
        List of reward function identifiers
    """
    return list(REWARD_FUNCTIONS.keys())
```

#### 3.1.3 Create Reward Module Init (`llmhalluc/reward/__init__.py`)

```python
"""Reward functions for GRPO training."""

from .base import BaseRewardFunction
from .manager import (
    REWARD_FUNCTIONS,
    get_reward_function,
    get_reward_functions,
    list_reward_functions,
    register_reward,
)

__all__ = [
    "BaseRewardFunction",
    "REWARD_FUNCTIONS",
    "get_reward_function",
    "get_reward_functions",
    "list_reward_functions",
    "register_reward",
]
```

---

### 3.2 Phase 2: GRPO Arguments

#### 3.2.1 Add GRPOArguments to `ft_args.py`

```python
from trl import GRPOConfig as BaseGRPOConfig

@dataclass
class GRPOArguments(BaseArguments, BaseGRPOConfig):
    """Arguments for Group Relative Policy Optimization (GRPO).

    Inherits GRPO-specific parameters from TRL's GRPOConfig including:
    - num_generations: Number of completions per prompt (default: 8)
    - max_completion_length: Max length of generated completions (default: 256)
    - beta: KL coefficient (default: 0.0, meaning no reference model)
    - loss_type: GRPO loss type ("grpo", "dr_grpo", "dapo", etc.)
    - temperature: Sampling temperature (default: 1.0)
    - epsilon: Clipping epsilon (default: 0.2)
    """

    run_name: str = field(default="grpo")
    tags: list[str] = field(default_factory=list)
    model_name_or_path: str = field(
        default=None, metadata={"help": "The model name or path to use for training"}
    )

    tokenizer_name_or_path: str = field(
        default=None,
        metadata={"help": "The tokenizer name or path to use for training"},
    )

    # Train dataset key (looks up hf_hub_url from dataset_info.json)
    dataset: str | None = field(
        default=None,
        metadata={"help": "The train dataset key in dataset_info.json"},
    )

    # Eval dataset key (looks up hf_hub_url from dataset_info.json)
    eval_dataset: str | None = field(
        default=None,
        metadata={"help": "The eval dataset key in dataset_info.json"},
    )

    config_path: str | Path = field(
        default=None,
        metadata={"help": "The path to save the config"},
    )

    load_from_cache_file: bool = field(
        default=True,
        metadata={"help": "Whether to load dataset from cache file"},
    )

    converter: str = field(
        default="grpo",
        metadata={"help": "The converter to use for dataset conversion"},
    )

    finetuning_type: str = field(
        default="full",
        metadata={"help": "The fine-tuning method to use: full, lora, or qlora"},
    )

    ### Lora
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target: str = "all"

    # QLoRA (4-bit quantization)
    load_in_4bit: bool = False
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True

    # GRPO-specific: Reward Function Configuration
    reward_funcs: str = field(
        default="",
        metadata={
            "help": (
                "Comma-separated list of reward function names. "
                "Each name must be registered in REWARD_FUNCTIONS. "
                "Example: 'accuracy,format,length'"
            )
        },
    )

    reward_weights: str | None = field(
        default=None,
        metadata={
            "help": (
                "Comma-separated list of weights for reward functions. "
                "Must match the number of reward_funcs. "
                "Example: '1.0,0.5,0.2'"
            )
        },
    )

    # ==========================================
    # vLLM Configuration (for efficient generation)
    # ==========================================
    use_vllm: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to use vLLM for generating completions. "
                "Significantly speeds up generation. Requires vllm to be installed."
            )
        },
    )

    vllm_mode: str = field(
        default="colocate",
        metadata={
            "help": (
                "Mode for vLLM integration. Options: 'colocate' or 'server'. "
                "'colocate' (default): vLLM runs in the same process and shares training GPUs. "
                "'server': Sends generation requests to a separate vLLM server."
            )
        },
    )

    vllm_gpu_memory_utilization: float = field(
        default=0.3,
        metadata={
            "help": (
                "GPU memory utilization for vLLM in colocate mode. "
                "Lower values leave more memory for training. Default: 0.3"
            )
        },
    )

    vllm_max_model_length: int | None = field(
        default=None,
        metadata={
            "help": (
                "Context window for vLLM. Should be at least max prompt length + max_completion_length. "
                "If None, inferred from model config."
            )
        },
    )

    vllm_tensor_parallel_size: int = field(
        default=1,
        metadata={
            "help": (
                "Tensor parallel size for vLLM in colocate mode. "
                "Set > 1 for multi-GPU inference."
            )
        },
    )

    vllm_enable_sleep_mode: bool = field(
        default=False,
        metadata={
            "help": (
                "Enable vLLM sleep mode to offload weights/cache during optimizer step. "
                "Keeps GPU memory usage low but adds latency."
            )
        },
    )

    # vLLM Server mode settings (only used when vllm_mode='server')
    vllm_server_host: str = field(
        default="0.0.0.0",
        metadata={"help": "Host of the vLLM server. Only used when vllm_mode='server'."},
    )

    vllm_server_port: int = field(
        default=8000,
        metadata={"help": "Port of the vLLM server. Only used when vllm_mode='server'."},
    )

    vllm_server_timeout: float = field(
        default=240.0,
        metadata={
            "help": "Timeout in seconds to wait for vLLM server. Only used when vllm_mode='server'."
        },
    )

    # Special Token Initialization
    init_special_tokens: bool = field(
        default=False,
        metadata={"help": "Whether to initialize special tokens"},
    )
    new_special_tokens_config: dict[str, str] | None = field(
        default=None,
        metadata={
            "help": "Dict mapping special tokens to descriptions for embedding init"
        },
    )
    replace_text: dict[str, str] | None = field(
        default=None,
        metadata={"help": "Dict mapping default tokens to target tokens in dataset"},
    )
    template: str | None = field(
        default=None,
        metadata={"help": "Model template name for SPECIAL_TOKEN_MAPPING fallback"},
    )

    @property
    def yaml_exclude(self):
        """Fields to exclude from YAML serialization."""
        excludes = set()
        if not self.init_special_tokens:
            excludes.add("init_special_tokens")
            excludes.add("new_special_tokens_config")
            excludes.add("replace_text")
        return excludes

    def __post_init__(self):
        super().__post_init__()
        if self.model_name_or_path is None:
            raise ValueError("model_name_or_path is required")
        if self.dataset is None:
            raise ValueError("dataset is required")
        if not self.reward_funcs:
            raise ValueError(
                "reward_funcs is required for GRPO training. "
                "Provide comma-separated reward function names."
            )

    def get_reward_funcs_list(self) -> list[str]:
        """Parse reward_funcs string into list of names."""
        if not self.reward_funcs:
            return []
        return [n.strip() for n in self.reward_funcs.split(",") if n.strip()]

    def get_reward_weights_list(self) -> list[float] | None:
        """Parse reward_weights string into list of floats."""
        if not self.reward_weights:
            return None
        return [float(w.strip()) for w in self.reward_weights.split(",") if w.strip()]
```

#### 3.2.2 Update `hparams/__init__.py`

```python
from .ft_args import DPOArguments, GRPOArguments, SFTArguments
from .patcher import patch_configs, patch_dpo_config, patch_grpo_config, patch_sft_config

__all__ = [
    "EvaluationArguments",
    "MergeArguments",
    "TrainArguments",
    "SFTArguments",
    "DPOArguments",
    "GRPOArguments",  # NEW
    "patch_configs",
    "patch_sft_config",
    "patch_dpo_config",
    "patch_grpo_config",  # NEW
]
```

---

### 3.3 Phase 3: Dataset Converter

#### 3.3.1 Create GRPO Dataset Converter (`llmhalluc/data/grpo.py`)

```python
"""GRPO dataset converter for TRL GRPOTrainer."""

from dataclasses import dataclass
from typing import Any

from .base import DatasetConverter


@dataclass
class GRPODatasetConverter(DatasetConverter):
    """Converter that transforms datasets to GRPO-compatible format.

    GRPO requires a dataset with a 'prompt' column. Additional columns
    can be passed to reward functions during training.

    The converter maps configurable input keys to the expected format.

    Supports both:
    - Standard format: plain text prompts
    - Conversational format: messages-style prompts

    Args:
        prompt_key: Key for the prompt/input in the source dataset.
        query_key: Optional key for user query (combined with prompt).
        ground_truth_key: Optional key for ground truth (passed to reward funcs).
    """

    prompt_key: str = "prompt"
    query_key: str = "query"
    ground_truth_key: str | None = None

    # Alternative parameter names for flexibility
    prompt: str | None = None
    query: str | None = None
    ground_truth: str | None = None

    def __post_init__(self):
        self.prompt_key = self.prompt or self.prompt_key
        self.query_key = self.query or self.query_key
        self.ground_truth_key = self.ground_truth or self.ground_truth_key

    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        """Convert an example to GRPO format.

        Args:
            example: Input example with prompt and optional query/ground_truth.

        Returns:
            Example with 'prompt' field and optional additional fields.
        """
        prompt = example.get(self.prompt_key, "")
        query = example.get(self.query_key, "")

        # Combine query and prompt if both exist
        if query and prompt:
            combined_prompt = query + "\n" + prompt
        elif query:
            combined_prompt = query
        else:
            combined_prompt = prompt

        result = {"prompt": combined_prompt}

        # Pass through ground truth for reward functions
        if self.ground_truth_key and self.ground_truth_key in example:
            result["ground_truth"] = example[self.ground_truth_key]

        # Pass through any additional columns that might be needed by reward functions
        # Common columns: labels, categories, metadata
        for key in ["labels", "category", "metadata", "answer"]:
            if key in example and key not in result:
                result[key] = example[key]

        return result
```

#### 3.3.2 Register in `data/manager.py`

```python
from .grpo import GRPODatasetConverter

DATASET_CONVERTERS = {
    "squad": SquadDatasetConverter,
    "backtrack": BacktrackDatasetConverter,
    "gsm8k": GSM8KDatasetConverter,
    "gsm8k_symbolic_backtrack": GSM8KSymbolicDatasetConverter,
    "gsm8k_backtrack": GSM8KBacktrackDatasetConverter,
    "sft": SFTDatasetConverter,
    "dpo": DPODatasetConverter,
    "grpo": GRPODatasetConverter,  # NEW
}
```

#### 3.3.3 Update `data/__init__.py`

```python
from .grpo import GRPODatasetConverter

__all__ = [
    "DatasetConverter",
    "DATASET_CONVERTERS",
    "get_dataset_converter",
    "get_dataset",
    "load_data_config",
    "SFTDatasetConverter",
    "DPODatasetConverter",
    "GRPODatasetConverter",  # NEW
]
```

---

### 3.4 Phase 4: GRPO Executor

#### 3.4.1 Create GRPO Executor (`llmhalluc/train/grpo.py`)

```python
"""GRPO training executor for TRL GRPOTrainer."""

import logging
from collections.abc import Callable

from trl import GRPOTrainer

from llmhalluc.hparams.ft_args import GRPOArguments
from llmhalluc.reward import get_reward_functions

from .base import BaseExecutor

logger = logging.getLogger(__name__)


class GRPOExecutor(BaseExecutor):
    """Executor for Group Relative Policy Optimization (GRPO) training.

    GRPO is an online RL algorithm that generates multiple completions
    per prompt and uses group-relative rewards to compute advantages.

    This executor handles:
    - Model and tokenizer setup (inherited from BaseExecutor)
    - Dataset loading and conversion to GRPO format
    - Reward function instantiation from registry
    - GRPOTrainer setup with PEFT support
    """

    def __init__(self, args: GRPOArguments):
        self.reward_functions = None
        self.reward_processing_classes = None
        super().__init__(args)

    def __post_init__(self):
        """Setup reward functions after base initialization."""
        self._setup_reward_functions()

    def _setup_reward_functions(self):
        """Initialize reward functions from the registry.

        Parses the reward_funcs argument (comma-separated names) and
        instantiates each reward function from REWARD_FUNCTIONS registry.
        """
        reward_func_names = self.args.get_reward_funcs_list()
        if not reward_func_names:
            raise ValueError(
                "No reward functions specified. "
                "Set reward_funcs in config (comma-separated names)."
            )

        logger.info(f"Loading reward functions: {reward_func_names}")

        self.reward_functions = get_reward_functions(reward_func_names)

        # Build reward_processing_classes list (None for custom functions)
        self.reward_processing_classes = [
            rf.processing_class for rf in self.reward_functions
        ]

        # Log reward weights if specified
        reward_weights = self.args.get_reward_weights_list()
        if reward_weights:
            if len(reward_weights) != len(self.reward_functions):
                raise ValueError(
                    f"Number of reward_weights ({len(reward_weights)}) must match "
                    f"number of reward_funcs ({len(self.reward_functions)})"
                )
            logger.info(f"Reward weights: {reward_weights}")

    def _get_trainer_class(self):
        """Return GRPOTrainer class for GRPO training."""
        return GRPOTrainer

    def _get_callable_reward_funcs(self) -> list[Callable]:
        """Get list of callable reward functions for GRPOTrainer.

        Returns:
            List of callables that match GRPOTrainer's reward_funcs interface.
        """
        return [rf.__call__ for rf in self.reward_functions]

    def setup_trainer(self):
        """Setup GRPOTrainer with model, tokenizer, dataset, and reward functions.

        Unlike SFT/DPO trainers, GRPO requires:
        - reward_funcs: List of reward functions (callables or model paths)
        - reward_processing_classes: Tokenizers for model-based rewards
        """
        trainer_class = self._get_trainer_class()

        # Build LoRA config if using PEFT
        peft_config = self._get_peft_config()

        # Get callable reward functions
        reward_funcs = self._get_callable_reward_funcs()

        # Get reward weights if specified
        reward_weights = self.args.get_reward_weights_list()

        # Build callbacks
        from .callbacks import get_callbacks
        callbacks = get_callbacks(self.args)

        # Filter out None processing classes (for custom reward functions)
        processing_classes = [
            pc for pc in self.reward_processing_classes if pc is not None
        ]

        trainer_kwargs = {
            "model": self.model,
            "args": self.args,
            "processing_class": self.tokenizer,
            "train_dataset": self.dataset.get("train"),
            "eval_dataset": self.dataset.get("eval"),
            "reward_funcs": reward_funcs,
            "peft_config": peft_config,
            "callbacks": callbacks if callbacks else None,
        }

        # Add reward_processing_classes only if we have model-based rewards
        if processing_classes:
            trainer_kwargs["reward_processing_classes"] = processing_classes

        # Add reward_weights if specified
        if reward_weights:
            # Note: GRPOConfig handles reward_weights, not trainer init
            # This is set via args which inherits from GRPOConfig
            pass

        self.trainer = trainer_class(**trainer_kwargs)


def run_grpo(args: GRPOArguments):
    """Run GRPO training with the given arguments.

    Args:
        args: GRPOArguments containing all training configuration.
    """
    executor = GRPOExecutor(args=args)
    executor.on_train_start()
    executor.fit()
    executor.on_train_end()
```

#### 3.4.2 Update `train/__init__.py`

```python
from .base import run_train
from .grpo import run_grpo

__all__ = [
    "run_train",
    "run_grpo",
]
```

#### 3.4.3 Update `train/base.py` - Add GRPO Case

```python
def run_train(args):
    if isinstance(args, SFTArguments):
        from .sft import run_sft
        run_sft(args)
    elif isinstance(args, DPOArguments):
        from .dpo import run_dpo
        run_dpo(args)
    elif isinstance(args, GRPOArguments):  # NEW
        from .grpo import run_grpo
        run_grpo(args)
    else:
        raise ValueError(f"Unknown support argument type: {type(args)}")
    logger.info("Training completed successfully")
```

Also add the import at the top:
```python
from llmhalluc.hparams import DPOArguments, GRPOArguments, SFTArguments
```

---

### 3.5 Phase 5: Parser and Config Integration

#### 3.5.1 Add `patch_grpo_config` Function

Add to `llmhalluc/hparams/patcher.py`:

```python
def patch_grpo_config(args) -> dict[str, any]:
    """Patch GRPO configuration with resolved dataset paths.

    Args:
        args: TrainArguments or dict containing GRPO config

    Returns:
        Dictionary with resolved configuration for GRPOArguments
    """
    if isinstance(args, BaseArguments):
        arg_dict = args.to_yaml()
    else:
        arg_dict = dict(args)

    from llmhalluc.data import load_data_config
    data_config = load_data_config()

    # Resolve tokenizer path
    if "tokenizer_name_or_path" not in arg_dict or not arg_dict.get("tokenizer_name_or_path"):
        arg_dict["tokenizer_name_or_path"] = arg_dict.get("model_name_or_path")

    # Resolve train dataset
    dataset_name = arg_dict.get("dataset")
    if dataset_name:
        dataset_info = data_config.get(dataset_name)
        if dataset_info is None:
            raise ValueError(f"Dataset '{dataset_name}' not found in data config")

    # Resolve eval dataset
    eval_dataset_name = arg_dict.get("eval_dataset")
    if eval_dataset_name:
        eval_dataset_info = data_config.get(eval_dataset_name)
        if eval_dataset_info is None:
            raise ValueError(f"Eval dataset '{eval_dataset_name}' not found in data config")

    # Validate reward_funcs
    reward_funcs = arg_dict.get("reward_funcs", "")
    if not reward_funcs:
        raise ValueError("reward_funcs must be specified for GRPO training")

    return arg_dict
```

#### 3.5.2 Update `parser.py` - Add GRPO Stage

```python
from .ft_args import DPOArguments, GRPOArguments, SFTArguments
from .patcher import patch_configs, patch_dpo_config, patch_grpo_config, patch_sft_config

def hf_cfg_setup(
    config_path: str,
    save_cfg: bool = True,
    cli_args: list[str] | None = None,
) -> EasyDict:
    """Setup HuggingFace training config with optional CLI overrides."""
    setup_dict = e2e_cfg_setup(config_path, save_cfg=save_cfg, cli_args=cli_args)
    train_args = setup_dict.args.train_args

    hf_args = None
    stage = getattr(train_args, "stage", "sft")

    if stage == "sft":
        raw_args: dict[str, any] = patch_sft_config(train_args)
        hf_args, *_ = HfArgumentParser(SFTArguments).parse_dict(
            raw_args, allow_extra_keys=True
        )
        if save_cfg:
            save_ft_config(hf_args, hf_args.config_path)

    elif stage == "dpo":
        raw_args: dict[str, any] = patch_dpo_config(train_args)
        hf_args, *_ = HfArgumentParser(DPOArguments).parse_dict(
            raw_args, allow_extra_keys=True
        )
        if save_cfg:
            save_ft_config(hf_args, hf_args.config_path)

    elif stage == "grpo":  # NEW
        raw_args: dict[str, any] = patch_grpo_config(train_args)
        hf_args, *_ = HfArgumentParser(GRPOArguments).parse_dict(
            raw_args, allow_extra_keys=True
        )
        if save_cfg:
            save_ft_config(hf_args, hf_args.config_path)

    else:
        raise ValueError(f"Unsupported stage: {stage}")

    setup_dict.args.hf_args = hf_args
    return setup_dict
```

---

## 4. Key Design Decisions

### 4.1 Reward Function Interface

Following TRL's interface, reward functions must:

1. **Accept standardized kwargs**: `prompts`, `completions`, `completions_ids`, `trainer_state`, plus dataset columns
2. **Return list of floats**: One reward per completion
3. **Support None returns**: For multi-task training where some rewards don't apply

### 4.2 `reward_funcs` Argument

The `reward_funcs` argument takes a comma-separated string of registered function names:

```yaml
# In config file
reward_funcs: "accuracy,format,length"
reward_weights: "1.0,0.5,0.2"
```

This differs from TRL's API which accepts callable objects directly. Our approach:
1. Parse string → list of names
2. Look up each name in `REWARD_FUNCTIONS` registry
3. Instantiate and pass callables to GRPOTrainer

### 4.3 `reward_processing_classes`

Per TRL documentation:
- Only needed for **model-based** reward functions (PreTrainedModel)
- For custom callable functions, this is **ignored**
- Set via `processing_class` property on `BaseRewardFunction`

Since we initially focus on custom reward functions, this will typically be `None`.

### 4.4 PEFT Support

GRPO supports PEFT training through the same mechanism as SFT/DPO:
- `peft_config` passed to GRPOTrainer constructor
- `_get_peft_config()` inherited from BaseExecutor
- Same `lora_*` and `bnb_4bit_*` arguments

### 4.5 vLLM Integration

vLLM is enabled by default for efficient generation during training. Key design decisions:

#### Default Mode: Colocate
- **`use_vllm: true`** - vLLM is enabled by default
- **`vllm_mode: "colocate"`** - vLLM runs in the same process as training
- No separate server needed, simplifies deployment
- Shares GPU memory with training process

#### Colocate Mode Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `vllm_gpu_memory_utilization` | 0.3 | Fraction of GPU memory for vLLM (rest for training) |
| `vllm_tensor_parallel_size` | 1 | Number of GPUs for tensor parallelism |
| `vllm_max_model_length` | None | Context window (auto-detected if None) |
| `vllm_enable_sleep_mode` | False | Offload weights during optimizer step |

#### Server Mode (Alternative)
For advanced setups, use `vllm_mode: "server"` with a separate vLLM server:
```bash
# Start vLLM server first
trl vllm-serve --model Qwen/Qwen2-0.5B-Instruct
```

#### Performance Considerations
- **Colocate mode** is simpler but may have resource contention
- **Server mode** avoids contention but requires separate process management
- For most use cases, **colocate with `gpu_memory_utilization=0.3`** works well

---

## 5. Example Configuration

### 5.1 GRPO Training Config with vLLM (`configs/grpo_example.yaml`)

```yaml
# GRPO Training Configuration
stage: grpo

# Model
model_name_or_path: "Qwen/Qwen2-0.5B-Instruct"

# Dataset
dataset: "gsm8k_train"  # Key in dataset_info.json
eval_dataset: "gsm8k_test"
converter: "grpo"

# Training
output_dir: "outputs/grpo"
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
num_train_epochs: 3
learning_rate: 1e-6
bf16: true

# GRPO-specific
num_generations: 8
max_completion_length: 256
temperature: 1.0
beta: 0.001  # KL coefficient
loss_type: "dapo"  # or "grpo", "dr_grpo"

# Reward functions
reward_funcs: "accuracy"
# reward_weights: "1.0"  # Optional, defaults to equal weighting

# vLLM Configuration (enabled by default for efficiency)
use_vllm: true
vllm_mode: "colocate"  # vLLM runs in same process as training
vllm_gpu_memory_utilization: 0.3  # Reserve 30% GPU memory for vLLM
# vllm_tensor_parallel_size: 1  # Increase for multi-GPU inference
# vllm_enable_sleep_mode: false  # Enable to reduce memory at cost of latency

# PEFT (optional)
finetuning_type: "lora"
lora_rank: 8
lora_alpha: 16

# Logging
report_to: "wandb"
logging_steps: 10
save_strategy: "steps"
save_steps: 500
```

### 5.2 GRPO with vLLM Server Mode (`configs/grpo_server.yaml`)

```yaml
# GRPO with separate vLLM server (for advanced setups)
stage: grpo
model_name_or_path: "Qwen/Qwen2-7B-Instruct"

# vLLM Server Mode
use_vllm: true
vllm_mode: "server"
vllm_server_host: "localhost"
vllm_server_port: 8000
vllm_server_timeout: 300.0

# ... rest of config
```

### 5.3 GRPO without vLLM (fallback)

```yaml
# Disable vLLM for debugging or compatibility
stage: grpo
model_name_or_path: "Qwen/Qwen2-0.5B-Instruct"

use_vllm: false  # Use model.generate() instead

# ... rest of config
```

---

## 6. Future Work

### 6.1 Planned Reward Functions

After the base infrastructure is complete, implement common reward functions:

| Name | Description | Use Case |
|------|-------------|----------|
| `accuracy` | Checks answer correctness (regex-based) | Math reasoning |
| `format` | Validates output format (e.g., `<think>...</think>`) | CoT reasoning |
| `length` | Rewards/penalizes completion length | Control verbosity |
| `fluency` | Perplexity-based fluency score | General quality |
| `no_hallucination` | Checks for factual consistency | Hallucination reduction |

### 6.2 Model-Based Rewards

Add support for pretrained reward models:

```python
REWARD_FUNCTIONS = {
    # Custom functions
    "accuracy": AccuracyReward,
    # Model-based rewards (auto-loaded from HF)
    "reward_model": "path/to/reward-model",
}
```

### 6.3 Advanced vLLM Configurations

Future enhancements for vLLM integration:
- **Structured outputs**: Use `vllm_structured_outputs_regex` for constrained generation
- **Multi-node training**: Scale to 70B+ models with distributed vLLM
- **Async reward computation**: Overlap reward calculation with generation

---

## 7. File Changes Summary

| File | Action | Description |
|------|--------|-------------|
| `llmhalluc/reward/__init__.py` | **Create** | Reward module exports |
| `llmhalluc/reward/base.py` | **Create** | BaseRewardFunction class |
| `llmhalluc/reward/manager.py` | **Create** | Reward registry and manager |
| `llmhalluc/data/grpo.py` | **Create** | GRPODatasetConverter |
| `llmhalluc/data/manager.py` | Modify | Register GRPODatasetConverter |
| `llmhalluc/data/__init__.py` | Modify | Export GRPODatasetConverter |
| `llmhalluc/hparams/ft_args.py` | Modify | Add GRPOArguments class |
| `llmhalluc/hparams/patcher.py` | Modify | Add patch_grpo_config |
| `llmhalluc/hparams/__init__.py` | Modify | Export GRPOArguments, patch_grpo_config |
| `llmhalluc/hparams/parser.py` | Modify | Add GRPO stage handling |
| `llmhalluc/train/grpo.py` | **Create** | GRPOExecutor class |
| `llmhalluc/train/__init__.py` | Modify | Export run_grpo |
| `llmhalluc/train/base.py` | Modify | Add GRPO case in run_train |

---

## 8. Implementation Order

### Phase 1: Foundation (Days 1-2)
1. Create reward module structure
2. Implement `BaseRewardFunction`
3. Implement reward manager with registry

### Phase 2: Arguments (Day 2)
4. Add `GRPOArguments` class
5. Add `patch_grpo_config` function
6. Update parser for GRPO stage

### Phase 3: Data (Day 3)
7. Create `GRPODatasetConverter`
8. Register in manager

### Phase 4: Training (Days 3-4)
9. Implement `GRPOExecutor`
10. Update `run_train` function
11. Test with dummy reward function

### Phase 5: Integration (Day 5)
12. Create example config
13. End-to-end testing
14. Documentation

---

## 9. Approval Checklist

- [ ] Reward function registry pattern approved
- [ ] `reward_funcs` as comma-separated string approved
- [ ] GRPOArguments field set approved
- [ ] GRPOExecutor design approved
- [ ] Dataset converter approach approved
- [ ] Implementation order approved

Please review this plan and provide feedback before implementation begins.
