"""Fine-tuning arguments for SFT and DPO training."""

from dataclasses import dataclass, field
from pathlib import Path

from trl import DPOConfig as BaseDPOConfig
from trl import SFTConfig as BaseSFTConfig

from .base_args import BaseArguments


@dataclass
class SFTArguments(BaseArguments, BaseSFTConfig):
    """Arguments for Supervised Fine-Tuning (SFT)."""

    run_name: str = field(default="sft")
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
        default="sft",
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

    # Early Stopping
    early_stopping: bool = field(
        default=False,
        metadata={"help": "Enable early stopping when eval metric stops improving"},
    )
    early_stopping_patience: int = field(
        default=3,
        metadata={"help": "Number of eval steps with no improvement before stopping"},
    )
    early_stopping_threshold: float = field(
        default=0.0,
        metadata={"help": "Minimum change to qualify as an improvement"},
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


@dataclass
class DPOArguments(BaseArguments, BaseDPOConfig):
    """Arguments for Direct Preference Optimization (DPO).

    Inherits DPO-specific parameters from TRL's DPOConfig including:
    - beta: Temperature parameter for DPO loss (default: 0.1)
    - loss_type: Type of DPO loss ("sigmoid", "hinge", "ipo", etc.)
    - label_smoothing: Label smoothing factor (default: 0.0)
    """

    run_name: str = field(default="dpo")
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
        default="dpo",
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

    # Early Stopping
    early_stopping: bool = field(
        default=False,
        metadata={"help": "Enable early stopping when eval metric stops improving"},
    )
    early_stopping_patience: int = field(
        default=3,
        metadata={"help": "Number of eval steps with no improvement before stopping"},
    )
    early_stopping_threshold: float = field(
        default=0.0,
        metadata={"help": "Minimum change to qualify as an improvement"},
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
