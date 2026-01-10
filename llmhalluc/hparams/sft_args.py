from dataclasses import dataclass, field
from pathlib import Path
from trl import SFTConfig as BaseSFTConfig

from .base_args import BaseArguments


@dataclass
class SFTArguments(BaseArguments, BaseSFTConfig):
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

    def __post_init__(self):
        super().__post_init__()
        if self.model_name_or_path is None:
            raise ValueError("model_name_or_path is required")
        if self.dataset is None:
            raise ValueError("dataset is required")
