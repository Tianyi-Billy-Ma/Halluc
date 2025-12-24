from dataclasses import dataclass, field
from trl import SFTConfig as BaseSFTConfig


@dataclass
class SFTArguments(BaseSFTConfig):
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

    def __post_init__(self):
        super().__post_init__()
        if self.model_name_or_path is None:
            raise ValueError("model_name_or_path is required")
        if self.dataset is None:
            raise ValueError("dataset is required")
