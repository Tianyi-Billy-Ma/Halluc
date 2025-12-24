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

    dataset_path: str = field(
        default=None, metadata={"help": "The dataset path to use for training"}
    )

    dataset_name: str | None = field(
        default=None, metadata={"help": "The dataset name to use for training"}
    )
    # eval_dataset_path: str | None = field(
    #     default=None, metadata={"help": "The eval dataset path to use for training"}
    # )
    # eval_dataset_name: str | None = field(
    #     default=None, metadata={"help": "The eval dataset name to use for training"}
    # )

    def __post_init__(self):
        super().__post_init__()
        if self.model_name_or_path is None:
            raise ValueError("model_name_or_path is required")
        if self.dataset_path is None:
            raise ValueError("dataset_path is required")
        if self.dataset_name is None:
            raise ValueError("dataset_name is required")
