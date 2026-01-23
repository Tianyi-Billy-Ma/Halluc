from datasets import Dataset, DatasetDict, load_dataset

from llmhalluc.utils import wrap_converter_with_replace

from .backtrack import (
    BacktrackDatasetConverter,
    RandomBacktrackConverter,
    SFTBacktrackDatasetConverter,
)
from .base import DatasetConverter
from .dpo import DPODatasetConverter
from .grpo import GRPODatasetConverter
from .gsm8k import (
    GSM8KBacktrackDatasetConverter,
    GSM8KDatasetConverter,
    GSM8KSymbolicDatasetConverter,
)
from .sft import SFTDatasetConverter
from .squad import SquadDatasetConverter

DATASET_CONVERTERS = {
    "squad": SquadDatasetConverter,
    "backtrack": BacktrackDatasetConverter,
    "random_backtrack": RandomBacktrackConverter,
    "gsm8k": GSM8KDatasetConverter,
    "gsm8k_symbolic_backtrack": GSM8KSymbolicDatasetConverter,
    "gsm8k_backtrack": GSM8KBacktrackDatasetConverter,
    "sft": SFTDatasetConverter,
    "dpo": DPODatasetConverter,
    "grpo": GRPODatasetConverter,
    "squad_v2": SquadDatasetConverter,
    "sft_backtrack": SFTBacktrackDatasetConverter,
}


def get_dataset_converter(name: str, **kwargs) -> DatasetConverter:
    """Get a dataset converter instance.

    Args:
        name: Name of the converter.
        **kwargs: Arguments to pass to the converter constructor.

    Returns:
        Converter instance.

    Raises:
        ValueError: If converter name not found.
    """
    if name not in DATASET_CONVERTERS:
        raise ValueError(f"Dataset converter {name} not found.")

    mapping_args = {
        "batched": name in ["gsm8k_backtrack"],
        "batch_size": 1 if name in ["gsm8k_backtrack"] else None,
    }
    return DATASET_CONVERTERS[name](**kwargs), mapping_args


def get_dataset(
    dataset_path: str,
    name: str | None = None,
    split: str | None = None,
    converter: DatasetConverter | None = None,
    converter_args: dict | None = None,
    replace_text: dict[str, str] | None = None,
) -> Dataset | DatasetDict:
    dataset = load_dataset(dataset_path, name=name, split=split)
    if converter:
        mapping_args = {}
        if isinstance(converter, str):
            converter, mapping_args = get_dataset_converter(converter, **converter_args)

        if replace_text:
            converter = wrap_converter_with_replace(
                converter,
                replace_text,
                mapping_args.get("batched", False),
            )

        dataset = dataset.map(
            converter, remove_columns=dataset.column_names, **mapping_args
        )
    return dataset
