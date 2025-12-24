from datasets import load_dataset, Dataset, DatasetDict

from .base import DatasetConverter
from .squad import SquadDatasetConverter
from .backtrack import BacktrackDatasetConverter
from .sft import SFTDatasetConverter
from .gsm8k import (
    GSM8KDatasetConverter,
    GSM8KSymbolicDatasetConverter,
    GSM8KBacktrackDatasetConverter,
)

DATASET_CONVERTERS = {
    "squad": SquadDatasetConverter,
    "backtrack": BacktrackDatasetConverter,
    "gsm8k": GSM8KDatasetConverter,
    "gsm8k_symbolic_backtrack": GSM8KSymbolicDatasetConverter,
    "gsm8k_backtrack": GSM8KBacktrackDatasetConverter,
    "sft": SFTDatasetConverter,
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

    converter_args = {
        "batched": name in ["gsm8k_backtrack"],
        "batch_size": 1 if name in ["gsm8k_backtrack"] else None,
    }
    return DATASET_CONVERTERS[name](**kwargs), converter_args


def get_dataset(
    dataset_path: str,
    name: str | None = None,
    split: str | None = None,
    converter: "DatasetConverter" | str | None = None,
) -> Dataset | DatasetDict:
    dataset = load_dataset(dataset_path, name=name, split=split)
    if converter:
        if isinstance(converter, str):
            converter, converter_args = get_dataset_converter(converter)
        dataset = dataset.map(
            converter, remove_columns=dataset.column_names, **converter_args
        )
    return dataset
