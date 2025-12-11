"""Dataset converters for llmhalluc."""

from .base import DatasetConverter
from .squad import SquadDatasetConverter
from .backtrack import BacktrackDatasetConverter, BACKTRACK_TOKEN
from .gsm8k import GSM8KDatasetConverter, GSM8KSymbolicDatasetConverter, GSM8KBacktrackDatasetConverter

# Registry of available converters
DATASET_CONVERTERS = {
    "squad": SquadDatasetConverter,
    "backtrack": BacktrackDatasetConverter,
    "gsm8k": GSM8KDatasetConverter,
    "gsm8k_symbolic_backtrack": GSM8KSymbolicDatasetConverter,
    "gsm8k_backtrack": GSM8KBacktrackDatasetConverter,
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

    return DATASET_CONVERTERS[name](**kwargs)


__all__ = [
    "DatasetConverter",
    "SquadDatasetConverter",
    "BacktrackDatasetConverter",
    "GSM8KSymbolicDatasetConverter",
    "BACKTRACK_TOKEN",
    "DATASET_CONVERTERS",
    "register_dataset_converter",
    "get_dataset_converter",
]
