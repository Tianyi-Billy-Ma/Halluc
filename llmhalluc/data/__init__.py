"""Dataset converters for llmhalluc."""

from .base import DatasetConverter
from .collator import BacktrackMaskingCollator
from .dpo import DPODatasetConverter
from .grpo import GRPODatasetConverter
from .manager import DATASET_CONVERTERS, get_dataset, get_dataset_converter
from .sft import SFTDatasetConverter
from .utils import load_data_config

# Registry of available converters


__all__ = [
    "DatasetConverter",
    "DATASET_CONVERTERS",
    "get_dataset_converter",
    "get_dataset",
    "load_data_config",
    "SFTDatasetConverter",
    "DPODatasetConverter",
    "GRPODatasetConverter",
    "BacktrackMaskingCollator",
]
