"""LLM Hallucination project with backtrack functionality."""

from .data import (
    DatasetConverter,
    SquadDatasetConverter,
    BacktrackDatasetConverter,
    BACKTRACK_TOKEN,
    get_dataset_converter,
)
from .utils import process_dataset

__version__ = "0.1.0"

__all__ = [
    "DatasetConverter",
    "SquadDatasetConverter",
    "BacktrackDatasetConverter",
    "BACKTRACK_TOKEN",
    "get_dataset_converter",
    "process_dataset",
]
