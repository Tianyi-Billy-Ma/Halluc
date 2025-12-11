"""Base dataset converter following LLaMA-Factory patterns."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
from collections import defaultdict


@dataclass
class DatasetConverter(ABC):
    """Base class for dataset converters.

    This class follows the same pattern as LLaMA-Factory's DatasetConverter,
    providing a common interface for converting dataset examples to a standard format.

    Converters are responsible only for transforming individual examples,
    not for loading or saving datasets.
    """

    @abstractmethod
    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        """Convert a single example in the dataset to the standard format.

        Args:
            example: A single example from the dataset.

        Returns:
            Converted example in standard format with keys like:
            - prompt: System/instruction prompt
            - query: User query/question
            - response: Expected response
            - Additional converter-specific fields
        """
        pass

    def batch_to_list(self, examples: dict[str, Any]) -> list[dict[str, Any]]:
        batch_size = len(next(iter(examples.values())))
        res = []
        for i in range(batch_size):
            example = {k: v[i] for k, v in examples.items()}
            res.append(example)
        return res

    def list_to_batch(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        res = defaultdict(list)

        for example in examples:
            for k, v in example.items():
                res[k].append(v)
        return dict(res)
