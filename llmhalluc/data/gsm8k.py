"""Backtrack dataset converter."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from .base import DatasetConverter
from llmhalluc.prompts.MathPrompt import MATH_INSTRUCTION


@dataclass
class GSM8KDatasetConverter(DatasetConverter):
    """Converter for backtrack dataset processing.

    This converter implements the main novelty of the project - adding backtrack
    functionality to dataset examples by introducing random tokens and backtrack signals.

    Args:
        tokenizer: Tokenizer for encoding/decoding text.
        max_tokens: Maximum number of random tokens to add.
        no_spc_vocab: List of non-special token IDs for random selection.
        split: Dataset split being processed (affects randomization).
        key_mapping: Mapping of dataset keys to standard keys.
    """

    split: str = "train"
    key_mapping: dict[str, str] | None = None
    prompt: str = MATH_INSTRUCTION

    def __post_init__(self) -> None:
        """Initialize default values after dataclass creation."""
        if self.key_mapping is None:
            self.key_mapping = {"prompt": "prompt", "query": "question", "response": "answer"}
        if self.prompt is None:
            self.prompt = MATH_INSTRUCTION

    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        """Convert an example to backtrack format.

        Args:
            example: Input example with prompt, query, and response.

        Returns:
            Converted example with backtrack content and modified response.
        """
        # Extract fields using key mapping
        prompt = self.prompt
        query = example[self.key_mapping["query"]]
        response = example[self.key_mapping["response"]]

        return {
            "prompt": prompt,
            "query": query,
            "response": response,
        }
