"""Backtrack dataset converter."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from transformers import PreTrainedTokenizer

from .base import DatasetConverter

BACKTRACK_TOKEN = "<|BACKTRACK|>"


@dataclass
class BacktrackDatasetConverter(DatasetConverter):
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

    tokenizer: PreTrainedTokenizer
    max_tokens: int = 10
    no_spc_vocab: list[int] | None = None
    split: str = "train"
    key_mapping: dict[str, str] | None = None

    def __post_init__(self) -> None:
        """Initialize default values after dataclass creation."""
        if self.key_mapping is None:
            self.key_mapping = {
                "prompt": "prompt",
                "query": "query",
                "response": "response",
            }

        if self.no_spc_vocab is None:
            vocab = set(self.tokenizer.get_vocab().values())
            special_tokens = set(self.tokenizer.all_special_ids)
            self.no_spc_vocab = list(vocab - special_tokens)

    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        """Convert an example to backtrack format.

        Args:
            example: Input example with prompt, query, and response.

        Returns:
            Converted example with backtrack content and modified response.
        """
        # Extract fields using key mapping
        prompt = example[self.key_mapping["prompt"]]
        query = example[self.key_mapping["query"]]
        response = example[self.key_mapping["response"]]

        # Tokenize response
        response_token_ids = self.tokenizer.encode(response)
        backtrack_id = self.tokenizer.encode(BACKTRACK_TOKEN)[0]

        # Determine number of random tokens to add
        random_int = (
            np.random.randint(0, self.max_tokens) if self.split == "train" else 0
        )

        # If no wrong tokens, return original example
        if random_int == 0:
            return {
                "prompt": prompt,
                "query": query,
                "original_response": response,
                "response": response,
                "backtrack_content": "",
            }

        # Generate backtrack content
        random_split = np.random.randint(0, len(response_token_ids))
        np.random.shuffle(self.no_spc_vocab)

        backtrack_token_ids = (
            response_token_ids[:random_split] + self.no_spc_vocab[:random_int]
        )

        curr_response_token_ids = [backtrack_id] * random_int + response_token_ids[
            random_split:
        ]

        # Decode modified content
        backtrack_content = self.tokenizer.decode(backtrack_token_ids)
        modified_response = self.tokenizer.decode(curr_response_token_ids)

        return {
            "prompt": prompt,
            "query": query,
            "response": modified_response,
            "backtrack_content": backtrack_content,
        }




@dataclass
class SFTBacktrackDatasetConverter(DatasetConverter):

    prompt_key: str = "prompt"
    query_key: str = "query"
    response_prefix_key: str = "backtrack_prefix"
    response_suffix_key: str = "backtrack_suffix"


    prompt: str | None = None
    query: str | None = None
    response_prefix: str | None = None
    response_suffix: str | None = None


    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        """Convert an example to TRL messages format.

        Args:
            example: Input example with prompt, query, and response fields.

        Returns:
            Example with 'messages' field containing list of role/content dicts.
        """
        prompt_content = example.get(self.prompt_key, "")
        query_content = example.get(self.query_key, "")
        response_prefix_content = example.get(self.response_prefix_key, "")
        response_suffix_content = example.get(self.response_suffix_key, "")
        return {
            "prompt": prompt_content,
            "query": f"Question: \n{query_content}\n\nAnswer: \n{response_prefix_content}",
            "response": response_suffix_content,
        }
