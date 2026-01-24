"""GRPO dataset converter for TRL GRPOTrainer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .base import DatasetConverter

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


@dataclass
class GRPODatasetConverter(DatasetConverter):
    """Converter that transforms datasets to GRPO-compatible format.

    GRPO requires a dataset with a 'prompt' column. Additional columns
    can be passed to reward functions during training.

    The converter maps configurable input keys to the expected format.

    Supports both:
    - Standard format: plain text prompts
    - Conversational format: messages-style prompts

    Args:
        prompt_key: Key for the prompt/input in the source dataset.
        query_key: Optional key for user query (combined with prompt).
        ground_truth_key: Key for ground truth (passed to reward funcs).
        tokenizer: Optional tokenizer for pre-tokenizing ground_truth.
        ground_truth_key: Key for ground truth (passed to reward funcs).
        tokenizer: Optional tokenizer for pre-tokenizing ground_truth.
        tokenize_labels: Whether to include ground_truth_ids in output.
            Requires tokenizer to be set. Default: True.
    """

    prompt_key: str = "prompt"
    query_key: str = "query"
    ground_truth_key: str = "ground_truth"

    # Alternative parameter names for flexibility
    prompt: str | None = None
    query: str | None = None
    ground_truth: str | None = None

    # Tokenization settings
    tokenizer: PreTrainedTokenizer | None = field(default=None, repr=False)
    tokenize_labels: bool = True

    def __post_init__(self):
        self.prompt_key = self.prompt or self.prompt_key
        self.query_key = self.query or self.query_key
        self.ground_truth_key = self.ground_truth or self.ground_truth_key

    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        """Convert an example to GRPO format.

        Args:
            example: Input example with prompt and optional query/ground_truth.

        Returns:
            Example with 'prompt', 'ground_truth', and optionally
            'ground_truth_ids' fields for GRPO training.
        """
        prompt = example.get(self.prompt_key, "")
        query = example.get(self.query_key, "")

        # Combine query and prompt if both exist
        if query and prompt:
            combined_prompt = query + "\n" + prompt
        elif query:
            combined_prompt = query
        else:
            combined_prompt = prompt

        ground_truth = example.get(self.ground_truth_key, "")

        result = {"prompt": combined_prompt, "ground_truth": ground_truth}

        # Pre-tokenize ground_truth if tokenizer is provided and flag is enabled
        if self.tokenize_labels and self.tokenizer is not None:
            result["ground_truth_ids"] = self.tokenizer.encode(
                ground_truth, add_special_tokens=False
            )

        return result
