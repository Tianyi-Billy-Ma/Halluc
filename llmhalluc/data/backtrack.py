"""Backtrack dataset converter."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from .base import DatasetConverter

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

BACKTRACK_TOKEN = "<|BACKTRACK|>"

logger = logging.getLogger(__name__)


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


@dataclass
class RandomBacktrackConverter(DatasetConverter):
    """Converter that injects random errors with backtrack recovery.

    This converter samples random positions in a response and inserts:
    1. Random error tokens at each position
    2. Backtrack tokens to "delete" the errors
    3. The original correct tokens

    The key difference from symbolic backtrack is that errors are randomly
    generated rather than derived from similar problems.

    Args:
        tokenizer: Tokenizer for encoding/decoding text (default: Llama-3).
        backtrack_ratio: Number of positions to sample.
            - int >= 1: Exact number of positions (e.g., 1, 3, 5)
            - float < 1: Fraction of response tokens (e.g., 0.1, 0.2)
        backtrack_num_errors: Number of error tokens to insert at each position.
        backtrack_token: Token used for backtracking (default: <|BACKTRACK|>).
        seed: Random seed for reproducibility.

    Example:
        Original response: "The answer is 42"
        With 1 error position, 2 error tokens:
        "The answer XX<|BACKTRACK|><|BACKTRACK|>is 42"

        Where XX are random tokens that get "deleted" by backtracks.

    Subset Naming Convention:
        p{ratio}_n{num_errors}
        - p1_n1: 1 position, 1 error token
        - p1_n3: 1 position, 3 error tokens
        - p0.1_n10: 10% of positions, 10 error tokens each
    """

    tokenizer: PreTrainedTokenizer | None = None
    backtrack_ratio: int | float = 1  # 1 = single position, 0.1 = 10% of tokens
    backtrack_num_errors: int = 1
    backtrack_token: str = BACKTRACK_TOKEN
    seed: int | None = None

    # Key mappings for input columns
    query_key: str = "question"
    response_key: str = "answer"

    # Internal state (not part of __init__)
    _rng: np.random.Generator | None = field(default=None, repr=False, init=False)
    _backtrack_token_id: int | None = field(default=None, repr=False, init=False)
    _non_special_token_ids: list[int] | None = field(
        default=None, repr=False, init=False
    )

    def __post_init__(self) -> None:
        """Initialize random generator and token IDs."""
        self._rng = np.random.default_rng(self.seed)

        if self.tokenizer is not None:
            # Get backtrack token ID
            bt_ids = self.tokenizer.encode(
                self.backtrack_token, add_special_tokens=False
            )
            if len(bt_ids) != 1:
                raise ValueError(
                    f"Backtrack token '{self.backtrack_token}' must encode to exactly "
                    f"one token, got {len(bt_ids)}: {bt_ids}"
                )
            self._backtrack_token_id = bt_ids[0]

            # Get non-special token IDs for random error generation
            self._non_special_token_ids = [
                i
                for i in range(self.tokenizer.vocab_size)
                if i not in self.tokenizer.all_special_ids
            ]
            logger.info(
                f"Initialized RandomBacktrackConverter: "
                f"ratio={self.backtrack_ratio}, num_errors={self.backtrack_num_errors}, "
                f"vocab_size={len(self._non_special_token_ids)}"
            )

    def _compute_num_positions(self, num_tokens: int) -> int:
        """Compute number of positions to sample based on ratio.

        Args:
            num_tokens: Total number of tokens in response.

        Returns:
            Number of positions to sample (at least 1).
        """
        if isinstance(self.backtrack_ratio, float) and self.backtrack_ratio < 1:
            # Ratio mode: fraction of tokens
            num_positions = max(1, int(num_tokens * self.backtrack_ratio))
        else:
            # Exact count mode
            num_positions = int(self.backtrack_ratio)

        # Ensure we don't sample more positions than available
        return min(num_positions, max(1, num_tokens - 1))

    def _sample_positions(self, num_tokens: int, num_positions: int) -> list[int]:
        """Sample random positions for error injection.

        Args:
            num_tokens: Total number of tokens.
            num_positions: Number of positions to sample.

        Returns:
            Sorted list of positions (0-indexed).
        """
        if num_tokens <= 1:
            return []

        # Sample without replacement, avoiding position 0 (keep first token)
        available = list(range(1, num_tokens))
        num_to_sample = min(num_positions, len(available))
        positions = self._rng.choice(available, size=num_to_sample, replace=False)
        return sorted(positions.tolist())

    def _generate_error_tokens(self, num_errors: int) -> list[int]:
        """Generate random error token IDs.

        Args:
            num_errors: Number of error tokens to generate.

        Returns:
            List of random token IDs.
        """
        return self._rng.choice(
            self._non_special_token_ids, size=num_errors, replace=True
        ).tolist()

    def _inject_backtracks(
        self, token_ids: list[int], positions: list[int], num_errors: int
    ) -> tuple[list[int], int]:
        """Inject error tokens and backtrack recovery at specified positions.

        At each position, we insert:
        - `num_errors` random error tokens
        - `num_errors` backtrack tokens
        - Then continue with the original token

        Args:
            token_ids: Original response token IDs.
            positions: Sorted positions to inject errors.
            num_errors: Number of error tokens per position.

        Returns:
            Tuple of (modified_token_ids, first_backtrack_index).
        """
        if not positions:
            return token_ids.copy(), -1

        modified = []
        first_bt_idx = -1
        pos_set = set(positions)

        for i, token_id in enumerate(token_ids):
            if i in pos_set:
                # Record first backtrack position
                if first_bt_idx == -1:
                    first_bt_idx = len(modified) + num_errors  # After error tokens

                # Insert error tokens
                error_tokens = self._generate_error_tokens(num_errors)
                modified.extend(error_tokens)

                # Insert backtrack tokens
                modified.extend([self._backtrack_token_id] * num_errors)

            # Add original token
            modified.append(token_id)

        return modified, first_bt_idx

    def _verify_backtrack(
        self, original_ids: list[int], modified_ids: list[int]
    ) -> bool:
        """Verify that applying backtracks recovers the original.

        Args:
            original_ids: Original token IDs.
            modified_ids: Modified token IDs with backtracks.

        Returns:
            True if verification passes.
        """
        recovered = []
        pending_bt = 0

        for token_id in modified_ids:
            if token_id == self._backtrack_token_id:
                pending_bt += 1
            else:
                if pending_bt > 0:
                    recovered = (
                        recovered[:-pending_bt] if pending_bt <= len(recovered) else []
                    )
                    pending_bt = 0
                recovered.append(token_id)

        return recovered == original_ids

    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        """Convert an example by injecting random backtracks.

        Args:
            example: Input example with query and response.

        Returns:
            Converted example with columns:
            - query: Original question
            - response: Original answer
            - backtrack_response: Answer with injected errors and backtracks
            - backtrack_prefix: Everything before first backtrack token
            - backtrack_suffix: Everything from first backtrack token onward
        """
        if self.tokenizer is None:
            raise ValueError(
                "RandomBacktrackConverter requires a tokenizer. "
                "Pass tokenizer=... when instantiating."
            )

        query = example.get(self.query_key, "")
        response = example.get(self.response_key, "")

        # Tokenize response
        response_ids = self.tokenizer.encode(response, add_special_tokens=False)
        num_tokens = len(response_ids)

        # Compute positions
        num_positions = self._compute_num_positions(num_tokens)
        positions = self._sample_positions(num_tokens, num_positions)

        # Inject backtracks
        modified_ids, first_bt_idx = self._inject_backtracks(
            response_ids, positions, self.backtrack_num_errors
        )

        # Verify correctness
        if not self._verify_backtrack(response_ids, modified_ids):
            logger.warning(
                f"Backtrack verification failed for example. "
                f"Original: {len(response_ids)} tokens, Modified: {len(modified_ids)} tokens"
            )

        # Decode results
        backtrack_response = self.tokenizer.decode(modified_ids)

        if first_bt_idx == -1:
            # No backtracks inserted
            backtrack_prefix = backtrack_response
            backtrack_suffix = ""
        else:
            backtrack_prefix = self.tokenizer.decode(modified_ids[:first_bt_idx])
            backtrack_suffix = self.tokenizer.decode(modified_ids[first_bt_idx:])

        return {
            "query": query,
            "response": response,
            "backtrack_response": backtrack_response,
            "backtrack_prefix": backtrack_prefix,
            "backtrack_suffix": backtrack_suffix,
        }
