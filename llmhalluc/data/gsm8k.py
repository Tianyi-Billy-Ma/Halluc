"""Backtrack dataset converter."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from transformers import PreTrainedTokenizer
from .backtrack import BACKTRACK_TOKEN
from .base import DatasetConverter
from ..prompts.MathPrompt import MATH_INSTRUCTION
from ..utils.alg_utils import cs_alg


@dataclass
class GSM8KBacktrackAttr:
    prompt: str
    query: str
    response: str
    backtrack_response: str
    backtrack_prefix: str
    backtrack_suffix: str
    original_query: str
    original_response: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "query": self.query,
            "response": self.response,
            "backtrack_response": self.backtrack_response,
            "backtrack_prefix": self.backtrack_prefix,
            "backtrack_suffix": self.backtrack_suffix,
            "original_query": self.original_query,
            "original_response": self.original_response,
        }


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
            self.key_mapping = {
                "prompt": "prompt",
                "query": "question",
                "response": "answer",
            }
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


@dataclass
class GSM8KSymbolicDatasetConverter(DatasetConverter):
    tokenizer: PreTrainedTokenizer
    max_tokens: int = 100
    split: str = "train"
    key_mapping: dict[str, str] | None = None
    prompt: str = MATH_INSTRUCTION
    backtrack_token: str = BACKTRACK_TOKEN
    backtrack_token_id: int = None
    option: str = "half"  # ["all", "half", "random"]

    def __post_init__(self) -> None:
        """Initialize default values after dataclass creation."""
        if self.key_mapping is None:
            self.key_mapping = {
                "prompt": "prompt",
                "query": "question",
                "response": "answer",
                "original_query": "original_question",
                "original_response": "original_answer",
            }

        if self.tokenizer is not None:
            backtrack_token_ids = self.tokenizer.encode(self.backtrack_token)
            if len(backtrack_token_ids) != 1:
                raise ValueError(
                    f"Backtrack token '{self.backtrack_token}' must encode to exactly one token, "
                    f"but got {len(backtrack_token_ids)} tokens: {backtrack_token_ids}"
                )
            self.backtrack_token_id = backtrack_token_ids[0]
            
        print(f"Option set to: {self.option}")

    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        query = example[self.key_mapping["query"]]
        response = example[self.key_mapping["response"]].replace("\n\n", "\n")
        original_query = example[self.key_mapping["original_query"]]
        ori_response = example[self.key_mapping["original_response"]].replace(
            "\n\n", "\n"
        )

        data_attr = GSM8KBacktrackAttr(
            prompt=self.prompt,
            query=query,
            response=response,
            backtrack_response="",
            backtrack_prefix="",
            backtrack_suffix="",
            original_query=original_query,
            original_response=ori_response,
        )

        sym_response_token_ids = self.tokenizer.encode(response)
        ori_response_token_ids = self.tokenizer.encode(ori_response)

        # Edge case: If responses are identical, return without backtracking
        if sym_response_token_ids == ori_response_token_ids:
            data_attr.backtrack_suffix = response
            data_attr.backtrack_response = response
            return data_attr.to_dict()

        sym_lcs_pairs, ori_lcs_pairs = cs_alg(
            sym_response_token_ids, ori_response_token_ids
        )

        # Edge case: If no matching segments found, handle gracefully
        if not sym_lcs_pairs:
            sym_lcs_pairs = [
                (0, 0),
                (len(sym_response_token_ids), len(sym_response_token_ids)),
            ]
            ori_lcs_pairs = [
                (0, 0),
                (len(ori_response_token_ids), len(ori_response_token_ids)),
            ]
        else:
            # Ensure the matching segments cover the full sequence
            if sym_lcs_pairs[0][0] != 0:
                sym_lcs_pairs.insert(0, (0, 0))
                ori_lcs_pairs.insert(0, (0, 0))
            if sym_lcs_pairs[-1][1] != len(sym_response_token_ids):
                sym_lcs_pairs.append(
                    (len(sym_response_token_ids), len(sym_response_token_ids))
                )
                ori_lcs_pairs.append(
                    (len(ori_response_token_ids), len(ori_response_token_ids))
                )

        num_candidates = len(sym_lcs_pairs) - 1

        # Edge case: If no divergence points exist, return without backtracking
        if num_candidates == 0:
            data_attr.backtrack_suffix = response
            data_attr.backtrack_response = response
            return data_attr.to_dict()

        # Randomly choose at most one divergence point to follow the original path
        #
        if self.option == "all":
            chosen_candidates = [True] * num_candidates
        else:
            p = 0.5 if self.option == "half" else np.random.uniform(0.4, 1)
            chosen_candidates = list(np.random.choice([True, False], size=num_candidates, p=[p, 1 - p]))

        # Edge case: Ensure at least one divergence is chosen for backtracking
        if not any(chosen_candidates):
            # If no divergence chosen, pick one randomly
            chosen_idx = np.random.randint(0, num_candidates)
            chosen_candidates[chosen_idx] = True

        modified_token_ids, backtrack_idx = [], None
        for idx, chosen in enumerate(chosen_candidates):
            sym_pair, ori_pair = sym_lcs_pairs[idx], ori_lcs_pairs[idx]
            sym_start, sym_end = sym_pair
            ori_start, ori_end = ori_pair

            sym_next_start = sym_lcs_pairs[idx + 1][0]
            ori_next_start = ori_lcs_pairs[idx + 1][0]

            assert (
                sym_response_token_ids[sym_start:sym_end]
                == ori_response_token_ids[ori_start:ori_end]
            )

            modified_sub_ids = []
            if chosen:
                modified_sub_ids = ori_response_token_ids[ori_start:ori_next_start]
                # Fix: Use 'is None' instead of 'not' to handle backtrack_idx=0 case
                if backtrack_idx is None:
                    backtrack_idx = len(modified_token_ids) + len(modified_sub_ids)
                num_backtrack_tokens = ori_next_start - ori_end
                modified_sub_ids.extend(
                    [self.backtrack_token_id] * num_backtrack_tokens
                )
                modified_sub_ids.extend(sym_response_token_ids[sym_end:sym_next_start])
            else:
                modified_sub_ids = sym_response_token_ids[sym_start:sym_next_start]
            modified_token_ids.extend(modified_sub_ids)

        # Add remaining tokens after all matching segments
        sym_final_start = sym_lcs_pairs[-1][0]
        modified_token_ids.extend(sym_response_token_ids[sym_final_start:])

        # Verify the modified response by simulating backtrack execution
        verify_token_ids, backtrack_count = [], 0
        for token_id in modified_token_ids:
            if token_id == self.backtrack_token_id:
                backtrack_count += 1
            else:
                verify_token_ids = (
                    verify_token_ids[:-backtrack_count]
                    if backtrack_count > 0
                    else verify_token_ids
                )
                verify_token_ids.append(token_id)
                backtrack_count = 0

        verify_response = self.tokenizer.decode(verify_token_ids)
        backtrack_response = self.tokenizer.decode(modified_token_ids)
        backtrack_prefix = self.tokenizer.decode(modified_token_ids[:backtrack_idx])
        backtrack_suffix = self.tokenizer.decode(modified_token_ids[backtrack_idx:])

        assert verify_token_ids == sym_response_token_ids, (
            f"The modified token ids are not correct. "
            f"Expected length {len(sym_response_token_ids)}, got {len(verify_token_ids)}"
        )
        assert verify_response == response, (
            f"The modified response is not correct.\n"
            f"Expected: {response}\n"
            f"Got: {verify_response}"
        )

        data_attr.backtrack_response = backtrack_response
        data_attr.backtrack_suffix = backtrack_suffix
        data_attr.backtrack_prefix = backtrack_prefix
        return data_attr.to_dict()
