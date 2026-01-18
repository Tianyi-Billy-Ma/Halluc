"""Backtrack dataset converter."""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from transformers import PreTrainedTokenizer

from ..prompts.MathPrompt import MATH_INSTRUCTION
from ..utils.alg_utils import cs_alg
from .backtrack import BACKTRACK_TOKEN
from .base import DatasetConverter

logger = logging.getLogger(__name__)


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

    prompt_key: str = "prompt"
    query_key: str = "question"
    response_key: str = "answer"

    prompt: str = ""
    query: str = ""
    response: str = ""

    def __post_init__(self) -> None:
        self.prompt_key = self.prompt or self.prompt_key
        self.query_key = self.query or self.query_key
        self.response_key = self.response or self.response_key

    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        """Convert an example to backtrack format.

        Args:
            example: Input example with prompt, query, and response.

        Returns:
            Converted example with backtrack content and modified response.
        """
        # Extract fields using key mapping
        prompt_content = example.get(self.prompt_key, "")
        query_content = example.get(self.query_key, "")
        response_content = example.get(self.response_key, "")

        prompt_content = query_content + "\n" + prompt_content

        return {
            "prompt": prompt_content,
            "query": "",
            "response": response_content,
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
    option: str = "half"  # ["all", "half", "random", "single"]

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

    def __call__(
        self, example: dict[str, Any] | list[dict[str, Any]]
    ) -> dict[str, Any]:
        instance_obj = list(example.values())[0]
        batch_size = len(instance_obj)
        if isinstance(instance_obj, list):
            return self._process_examples(example, batch_size=batch_size)
        else:
            return self._process_example(example)

    def _process_examples(
        self, examples: dict[str, Any], batch_size: int
    ) -> list[dict[str, Any]]:
        example_list = self.batch_to_list(examples)
        return self.list_to_batch(
            [self._process_example(example) for example in example_list]
        )

    def _preprocess_example(
        self, example: dict[str, Any]
    ) -> tuple[GSM8KBacktrackAttr, list[int], list[int]]:
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

        sym_token_ids = self.tokenizer.encode(response)
        ori_token_ids = self.tokenizer.encode(ori_response)
        return data_attr, ori_token_ids, sym_token_ids

    def _find_lcs(
        self, ori_token_ids: list[int], sym_token_ids: list[int]
    ) -> tuple[list[int], list[int]]:
        sym_lcs_pairs, ori_lcs_pairs = cs_alg(sym_token_ids, ori_token_ids)

        # Edge case: If no matching segments found, handle gracefully
        if not sym_lcs_pairs:
            sym_lcs_pairs = [
                (0, 0),
                (len(sym_token_ids), len(sym_token_ids)),
            ]
            ori_lcs_pairs = [
                (0, 0),
                (len(ori_token_ids), len(ori_token_ids)),
            ]
        else:
            # Ensure the matching segments cover the full sequence
            if sym_lcs_pairs[0][0] != 0:
                sym_lcs_pairs.insert(0, (0, 0))
                ori_lcs_pairs.insert(0, (0, 0))
            if sym_lcs_pairs[-1][1] != len(sym_token_ids):
                sym_lcs_pairs.append((len(sym_token_ids), len(sym_token_ids)))
                ori_lcs_pairs.append((len(ori_token_ids), len(ori_token_ids)))

        return ori_lcs_pairs, sym_lcs_pairs

    def _modify(
        self,
        ori_lcs_pairs: list[int],
        sym_lcs_pairs: list[int],
        ori_token_ids: list[int],
        sym_token_ids: list[int],
        chosen_candidates: list[bool],
    ):
        modified_token_ids, backtrack_idx = [], None
        for idx, chosen in enumerate(chosen_candidates):
            sym_pair, ori_pair = sym_lcs_pairs[idx], ori_lcs_pairs[idx]
            sym_start, sym_end = sym_pair
            ori_start, ori_end = ori_pair

            sym_next_start = sym_lcs_pairs[idx + 1][0]
            ori_next_start = ori_lcs_pairs[idx + 1][0]

            assert sym_token_ids[sym_start:sym_end] == ori_token_ids[ori_start:ori_end]

            modified_sub_ids = []
            if chosen:
                modified_sub_ids = ori_token_ids[ori_start:ori_next_start]
                # Fix: Use 'is None' instead of 'not' to handle backtrack_idx=0 case
                if backtrack_idx is None:
                    backtrack_idx = len(modified_token_ids) + len(modified_sub_ids)
                num_backtrack_tokens = ori_next_start - ori_end
                modified_sub_ids.extend(
                    [self.backtrack_token_id] * num_backtrack_tokens
                )
                modified_sub_ids.extend(sym_token_ids[sym_end:sym_next_start])
            else:
                modified_sub_ids = sym_token_ids[sym_start:sym_next_start]
            modified_token_ids.extend(modified_sub_ids)

        # Add remaining tokens after all matching segments
        sym_final_start = sym_lcs_pairs[-1][0]
        modified_token_ids.extend(sym_token_ids[sym_final_start:])

        return modified_token_ids, backtrack_idx

    def _verify(
        self,
        response_token_ids: list[int] | None = None,
        modified_token_ids: list[int] | None = None,
    ):
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

        assert verify_token_ids == response_token_ids, (
            f"The modified token ids are not correct. "
            f"Expected length {len(response_token_ids)}, got {len(verify_token_ids)}"
        )

    def _choose_candidates(
        self, num_candidates: int, option: str = "half"
    ) -> list[bool]:
        option = option or self.option
        chosen_candidates = [False] * num_candidates
        if option == "all":
            chosen_candidates = [True] * num_candidates
        elif option == "single":
            chosen_idx = np.random.randint(0, num_candidates)
            chosen_candidates[chosen_idx] = True

        elif option in ["half", "random"]:
            p = 0.5 if option == "half" else np.random.uniform(0.4, 1)
            chosen_candidates = list(
                np.random.choice([True, False], size=num_candidates, p=[p, 1 - p])
            )
        else:
            raise ValueError(f"Invalid option: {option}")

        # Edge case: Ensure at least one divergence is chosen for backtracking
        if not any(chosen_candidates):
            # If no divergence chosen, pick one randomly
            chosen_idx = np.random.randint(0, num_candidates)
            chosen_candidates[chosen_idx] = True

        return chosen_candidates

    def _process_example(self, example: dict[str, Any]) -> dict[str, Any]:
        data_attr, ori_response_token_ids, sym_response_token_ids = (
            self._preprocess_example(example)
        )
        # Edge case: If responses are identical, return without backtracking
        if sym_response_token_ids == ori_response_token_ids:
            data_attr.backtrack_suffix = data_attr.response
            data_attr.backtrack_response = data_attr.response
            return data_attr.to_dict()

        ori_lcs_pairs, sym_lcs_pairs = self._find_lcs(
            ori_response_token_ids, sym_response_token_ids
        )

        num_candidates = len(sym_lcs_pairs) - 1

        if num_candidates == 0:
            data_attr.backtrack_suffix = data_attr.response
            data_attr.backtrack_response = data_attr.response
            return data_attr.to_dict()

        chosen_candidates = self._choose_candidates(num_candidates, self.option)

        modified_token_ids, backtrack_idx = self._modify(
            ori_lcs_pairs,
            sym_lcs_pairs,
            ori_response_token_ids,
            sym_response_token_ids,
            chosen_candidates,
        )

        self._verify(
            response_token_ids=sym_response_token_ids,
            modified_token_ids=modified_token_ids,
        )

        backtrack_response = self.tokenizer.decode(modified_token_ids)
        backtrack_prefix = self.tokenizer.decode(modified_token_ids[:backtrack_idx])
        backtrack_suffix = self.tokenizer.decode(modified_token_ids[backtrack_idx:])

        data_attr.backtrack_response = backtrack_response
        data_attr.backtrack_suffix = backtrack_suffix
        data_attr.backtrack_prefix = backtrack_prefix
        return data_attr.to_dict()


@dataclass
class GSM8KBacktrackDatasetConverter(GSM8KSymbolicDatasetConverter):
    prop: float = 0.5
    _num_prop: float | None = None

    def __call__(
        self, example: dict[str, Any] | list[dict[str, Any]]
    ) -> dict[str, Any]:
        if self.option == "single":
            logger.warning(
                "The converter does not support batch processing with multiple examples. Please use convert 'GSM8KSymbolicDatasetConverter' for batch processing."
            )
            self.option = "single"

        return self._process_examples(example)

    def _process_example(self, example: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError("This method is not implemented for this converter.")

    def _process_examples(
        self, examples: dict[str, Any], batch_size: int | None = None
    ) -> dict[str, Any]:
        batch_size = batch_size or len(next(iter(examples.values())))
        if batch_size > 1:
            raise ValueError(
                "The converter does not support batch processing with multiple examples. Please use convert 'GSM8KSymbolicDatasetConverter' for batch processing."
            )

        example = {k: v[0] for k, v in examples.items()}

        data_attr, ori_response_token_ids, sym_response_token_ids = (
            self._preprocess_example(example)
        )

        # Edge case: If responses are identical, return without backtracking
        if sym_response_token_ids == ori_response_token_ids:
            data_attr.backtrack_suffix = data_attr.response
            data_attr.backtrack_response = data_attr.response
            return self.list_to_batch([data_attr.to_dict()])

        ori_lcs_pairs, sym_lcs_pairs = self._find_lcs(
            ori_response_token_ids, sym_response_token_ids
        )

        num_candidates = len(sym_lcs_pairs) - 1

        if num_candidates == 0:
            data_attr.backtrack_suffix = data_attr.response
            data_attr.backtrack_response = data_attr.response
            return self.list_to_batch([data_attr.to_dict()])

        results = []
        self.option = "single"

        nums = max(int(num_candidates * batch_size), 1)

        for _ in range(nums):
            chosen_candidates = self._choose_candidates(num_candidates, self.option)

            modified_token_ids, backtrack_idx = self._modify(
                ori_lcs_pairs,
                sym_lcs_pairs,
                ori_response_token_ids,
                sym_response_token_ids,
                chosen_candidates,
            )

            self._verify(
                response_token_ids=sym_response_token_ids,
                modified_token_ids=modified_token_ids,
            )

            backtrack_response = self.tokenizer.decode(modified_token_ids)
            backtrack_prefix = self.tokenizer.decode(modified_token_ids[:backtrack_idx])
            backtrack_suffix = self.tokenizer.decode(modified_token_ids[backtrack_idx:])

            data_attr.backtrack_response = backtrack_response
            data_attr.backtrack_suffix = backtrack_suffix
            data_attr.backtrack_prefix = backtrack_prefix
            results.append(data_attr.to_dict())
        return self.list_to_batch(results)
