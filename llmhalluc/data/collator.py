"""Data collators for backtrack training."""

from dataclasses import dataclass, field
from typing import Any

import torch
from transformers import PreTrainedTokenizer
from transformers.data.data_collator import DataCollatorMixin


@dataclass
class BacktrackMaskedCollator(DataCollatorMixin):
    """
    Data collator that masks error tokens from loss computation.

    For backtrack training, we want the model to learn:
    - WHEN to emit <|BACKTRACK|> tokens (after seeing errors)
    - WHAT the correct tokens are (after backtracking)
    - NOT to generate error tokens themselves

    This collator detects backtrack tokens in the input and masks the
    preceding tokens (the "errors") from the loss computation by setting
    their labels to -100.

    Logic:
        For each run of N consecutive backtrack tokens, mask the N tokens
        immediately preceding that run.

    Example:
        input_ids: [A, B, ERR1, ERR2, BT, BT, CORRECT, ...]
        labels:    [A, B, -100, -100, BT, BT, CORRECT, ...]
                         ^^^^^ ^^^^^ masked (won't contribute to loss)

    Args:
        tokenizer: The tokenizer used for encoding.
        backtrack_token_id: Token ID of the backtrack token. If None,
            will be auto-detected from tokenizer using backtrack_token.
        backtrack_token: String representation of backtrack token.
            Used to get token ID if backtrack_token_id is None.
        pad_to_multiple_of: Pad sequences to a multiple of this value.
        return_tensors: Type of tensors to return ("pt" for PyTorch).
    """

    tokenizer: PreTrainedTokenizer
    backtrack_token_id: int = field(default=-1, init=True)
    backtrack_token: str = "<|BACKTRACK|>"
    pad_to_multiple_of: int | None = None
    return_tensors: str = "pt"
    reset_position_ids: bool = False

    def __post_init__(self):
        """Initialize backtrack_token_id if not provided."""
        if self.backtrack_token_id == -1:
            # Try to get token ID from tokenizer
            token_id = self.tokenizer.convert_tokens_to_ids(self.backtrack_token)

            if isinstance(token_id, list):
                if len(token_id) == 1:
                    token_id = token_id[0]
                else:
                    raise ValueError(
                        f"Backtrack token '{self.backtrack_token}' maps to "
                        f"multiple token IDs: {token_id}"
                    )
            if token_id == self.tokenizer.unk_token_id:
                # Token not in vocabulary, try encoding
                encoded = self.tokenizer.encode(
                    self.backtrack_token, add_special_tokens=False
                )
                if len(encoded) == 1:
                    token_id = encoded[0]
                else:
                    raise ValueError(
                        f"Backtrack token '{self.backtrack_token}' encodes to "
                        f"{len(encoded)} tokens, expected 1. "
                        f"Ensure the token is properly added to the tokenizer."
                    )
            self.backtrack_token_id = int(token_id)

    def _compute_error_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute mask indicating which tokens are errors (to be masked from loss).

        For each run of N consecutive backtrack tokens, the N tokens immediately
        preceding that run are marked as errors.

        Args:
            input_ids: 1D tensor of token IDs for a single sequence.

        Returns:
            Boolean tensor where True indicates error tokens (to be masked).
        """
        seq_len = len(input_ids)
        error_mask = torch.zeros(seq_len, dtype=torch.bool)

        # Find all backtrack token positions
        is_backtrack = input_ids == self.backtrack_token_id

        if not is_backtrack.any():
            return error_mask

        # Process the sequence to find backtrack runs and mask preceding tokens
        i = 0
        while i < seq_len:
            if is_backtrack[i]:
                # Count consecutive backtrack tokens
                bt_count = 0
                bt_start = i
                while i < seq_len and is_backtrack[i]:
                    bt_count += 1
                    i += 1

                # Mask the bt_count tokens preceding this backtrack run
                # (but don't go below index 0)
                mask_start = max(0, bt_start - bt_count)
                mask_end = bt_start
                error_mask[mask_start:mask_end] = True
            else:
                i += 1

        return error_mask

    def _compute_position_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute position IDs with backtrack rewinding logic.

        Logic:
            x x x x x b b x x => 0 1 2 3 4 5 6 3 4
            x x x b x b => 0 1 2 3 2 3

            - Normal tokens (x): Use stack_depth (semantic position).
            - Backtrack tokens (b): Use last_pos + 1 (continue physical sequence).
            - Backtrack tokens decrement stack_depth (rewind semantic position).
        """
        positions = []
        stack_depth = 0
        last_pos = -1

        for token_id in input_ids:
            if token_id == self.backtrack_token_id:
                # Backtrack token: continues strictly from previous position
                current = last_pos + 1
                stack_depth = max(0, stack_depth - 1)
            else:
                # Normal token: takes logical position (stack depth)
                current = stack_depth
                stack_depth += 1

            positions.append(current)
            last_pos = current

        return torch.tensor(positions, dtype=torch.long)

    def torch_call(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """
        Collate examples and apply error masking to labels.

        Args:
            examples: List of examples, each containing 'input_ids' and
                optionally 'attention_mask'.

        Returns:
            Batch dict with 'input_ids', 'attention_mask', and 'labels'.
        """
        # Extract input_ids from examples
        input_ids_list = [
            torch.tensor(ex["input_ids"])
            if not isinstance(ex["input_ids"], torch.Tensor)
            else ex["input_ids"]
            for ex in examples
        ]

        # Get or create attention masks
        if "attention_mask" in examples[0]:
            attention_mask_list = [
                torch.tensor(ex["attention_mask"])
                if not isinstance(ex["attention_mask"], torch.Tensor)
                else ex["attention_mask"]
                for ex in examples
            ]
        else:
            attention_mask_list = [torch.ones_like(ids) for ids in input_ids_list]

        # Compute error masks for each sequence
        error_masks = [self._compute_error_mask(ids) for ids in input_ids_list]

        # Compute position IDs if requested
        position_ids_list = None
        if self.reset_position_ids:
            position_ids_list = [
                self._compute_position_ids(ids) for ids in input_ids_list
            ]

        # Create labels (clone of input_ids, will mask errors)
        labels_list = [ids.clone() for ids in input_ids_list]

        # Apply error masking to labels
        for labels, error_mask in zip(labels_list, error_masks):
            labels[error_mask] = -100

        # Pad sequences
        batch_size = len(input_ids_list)
        max_len = max(len(ids) for ids in input_ids_list)

        # Apply pad_to_multiple_of if specified
        if self.pad_to_multiple_of is not None and max_len % self.pad_to_multiple_of:
            max_len = (
                (max_len // self.pad_to_multiple_of) + 1
            ) * self.pad_to_multiple_of

        # Initialize padded tensors
        pad_val = 0
        if self.tokenizer.pad_token_id is not None:
            pad_val = int(self.tokenizer.pad_token_id)

        padded_input_ids = torch.full((batch_size, max_len), pad_val, dtype=torch.long)
        padded_attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        padded_labels = torch.full((batch_size, max_len), -100, dtype=torch.long)

        padded_position_ids = None
        if position_ids_list is not None:
            padded_position_ids = torch.zeros((batch_size, max_len), dtype=torch.long)

        # Get padding side from tokenizer (default to right if not set)
        padding_side = getattr(self.tokenizer, "padding_side", "right")

        # Fill in values based on padding side
        # Create iterator that handles optional position_ids
        iter_data = zip(input_ids_list, attention_mask_list, labels_list)
        if position_ids_list is not None:
            iter_data = zip(
                input_ids_list, attention_mask_list, labels_list, position_ids_list
            )

        for i, items in enumerate(iter_data):
            if position_ids_list is not None:
                ids, mask, labels, pos_ids = items
            else:
                ids, mask, labels = items
                pos_ids = None

            seq_len = len(ids)
            if padding_side == "left":
                # Left padding: sequence is right-aligned
                start_idx = max_len - seq_len
                padded_input_ids[i, start_idx:] = ids
                padded_attention_mask[i, start_idx:] = mask
                padded_labels[i, start_idx:] = labels
                if pos_ids is not None:
                    padded_position_ids[i, start_idx:] = pos_ids
            else:
                # Right padding (default): sequence is left-aligned
                padded_input_ids[i, :seq_len] = ids
                padded_attention_mask[i, :seq_len] = mask
                padded_labels[i, :seq_len] = labels
                if pos_ids is not None:
                    padded_position_ids[i, :seq_len] = pos_ids

        batch = {
            "input_ids": padded_input_ids,
            "attention_mask": padded_attention_mask,
            "labels": padded_labels,
        }

        if padded_position_ids is not None:
            batch["position_ids"] = padded_position_ids

        return batch
