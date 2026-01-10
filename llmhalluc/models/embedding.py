"""Embedding layer utilities for special token initialization.

This module provides deterministic embedding initialization for special tokens.
Unlike the original approach that relies on num_new_tokens and assumes tokens
are at the end of the embedding matrix, this implementation:
1. Gets specific token IDs from the tokenizer
2. Initializes those specific embeddings directly
3. Works correctly for models with pre-existing reserved tokens (e.g., LLaMA3)
"""

import logging
import math
from contextlib import nullcontext
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)


def _initialize_token_embedding(
    embed_weight: torch.Tensor,
    token_id: int,
    base_embedding: torch.Tensor,
    add_noise: bool = True,
) -> None:
    """Initialize a single token embedding.

    Args:
        embed_weight: The embedding weight matrix
        token_id: The token ID to initialize
        base_embedding: The base embedding to use
        add_noise: Whether to add Gaussian noise
    """
    embedding_dim = embed_weight.size(1)

    if add_noise:
        noise = torch.randn_like(base_embedding) * (1.0 / math.sqrt(embedding_dim))
        embed_weight[token_id] = base_embedding + noise
    else:
        embed_weight[token_id] = base_embedding


def _get_description_embedding(
    description: str,
    tokenizer: "PreTrainedTokenizer",
    model: "PreTrainedModel",
    exclude_token_ids: set[int] | None = None,
) -> torch.Tensor:
    """Get embedding for a token based on its description.

    Args:
        description: Text description of the token
        tokenizer: The tokenizer instance
        model: The model instance
        exclude_token_ids: Token IDs to exclude (the special tokens being initialized)

    Returns:
        Average embedding of the description tokens
    """
    exclude_token_ids = exclude_token_ids or set()

    # Tokenize description
    tokens = tokenizer(description, return_tensors="pt", add_special_tokens=False)
    token_ids = tokens["input_ids"][0]

    # Get device from model embeddings
    embed_weight = model.get_input_embeddings().weight
    device = embed_weight.device
    token_ids = token_ids.to(device)

    # Filter out tokens we're initializing (they don't have valid embeddings yet)
    valid_mask = torch.tensor(
        [tid.item() not in exclude_token_ids for tid in token_ids],
        device=device,
    )
    valid_token_ids = token_ids[valid_mask]

    if len(valid_token_ids) == 0:
        # Fallback: use mean of all embeddings
        logger.warning(
            f"Description '{description[:50]}...' contains no valid tokens. "
            "Using mean of all embeddings."
        )
        return embed_weight.mean(dim=0)

    # Get embeddings and average
    with torch.no_grad():
        token_embeds = model.get_input_embeddings()(valid_token_ids)
        return token_embeds.mean(dim=0)


def initialize_special_token_embeddings(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizer",
    special_tokens_config: dict[str, str],
    add_noise: bool = True,
) -> None:
    """Initialize embeddings for specific special tokens based on descriptions.

    This is a deterministic function that initializes specific token IDs,
    not relying on position assumptions.

    Args:
        model: The model to modify
        tokenizer: The tokenizer (used to get token IDs)
        special_tokens_config: Dict mapping token strings to descriptions
                               e.g., {"<|BACKTRACK|>": "Token to delete previous token"}
        add_noise: Whether to add Gaussian noise to initialized embeddings
    """
    if not special_tokens_config:
        return

    # Get token IDs for all special tokens we need to initialize
    token_id_map = {}
    for token_str in special_tokens_config.keys():
        token_id = tokenizer.convert_tokens_to_ids(token_str)
        if token_id == tokenizer.unk_token_id:
            logger.warning(
                f"Token '{token_str}' not found in tokenizer vocabulary, skipping."
            )
            continue
        token_id_map[token_str] = token_id

    if not token_id_map:
        logger.warning(
            "No valid tokens found in special_tokens_config, skipping initialization."
        )
        return

    # Set of token IDs to exclude when computing description embeddings
    exclude_ids = set(token_id_map.values())

    logger.info(
        f"Initializing {len(token_id_map)} special token embeddings: {list(token_id_map.keys())}"
    )

    # Initialize each token's embedding
    with torch.no_grad():
        input_embed_weight = model.get_input_embeddings().weight.data
        output_embed_weight = None
        if (
            model.get_output_embeddings() is not None
            and not model.config.tie_word_embeddings
        ):
            output_embed_weight = model.get_output_embeddings().weight.data

        for token_str, token_id in token_id_map.items():
            description = special_tokens_config[token_str]

            # Get base embedding from description
            base_embedding = _get_description_embedding(
                description=description,
                tokenizer=tokenizer,
                model=model,
                exclude_token_ids=exclude_ids,
            )

            # Initialize input embedding
            _initialize_token_embedding(
                embed_weight=input_embed_weight,
                token_id=token_id,
                base_embedding=base_embedding,
                add_noise=add_noise,
            )

            # Initialize output embedding if not tied
            if output_embed_weight is not None:
                _initialize_token_embedding(
                    embed_weight=output_embed_weight,
                    token_id=token_id,
                    base_embedding=base_embedding,
                    add_noise=add_noise,
                )

            logger.info(f"Initialized embedding for '{token_str}' (id={token_id})")


def resize_embedding_layer(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizer",
    new_special_tokens_config: dict[str, str] | None = None,
) -> None:
    """Resize token embeddings if needed and initialize special tokens.

    This function:
    1. Resizes model embeddings if tokenizer vocab > model vocab
    2. Initializes specific special token embeddings based on descriptions

    Works correctly for models like LLaMA where reserved tokens already exist
    in the vocabulary - no resize needed but embeddings still get initialized.

    Args:
        model: The model to resize/modify
        tokenizer: The tokenizer (determines target vocab size)
        new_special_tokens_config: Dict mapping special tokens to descriptions
    """
    try:
        from transformers.integrations import is_deepspeed_zero3_enabled

        if is_deepspeed_zero3_enabled():
            import deepspeed

            params = [model.get_input_embeddings().weight]
            if (
                model.get_output_embeddings() is not None
                and not model.config.tie_word_embeddings
            ):
                params.append(model.get_output_embeddings().weight)

            context_maybe_zero3 = deepspeed.zero.GatheredParameters(
                params, modifier_rank=0
            )
        else:
            context_maybe_zero3 = nullcontext()
    except ImportError:
        context_maybe_zero3 = nullcontext()

    with context_maybe_zero3:
        current_embedding_size = model.get_input_embeddings().weight.size(0)

    # Resize embeddings if tokenizer has more tokens than model
    if len(tokenizer) > current_embedding_size:
        if getattr(model, "quantization_method", None):
            raise ValueError("Cannot resize embedding layers of a quantized model.")

        if not isinstance(model.get_output_embeddings(), torch.nn.Linear):
            raise ValueError(
                "Current model does not support resizing embedding layers."
            )

        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)

        with context_maybe_zero3:
            new_embedding_size = model.get_input_embeddings().weight.size(0)

        logger.info(
            f"Resized embeddings: {current_embedding_size} -> {new_embedding_size}"
        )
        model.config.vocab_size = new_embedding_size

    # Initialize special token embeddings (works for both new and existing tokens)
    # This runs regardless of whether resize happened - critical for LLaMA reserved tokens
    if new_special_tokens_config:
        with context_maybe_zero3:
            initialize_special_token_embeddings(
                model=model,
                tokenizer=tokenizer,
                special_tokens_config=new_special_tokens_config,
            )
