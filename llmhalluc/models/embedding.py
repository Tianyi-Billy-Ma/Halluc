"""Embedding layer utilities for special token initialization.

This module provides deterministic embedding initialization for special tokens.
Unlike the original approach that relies on num_new_tokens and assumes tokens
are at the end of the embedding matrix, this implementation:
1. Gets specific token IDs from the tokenizer
2. Initializes those specific embeddings directly
3. Works correctly for models with pre-existing reserved tokens (e.g., LLaMA3)

Supports multiple initialization strategies:
- "description": Average embeddings of tokens in a text description
- "semantic": Average embeddings of semantically similar words
- "combined": Weighted combination of description and semantic approaches
- "mean": Mean of all existing embeddings (fallback)
"""

import logging
import math
from contextlib import nullcontext
from enum import Enum
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)


class InitStrategy(str, Enum):
    """Initialization strategy for special token embeddings."""

    DESCRIPTION = "description"
    SEMANTIC = "semantic"
    COMBINED = "combined"
    MEAN = "mean"


# Default semantic words for backtrack-like tokens
DEFAULT_SEMANTIC_WORDS = [
    "delete",
    "remove",
    "undo",
    "erase",
    "back",
    "cancel",
    "retry",
    "revert",
    "reset",
    "clear",
    "backspace",
]


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


def _get_semantic_embedding(
    semantic_words: list[str],
    tokenizer: "PreTrainedTokenizer",
    model: "PreTrainedModel",
    exclude_token_ids: set[int] | None = None,
) -> torch.Tensor:
    """Get embedding by averaging embeddings of semantically similar words.

    This approach initializes the token embedding by averaging the embeddings
    of words that are semantically related to the token's intended meaning.
    For example, for a backtrack token, we might use ["delete", "remove", "undo"].

    Args:
        semantic_words: List of semantically similar words
        tokenizer: The tokenizer instance
        model: The model instance
        exclude_token_ids: Token IDs to exclude (the special tokens being initialized)

    Returns:
        Average embedding of valid semantic word tokens
    """
    exclude_token_ids = exclude_token_ids or set()

    embed_weight = model.get_input_embeddings().weight
    device = embed_weight.device

    valid_token_ids = []

    for word in semantic_words:
        # Try the word as a single token first
        token_id = tokenizer.convert_tokens_to_ids(word)
        if token_id != tokenizer.unk_token_id and token_id not in exclude_token_ids:
            valid_token_ids.append(token_id)
        else:
            # If not a single token, encode and get all subword tokens
            subword_ids = tokenizer.encode(word, add_special_tokens=False)
            for sid in subword_ids:
                if sid != tokenizer.unk_token_id and sid not in exclude_token_ids:
                    valid_token_ids.append(sid)

    # Remove duplicates while preserving order
    seen = set()
    unique_token_ids = []
    for tid in valid_token_ids:
        if tid not in seen:
            seen.add(tid)
            unique_token_ids.append(tid)

    if not unique_token_ids:
        logger.warning(
            f"No valid tokens found for semantic words {semantic_words[:5]}... "
            "Using mean of all embeddings."
        )
        return embed_weight.mean(dim=0)

    # Get embeddings and average
    token_ids_tensor = torch.tensor(unique_token_ids, device=device)
    with torch.no_grad():
        token_embeds = model.get_input_embeddings()(token_ids_tensor)
        logger.debug(
            f"Semantic embedding from {len(unique_token_ids)} tokens: "
            f"{[tokenizer.convert_ids_to_tokens(tid) for tid in unique_token_ids[:10]]}"
        )
        return token_embeds.mean(dim=0)


def _get_combined_embedding(
    description: str | None,
    semantic_words: list[str] | None,
    tokenizer: "PreTrainedTokenizer",
    model: "PreTrainedModel",
    exclude_token_ids: set[int] | None = None,
    description_weight: float = 0.5,
) -> torch.Tensor:
    """Get embedding by combining description and semantic similarity approaches.

    This combines both methods with configurable weighting, providing a more
    robust initialization that captures both the functional description and
    semantic similarity to related concepts.

    Args:
        description: Text description of the token (optional)
        semantic_words: List of semantically similar words (optional)
        tokenizer: The tokenizer instance
        model: The model instance
        exclude_token_ids: Token IDs to exclude
        description_weight: Weight for description embedding (0.0 to 1.0)
                           semantic_weight = 1.0 - description_weight

    Returns:
        Weighted average of description and semantic embeddings
    """
    embed_weight = model.get_input_embeddings().weight
    embeddings_to_combine = []
    weights = []

    # Get description embedding if available
    if description:
        desc_embedding = _get_description_embedding(
            description=description,
            tokenizer=tokenizer,
            model=model,
            exclude_token_ids=exclude_token_ids,
        )
        embeddings_to_combine.append(desc_embedding)
        weights.append(description_weight)

    # Get semantic embedding if available
    if semantic_words:
        semantic_embedding = _get_semantic_embedding(
            semantic_words=semantic_words,
            tokenizer=tokenizer,
            model=model,
            exclude_token_ids=exclude_token_ids,
        )
        embeddings_to_combine.append(semantic_embedding)
        weights.append(1.0 - description_weight)

    if not embeddings_to_combine:
        logger.warning(
            "No description or semantic words provided. Using mean of all embeddings."
        )
        return embed_weight.mean(dim=0)

    if len(embeddings_to_combine) == 1:
        return embeddings_to_combine[0]

    # Normalize weights
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]

    # Weighted combination
    combined = torch.zeros_like(embeddings_to_combine[0])
    for emb, weight in zip(embeddings_to_combine, weights):
        combined += weight * emb

    return combined


def initialize_special_token_embeddings(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizer",
    special_tokens_config: dict,
    add_noise: bool = True,
    default_strategy: str = "combined",
) -> None:
    """Initialize embeddings for specific special tokens.

    Supports multiple initialization strategies:
    - "description": Average embeddings of tokens in a text description
    - "semantic": Average embeddings of semantically similar words
    - "combined": Weighted combination of description and semantic approaches
    - "mean": Mean of all existing embeddings (fallback)

    Args:
        model: The model to modify
        tokenizer: The tokenizer (used to get token IDs)
        special_tokens_config: Dict mapping token strings to config dict.
            Config dict keys:
                - "description": Text description of the token
                - "semantic_words": List of semantically similar words
                - "strategy": One of "description", "semantic", "combined", "mean"
                - "description_weight": Weight for description in combined mode (0-1)
            Example:
                {
                    "<|BACKTRACK|>": {
                        "description": "Token to delete the previous token",
                        "semantic_words": ["delete", "remove", "undo", "erase"],
                        "strategy": "combined",
                        "description_weight": 0.5
                    }
                }
        add_noise: Whether to add Gaussian noise to initialized embeddings
        default_strategy: Default strategy when not specified in config
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

    # Set of token IDs to exclude when computing embeddings
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
            config = special_tokens_config[token_str]

            description: str | None = config.get("description")
            semantic_words: list[str] = config.get(
                "semantic_words", DEFAULT_SEMANTIC_WORDS
            )
            strategy: str = config.get("strategy", default_strategy)
            description_weight: float = config.get("description_weight", 0.5)

            # Get base embedding based on strategy
            if strategy == InitStrategy.DESCRIPTION or strategy == "description":
                if not description:
                    logger.warning(
                        f"No description for '{token_str}' with description strategy. "
                        "Falling back to mean."
                    )
                    base_embedding = model.get_input_embeddings().weight.mean(dim=0)
                else:
                    base_embedding = _get_description_embedding(
                        description=description,
                        tokenizer=tokenizer,
                        model=model,
                        exclude_token_ids=exclude_ids,
                    )
                    logger.info(
                        f"Using description strategy for '{token_str}': "
                        f"'{description[:50]}...'"
                    )

            elif strategy == InitStrategy.SEMANTIC or strategy == "semantic":
                base_embedding = _get_semantic_embedding(
                    semantic_words=semantic_words,
                    tokenizer=tokenizer,
                    model=model,
                    exclude_token_ids=exclude_ids,
                )
                logger.info(
                    f"Using semantic strategy for '{token_str}': "
                    f"{semantic_words[:5]}..."
                )

            elif strategy == InitStrategy.COMBINED or strategy == "combined":
                base_embedding = _get_combined_embedding(
                    description=description,
                    semantic_words=semantic_words,
                    tokenizer=tokenizer,
                    model=model,
                    exclude_token_ids=exclude_ids,
                    description_weight=description_weight,
                )
                logger.info(
                    f"Using combined strategy for '{token_str}': "
                    f"desc_weight={description_weight}"
                )

            elif strategy == InitStrategy.MEAN or strategy == "mean":
                base_embedding = model.get_input_embeddings().weight.mean(dim=0)
                logger.info(f"Using mean strategy for '{token_str}'")

            else:
                logger.warning(
                    f"Unknown strategy '{strategy}' for '{token_str}'. "
                    "Falling back to combined."
                )
                base_embedding = _get_combined_embedding(
                    description=description,
                    semantic_words=semantic_words,
                    tokenizer=tokenizer,
                    model=model,
                    exclude_token_ids=exclude_ids,
                    description_weight=description_weight,
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

            logger.info(
                f"Initialized embedding for '{token_str}' (id={token_id}) "
                f"using {strategy} strategy"
            )


def resize_embedding_layer(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizer",
    new_special_tokens_config: dict | None = None,
) -> None:
    """Resize token embeddings if needed and initialize special tokens.

    This function:
    1. Resizes model embeddings if tokenizer vocab > model vocab
    2. Initializes specific special token embeddings based on config

    Works correctly for models like LLaMA where reserved tokens already exist
    in the vocabulary - no resize needed but embeddings still get initialized.

    Args:
        model: The model to resize/modify
        tokenizer: The tokenizer (determines target vocab size)
        new_special_tokens_config: Dict mapping special tokens to config.
            Can be str (description) or dict with strategy options.
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
