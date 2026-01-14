"""Model and tokenizer patching utilities."""

import logging

from transformers import PreTrainedModel, PreTrainedTokenizer

from llmhalluc.extras.template import DEFAULT_CHAT_TEMPLATE

from .embedding import resize_embedding_layer

logger = logging.getLogger(__name__)


def _get_special_tokens_config(args) -> dict | None:
    """Get special tokens config from args.

    Args:
        args: Arguments object with init_special_tokens and new_special_tokens_config

    Returns:
        Dict mapping special tokens to config (str or dict), or None if disabled

    Raises:
        ValueError: If init_special_tokens is True but new_special_tokens_config is not provided
    """
    if args is None:
        return None

    init_special_tokens = getattr(args, "init_special_tokens", False)
    if not init_special_tokens:
        return None

    config = getattr(args, "new_special_tokens_config", None)
    if not config:
        raise ValueError(
            "init_special_tokens is True but new_special_tokens_config is not provided. "
            "Please provide new_special_tokens_config in your config file."
        )

    return config


def patch_tokenizer(tokenizer: PreTrainedTokenizer, args=None) -> PreTrainedTokenizer:
    """Patch tokenizer with chat template, pad token, and special tokens.

    Args:
        tokenizer: The tokenizer to patch
        args: Optional arguments object with special token configuration

    Returns:
        Patched tokenizer
    """
    # Set default chat template if not present
    tokenizer.chat_template = tokenizer.chat_template or DEFAULT_CHAT_TEMPLATE

    # Set pad token if not present
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    # Add special tokens if configured
    special_tokens_config = _get_special_tokens_config(args)
    if special_tokens_config:
        special_tokens = list(special_tokens_config.keys())
        num_added = tokenizer.add_special_tokens(
            {"additional_special_tokens": special_tokens},
            replace_additional_special_tokens=False,
        )
        if num_added > 0:
            logger.info(f"Added {num_added} special tokens: {special_tokens}")

    return tokenizer


def patch_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer | None = None,
    args=None,
) -> PreTrainedModel:
    """Patch model with resized embeddings for special tokens.

    Args:
        model: The model to patch
        tokenizer: The tokenizer (needed for embedding resize)
        args: Optional arguments object with special token configuration

    Returns:
        Patched model
    """
    if tokenizer is None:
        return model

    # Resize embeddings and initialize special tokens if configured
    special_tokens_config = _get_special_tokens_config(args)
    if special_tokens_config:
        resize_embedding_layer(
            model=model,
            tokenizer=tokenizer,
            new_special_tokens_config=special_tokens_config,
        )

    return model
