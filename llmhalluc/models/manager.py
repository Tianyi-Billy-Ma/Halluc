"""Model and tokenizer loading utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .patcher import patch_model, patch_tokenizer

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


def get_model(
    model_name_or_path: str,
    args=None,
    tokenizer: PreTrainedTokenizer | None = None,
    **kwargs,
):
    """Load and patch a model.

    Args:
        model_name_or_path: Model identifier or path
        args: Optional arguments object with special token configuration
        tokenizer: Optional tokenizer (needed for embedding resize with special tokens)
        **kwargs: Additional arguments passed to from_pretrained

    Returns:
        Loaded and patched model
    """
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
    model = patch_model(model, tokenizer=tokenizer, args=args)
    return model


def get_tokenizer(tokenizer_name_or_path: str, args=None, **kwargs):
    """Load and patch a tokenizer.

    Args:
        tokenizer_name_or_path: Tokenizer identifier or path
        args: Optional arguments object with special token configuration
        **kwargs: Additional arguments passed to from_pretrained

    Returns:
        Loaded and patched tokenizer
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, **kwargs)
    tokenizer = patch_tokenizer(tokenizer, args=args)
    return tokenizer
