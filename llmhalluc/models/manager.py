"""Model and tokenizer loading utilities."""

from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

from .patcher import patch_model, patch_tokenizer


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
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, **kwargs)
    except Exception:
        # Fallback: sometimes tokenizer loading fails if config is missing, try trusting remote code or ignoring mismatch
        # Or if it's a directory issue
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path, trust_remote_code=True, **kwargs
        )

    tokenizer = patch_tokenizer(tokenizer, args=args)
    return tokenizer
