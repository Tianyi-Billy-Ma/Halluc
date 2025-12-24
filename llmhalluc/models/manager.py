from transformers import AutoModelForCausalLM, AutoTokenizer

from .patcher import patch_model, patch_tokenizer


def get_model(model_name_or_path: str, **kwargs):
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)

    model = patch_model(model)


def get_tokenizer(tokenizer_name_or_path: str, **kwargs):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, **kwargs)

    tokenizer = patch_tokenizer(tokenizer)

    return tokenizer
