from transformers import PreTrainedModel, PreTrainedTokenizer
from llmhalluc.extras.template import DEFAULT_CHAT_TEMPLATE


def patch_model(model: PreTrainedModel):
    # Saved for future use.
    return model


def patch_tokenizer(tokenizer: PreTrainedTokenizer):
    tokenizer.chat_template = tokenizer.chat_template or DEFAULT_CHAT_TEMPLATE

    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    return tokenizer
