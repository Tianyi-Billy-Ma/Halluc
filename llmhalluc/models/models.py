import logging

from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM

from llmhalluc.extras.constant import SPECIAL_TOKEN_MAPPING

logger = logging.getLogger(__name__)

@register_model("hf-bt")
class HFLMBT(HFLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._patch_tokenizer()

    def tok_decode(
        self, tokens: list[int] | int, skip_special_tokens: bool = True
    ) -> str:
        if isinstance(tokens, int):
            tokens = [tokens]
        processed_tokens = []
        for token in tokens:
            if token == self.backtrack_token_id and len(processed_tokens) > 0:
                processed_tokens = processed_tokens[:-1]
            else:
                processed_tokens.append(token)
        return super().tok_decode(processed_tokens, skip_special_tokens)

    def _patch_tokenizer(self):

        model_name = self.model.name_or_path

        for key, value in SPECIAL_TOKEN_MAPPING.items():
            if key in model_name:
                backtrack_token = list(value.keys())[0]
                break
        # self.tokenizer.add_special_tokens(
        #     {"additional_special_tokens": [backtrack_token]},
        #     replace_additional_special_tokens=False,
        # )
        self.backtrack_token_id = self.tokenizer.encode(backtrack_token)[0]
        logger.info(
            f"Patched tokenizer with backtrack token '{backtrack_token}' (id: {self.backtrack_token_id})"
        )


# <<<<<<<<
