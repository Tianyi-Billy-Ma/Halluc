from lm_eval.api.filter import Filter
from lm_eval.api.registry import register_filter


# >>>>>>>>
@register_filter("straight")
class StraightFilter(Filter):
    def __init__(self) -> None:
        pass

    def apply(self, resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
        return resps


# <<<<<<<<
