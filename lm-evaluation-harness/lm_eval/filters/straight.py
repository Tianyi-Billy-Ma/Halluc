from lm_eval.api.filter import Filter
from lm_eval.api.registry import register_filter


# >>>>>>>>
@register_filter("straight")
class StraightFilter(Filter):
    def __init__(self, fallback: str = "[invalid]") -> None:
        pass

    def apply(self, resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
        def filter_set(inst):
            filtered = []
            for resp in inst:
                if resp:
                    match = resp.strip()
                else:
                    match = self.fallback
                filtered.append(match)
            return filtered

        filtered_resps = list(map(lambda x: filter_set(x), resps))
        return filtered_resps


# <<<<<<<<
