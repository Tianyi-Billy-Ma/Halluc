from lm_eval.api.registry import register_aggregation, register_metric
from rouge_score import rouge_scorer


def _compute_rouge(items, rouge_type):
    """Compute ROUGE score for a list of (reference, prediction) pairs."""
    scorer = rouge_scorer.RougeScorer([rouge_type], use_stemmer=True)
    refs, preds = zip(*items)
    scores = [
        scorer.score(str(r), str(p))[rouge_type].fmeasure for r, p in zip(refs, preds)
    ]
    return sum(scores) / len(scores)


@register_aggregation("rouge1")
def rouge1_agg(items):
    return _compute_rouge(items, "rouge1")


@register_aggregation("rouge2")
def rouge2_agg(items):
    return _compute_rouge(items, "rouge2")


@register_aggregation("rougeL")
def rougeL_agg(items):
    return _compute_rouge(items, "rougeL")


@register_metric(
    metric="rouge1",
    higher_is_better=True,
    output_type="generate_until",
    aggregation="rouge1",
)
def rouge1_fn(items):
    return items


@register_metric(
    metric="rouge2",
    higher_is_better=True,
    output_type="generate_until",
    aggregation="rouge2",
)
def rouge2_fn(items):
    return items


@register_metric(
    metric="rougeL",
    higher_is_better=True,
    output_type="generate_until",
    aggregation="rougeL",
)
def rougeL_fn(items):
    return items
