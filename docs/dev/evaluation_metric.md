# Evaluation Metric Implementation


## Background 

We need to implement several evaluation metrics into the `lm-evaluation-harness` framework.


## Metrics 

We aim to implement the following metrics: 
- Rouge1, Rouge2, RougeL
- BLUE


## Notes 

We should follow the instruction in [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) to implement the metrics.

The metrics should be implemented in the `llmhalluc/eval/metrics.py` file.
The filters functions should be implemented in the `llmhalluc/eval/filters.py` file.

After we implement these metrics, we should create a new task in the `configs/lm_eval/tasks/` directory.
You can follow the instruction in [task_guide.md](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/new_task_guide.md) to create a new task.
The new task should be named as `gsm8k_comprehensive.yaml`


# Implementation Plan

## Overview

The `lm-evaluation-harness` framework uses a decorator-based registration system for metrics. BLEU is natively supported, but ROUGE metrics (rouge1, rouge2, rougeL) need custom implementation.

## Step 1: Implement Custom Metrics in `llmhalluc/eval/metrics.py`

Register ROUGE metrics using `@register_metric` and `@register_aggregation` decorators from `lm_eval.api.registry`:

```python
from lm_eval.api.registry import register_metric, register_aggregation
from rouge_score import rouge_scorer

# Aggregation functions
@register_aggregation("rouge1")
def rouge1_agg(items):
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    refs, preds = zip(*items)
    scores = [scorer.score(r, p)['rouge1'].fmeasure for r, p in zip(refs, preds)]
    return sum(scores) / len(scores)

@register_aggregation("rouge2")
def rouge2_agg(items):
    # Similar pattern for rouge2

@register_aggregation("rougeL")
def rougeL_agg(items):
    # Similar pattern for rougeL

# Metric registrations (passthrough functions)
@register_metric(metric="rouge1", higher_is_better=True, output_type="generate_until", aggregation="rouge1")
def rouge1_fn(items):
    return items

# Similar for rouge2, rougeL
```

## Step 2: Ensure Metrics Module is Imported

The metrics module must be imported before `lm_eval` runs so decorators execute. Options:
- Import in `llmhalluc/eval/__init__.py`
- Import in `llmhalluc/run_eval.py` before calling `run_eval()`

## Step 3: Create `gsm8k_comprehensive.yaml` Task

Create new task file at `configs/lm_eval/tasks/gsm8k/gsm8k_comprehensive.yaml`:

```yaml
task: gsm8k_comprehensive
include: gsm8k_custom.yaml
metric_list:
  - metric: exact_match
    aggregation: mean
    higher_is_better: true
    ignore_case: true
    regexes_to_ignore:
      - ","
      - "\\$"
      - "(?s).*#### "
      - "\\.$"
  - metric: bleu
    aggregation: bleu
    higher_is_better: true
  - metric: rouge1
    aggregation: rouge1
    higher_is_better: true
  - metric: rouge2
    aggregation: rouge2
    higher_is_better: true
  - metric: rougeL
    aggregation: rougeL
    higher_is_better: true
```

## Step 4: Add Dependencies

Ensure `rouge-score` package is available (add to requirements if needed).

## Files to Modify/Create

| File | Action |
|------|--------|
| `llmhalluc/eval/metrics.py` | Implement ROUGE metrics |
| `llmhalluc/eval/__init__.py` | Import metrics module |
| `configs/lm_eval/tasks/gsm8k/gsm8k_comprehensive.yaml` | Create new task |

## Notes

- BLEU is already natively supported by `lm-evaluation-harness` via `sacrebleu`
- The `filters.py` file is not needed for these metrics as they operate on raw model outputs
- The existing `gsm8k_bt.yaml` already references these metrics but they may fail without the custom implementations