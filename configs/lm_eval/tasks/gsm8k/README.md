# GSM8K Evaluation Configuration

This directory contains the GSM8K (Grade School Math 8K) evaluation configuration with custom metrics.

## Files

- **`gsm8k_simple.yaml`**: Main evaluation configuration
- **`utils.py`**: Custom utility functions for metric computation

## Evaluation Metrics

The `gsm8k_simple` task computes **6 metrics**:

### Filtered Metrics (Exact Match)

1. **`exact_match,strict-match`**: Exact match using strict GSM8K format
   - Extracts answer using pattern: `#### <number>`
   - Normalizes by removing commas, dollar signs, trailing periods
   - Case-insensitive comparison

2. **`exact_match,flexible-extract`**: Exact match using flexible extraction
   - Extracts the last number found in the model output
   - More lenient, catches answers in various formats
   - Same normalization as strict-match

### Unfiltered Metrics (Generation Quality)

3. **`bleu`**: BLEU score on full model output vs ground truth
   - Measures n-gram overlap
   - Computed on entire response (not just the answer)

4. **`rouge1`**: ROUGE-1 score on full output
   - Unigram (single word) overlap
   - Computed on entire response

5. **`rouge2`**: ROUGE-2 score on full output
   - Bigram (two-word sequence) overlap
   - Computed on entire response

6. **`rougeL`**: ROUGE-L score on full output
   - Longest common subsequence
   - Computed on entire response

## Implementation Details

### Custom `process_results` Function

The evaluation uses a custom `process_results` function in `utils.py` that:

1. **Receives unfiltered output**: Gets the full model response
2. **Applies filters manually**: Extracts answers using two strategies
3. **Computes exact match**: On filtered outputs
4. **Computes BLEU/ROUGE**: On unfiltered outputs
5. **Returns all metrics**: In a single dictionary

This approach is necessary because the default lm-evaluation-harness framework applies filters to ALL metrics. To compute some metrics on filtered outputs and others on unfiltered outputs, we need a custom function.

### Filter Strategies

**Strict Match (`#### <number>`):**
```python
# Looks for GSM8K format
"Step by step... #### 42" → "42"
```

**Flexible Extract (last number):**
```python
# Finds last number in text
"The answer is 42." → "42"
"Total: $1,234" → "1234"
```

### Normalization

Both filters normalize answers by:
- Removing commas: `1,234` → `1234`
- Removing dollar signs: `$100` → `100`
- Removing trailing periods: `42.` → `42`
- Converting to lowercase for comparison

## Usage

```bash
lm-eval --model hf \
    --model_args pretrained=meta-llama/Llama-2-7b-hf \
    --tasks gsm8k_simple \
    --include_path /users/tma2/Projects/Halluc/configs/lm_eval/tasks \
    --output_path ./results/
```

## Expected Output

```
Results:
  exact_match,strict-match: 0.75
  exact_match,flexible-extract: 0.78
  bleu: 12.34
  rouge1: 45.67
  rouge2: 23.45
  rougeL: 38.90
```

**Interpretation:**
- `exact_match,strict-match`: 75% of answers match when using strict format
- `exact_match,flexible-extract`: 78% match with flexible extraction (catches more)
- `bleu`: BLEU score of 12.34 for full response quality
- `rouge1/2/L`: ROUGE scores measuring content overlap

## Comparison with Standard Approach

### Standard Filter-Based Approach
```yaml
# Applies filters to ALL metrics
filter_list:
  - name: "strict-match"
    filter: [...]
metric_list:
  - metric: exact_match  # Applied to filtered output
  - metric: bleu         # Also applied to filtered output ❌
```

### Our Custom Approach
```yaml
# Custom function handles filtering selectively
process_results: !function utils.process_results
metric_list:
  - metric: exact_match,strict-match    # Filtered ✓
  - metric: exact_match,flexible-extract # Filtered ✓
  - metric: bleu                         # Unfiltered ✓
  - metric: rouge1                       # Unfiltered ✓
```

## Dependencies

The custom functions require:
- `sacrebleu`: For BLEU computation
- `rouge-score`: For ROUGE computation

These are standard dependencies in lm-evaluation-harness.

## References

- GSM8K Paper: https://arxiv.org/abs/2110.14168
- lm-evaluation-harness: https://github.com/EleutherAI/lm-evaluation-harness
- BLEU: https://aclanthology.org/P02-1040/
- ROUGE: https://aclanthology.org/W04-1013/
