---
license: mit
task_categories:
- text-generation
- question-answering
language:
- en
tags:
- math
- gsm8k
- backtracking
- error-correction
- llm-training
size_categories:
- 1K<n<10K
---

# GSM8K-Random-All

A dataset for training LLMs with **random backtracking** capabilities. This dataset augments the original [GSM8K](https://huggingface.co/datasets/openai/gsm8k) math word problems with synthetic error injection and backtrack recovery sequences.

## Overview

This dataset teaches models to:
1. Make "mistakes" (random error tokens)
2. Recognize the mistake
3. Use `<|BACKTRACK|>` tokens to "delete" the errors
4. Continue with the correct solution

### Backtracking Mechanism

The `<|BACKTRACK|>` token functionally acts as a backspace. When a model generates this token, the previous token is conceptually deleted. This enables self-correction during generation.

**Example:**
```
Original:  "The answer is 42"
Modified:  "The answer XX<|BACKTRACK|><|BACKTRACK|>is 42"
```

When processed, the two `<|BACKTRACK|>` tokens delete the two `XX` error tokens, recovering the original text.

## Available Subsets

| Subset | `backtrack_ratio` | `backtrack_num_errors` | Description |
|--------|-------------------|------------------------|-------------|
| `p1_n1` | 1 | 1 | 1 random position, 1 error token |
| `p1_n3` | 1 | 3 | 1 random position, 3 error tokens |
| `p0.1_n10` | 0.1 | 10 | 10% of positions, 10 error tokens each |

### Subset Naming Convention

Format: `p{ratio}_n{num_errors}`

- **p_ratio**: Number of positions to inject errors
  - Integer ≥ 1: Exact number of positions (e.g., `p1` = 1 position, `p3` = 3 positions)
  - Float < 1: Fraction of response tokens (e.g., `p0.1` = 10% of tokens)
- **n_num_errors**: Number of error tokens inserted at each position

## Dataset Structure

Each example contains:

| Column | Description |
|--------|-------------|
| `query` | Original math word problem question |
| `response` | Original correct answer/solution |
| `backtrack_response` | Modified response with error tokens and backtracks |
| `backtrack_prefix` | Everything before the first `<|BACKTRACK|>` token |
| `backtrack_suffix` | Everything from the first `<|BACKTRACK|>` token onward |

**Invariant:** `backtrack_response = backtrack_prefix + backtrack_suffix`

## Usage

```python
from datasets import load_dataset

# Load a specific subset
dataset = load_dataset("mtybilly/GSM8K-Random-All", "p1_n1")

# Access training data
train_data = dataset["train"]
print(train_data[0])
```

## Technical Details

### Tokenizer

All processing uses the **Llama 3** tokenizer (`meta-llama/Llama-3.2-1B`).

The `<|BACKTRACK|>` token is added as a special token and always encodes to exactly one token ID.

### Error Injection Algorithm

1. **Position Sampling**: 
   - If `backtrack_ratio` is an integer ≥ 1: Sample exactly that many positions
   - If `backtrack_ratio` is a float < 1: Sample `floor(num_tokens * ratio)` positions
   - Positions are sampled without replacement, excluding position 0

2. **Error Injection**:
   At each sampled position:
   ```
   [original tokens before position]
   + [random_error_tokens × num_errors]
   + [<|BACKTRACK|> × num_errors]
   + [original token at position]
   + [remaining original tokens]
   ```

3. **Verification**:
   Each example is verified by simulating backtrack execution to ensure the original response is recoverable.

### Random Seed

All subsets are generated with `seed=42` for reproducibility.

## Source Dataset

Based on [mtybilly/GSM8K](https://huggingface.co/datasets/mtybilly/GSM8K), a preprocessed version of [OpenAI GSM8K](https://huggingface.co/datasets/openai/gsm8k) with:
- Calculator annotations removed (`<<...>>`)
- Answer marker changed from `####` to `The answer is:`
- Train split divided into train/eval (90/10)

**Split sizes:**
- Train: 6,725 examples
- Eval: 748 examples  
- Test: 1,319 examples

## License

MIT License

## Related

- [GSM8K](https://huggingface.co/datasets/openai/gsm8k) - Original dataset
- [GSM8K-Backtrack-all](https://huggingface.co/datasets/mtybilly/GSM8K-Backtrack-all) - Symbolic backtrack variant
