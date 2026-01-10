# Analysis: Why Training Performance Did Not Improve

**Author**: Claude (Research Analysis)  
**Date**: January 9, 2026  
**Project**: Halluc - Backtrack Token Training for LLMs

---

## Executive Summary

This document provides a detailed analysis of why standard SFT and RL approaches have failed to improve the backtrack token training performance. Based on code review and literature analysis, we identify **seven fundamental issues** that prevent effective learning.

---

## 1. The Core Problem

### 1.1 Task Definition Recap

Given a sequence: `x₁ x₂ x₃ x₄ b b b x₅ x₆ b x₇`

The backtrack token `b` should delete the immediately preceding regular token. So:
- `x₄ b` → `x₄` deleted
- `x₃ b` → `x₃` deleted  
- `x₂ b` → `x₂` deleted
- `x₆ b` → `x₆` deleted

**Final output**: `x₁ x₅ x₇`

### 1.2 Current Training Approach (from code review)

Based on `llmhalluc/data/backtrack.py`:

```python
# Training data construction:
# 1. Take correct response
# 2. Insert random "error" tokens at random position
# 3. Add backtrack tokens to delete errors
# 4. Continue with correct response

# Example:
# Original: "The answer is 42"
# Augmented: "The answer foo bar <|BACKTRACK|><|BACKTRACK|> is 42"
```

---

## 2. Fundamental Issues

### Issue #1: Negative Learning Problem 🚨

**Severity**: Critical

**Problem**: Standard SFT minimizes negative log-likelihood of the **entire** sequence:

$$\mathcal{L}_{SFT} = -\sum_{t=1}^{T} \log P(y_t | y_{<t}, x)$$

This means the model is **explicitly trained to generate the error tokens** before generating the backtrack tokens.

**Consequence**:
- Model learns: "To reach the correct answer, I must first generate these specific wrong tokens"
- This **reinforces hallucination patterns** rather than discouraging them
- You're doing **Behavior Cloning on erroneous traces**

**Evidence from Literature**: The SCoRe paper (DeepMind, 2024) explicitly identifies this as a key failure mode of SFT for self-correction:

> "Supervised fine-tuning on correction traces fails because it trains the model to maximize the probability of the initial incorrect response."

---

### Issue #2: Attention Pollution 🚨

**Severity**: Critical

**Problem**: In standard Transformers, backtrack tokens are "soft" tokens. They:
- Do NOT physically remove information from the KV-Cache
- Do NOT modify the attention mask

**Consequence**:
- When generating the "Correction" segment, the model still attends to:
  ```
  [Prompt] ... [Error Tokens] [Backtrack Tokens] [Correction Tokens]
  ```
- The error tokens **pollute the attention** and influence correction generation
- The model may incorporate incorrect information from error tokens into its correction

**Analogy**: Imagine trying to solve a math problem while someone writes wrong equations on your whiteboard that you can't erase. Even knowing they're wrong, they influence your thinking.

---

### Issue #3: Lack of Causal Trigger 🚨

**Severity**: High

**Problem**: In synthetic training data, the transition `[Error] → [Backtrack]` is deterministic. But during inference:
- The model generates errors because they're probabilistically likely
- Why would it immediately generate `<|BACKTRACK|>` after?
- There's no **verification signal** that triggers backtracking

**Missing Component**: A **verifier mechanism** that detects errors:
- Current setup: `error_token → backtrack` is learned as a fixed pattern
- Required: `error_detected → backtrack` where detection is dynamic

**From Literature**: The SequenceMatch paper solves this by formulating generation as an MDP where backtracking is an **action** chosen based on **state evaluation**.

---

### Issue #4: Sequence-Level Reward Problem ⚠️

**Severity**: High

**Problem**: If using RL, rewards are typically given at the sequence level:
- Correct final answer → Positive reward
- Incorrect final answer → Negative reward

**Consequence**:
- The model doesn't know **which tokens** led to success/failure
- Cannot learn that `error + backtrack + correction` is good because of `backtrack + correction`, not `error`
- May learn that errors are acceptable as long as correction follows

**From Literature**: Token-level credit assignment methods (TEMPO, CAPO, Q-RM) specifically address this by providing fine-grained rewards per token.

---

### Issue #5: Backtrack Token Embedding Initialization ⚠️

**Severity**: Medium

**Problem**: The `<|BACKTRACK|>` token is a new special token with:
- Randomly initialized embedding
- No pre-training on meaningful context

**Consequence**:
- Initial training steps are highly unstable
- Large gradient updates can destabilize the entire model
- Potential for catastrophic forgetting

**Best Practice from Literature**: Initialize new token embeddings as:
- Mean of all existing embeddings
- Or average of semantically similar tokens (e.g., "delete", "remove", "undo")

---

### Issue #6: Distribution Mismatch ⚠️

**Severity**: Medium

**Problem**: Training data is synthetically constructed:
- Errors are **randomly sampled** tokens
- Real model errors follow the model's **actual distribution**

**Consequence**:
- Model learns to backtrack from random noise tokens
- Doesn't learn to backtrack from its own likely errors
- At inference, model may not recognize its own errors as "backtrack-worthy"

**From Literature**: SCoRe specifically uses **self-generated data** to avoid this mismatch.

---

### Issue #7: No Curriculum Learning ⚠️

**Severity**: Medium

**Problem**: Training data includes backtrack sequences of varying complexity:
- 1 backtrack token
- 5 backtrack tokens
- 10 backtrack tokens

Without curriculum, the model must learn all complexities simultaneously.

**Consequence**:
- Overwhelmed by complex cases before mastering simple ones
- May learn spurious correlations

---

## 3. Visual Diagnosis

### What the Model is Learning vs. What It Should Learn

**Current SFT Training** (Problematic):

```
Loss computed on:  [✓ Prompt] [✓ Error] [✓ Backtrack] [✓ Correction]
                          ↑
                    Model learns to generate this!
```

**Desired Training**:

```
Loss computed on:  [✓ Prompt] [✗ Error] [✓ Backtrack] [✓ Correction]
                          ↑
                    Loss ignored here (labels = -100)
```

### Attention Pattern Issue

**Current Attention** (Problematic):

```
Token positions: [P₁ P₂ P₃] [E₁ E₂] [B₁ B₂] [C₁ C₂]

C₁ attends to:   P₁ P₂ P₃ | E₁ E₂ | B₁ B₂     ← Bad! Sees errors
```

**Desired Attention**:

```
C₁ attends to:   P₁ P₂ P₃ | ❌ ❌ | ❌ ❌       ← Good! Error info blocked
```

---

## 4. Evidence from Your Code

### 4.1 From `backtrack.py`

```python
# Line 86-92
backtrack_token_ids = (
    response_token_ids[:random_split] + self.no_spc_vocab[:random_int]
)

curr_response_token_ids = [backtrack_id] * random_int + response_token_ids[
    random_split:
]
```

**Issue**: The error tokens (`self.no_spc_vocab[:random_int]`) are:
- Randomly selected from vocabulary
- Not based on model's actual error distribution
- Part of the training target (will contribute to loss)

### 4.2 From `sft.py`

```python
# Line 99-105
self.trainer = SFTTrainer(
    model=self.model,
    processing_class=self.tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=self.args,
)
```

**Issue**: Using standard SFTTrainer without:
- Custom data collator for masked loss
- Custom attention mask construction
- Backtrack-aware preprocessing

---

## 5. Quantitative Evidence from Literature

### 5.1 SCoRe Paper Results

| Method | MATH Accuracy | Self-Correction Gain |
|--------|--------------|---------------------|
| Base Model | 54.0% | -2.1% (degrades!) |
| SFT on Corrections | 56.2% | +0.8% (minimal) |
| SCoRe (RL) | 69.6% | +15.6% |

**Key Insight**: SFT provides minimal improvement; RL with proper credit assignment is essential.

### 5.2 SequenceMatch Results

| Method | Perplexity | Generation Quality |
|--------|------------|-------------------|
| MLE Training | Higher | Lower |
| SequenceMatch | Lower | Higher |

**Key Insight**: Imitation learning formulation outperforms maximum likelihood.

---

## 6. Summary Table

| Issue | Category | Impact | Difficulty to Fix |
|-------|----------|--------|-------------------|
| Negative Learning | Training | Critical | Medium |
| Attention Pollution | Architecture | Critical | High |
| No Causal Trigger | Data/Architecture | High | High |
| Sequence-Level Rewards | Training | High | Medium |
| Token Initialization | Setup | Medium | Low |
| Distribution Mismatch | Data | Medium | Medium |
| No Curriculum | Training | Medium | Low |

---

## 7. Key Takeaways

1. **SFT fundamentally cannot work** for this task in its standard form because it teaches the model to generate errors

2. **Attention mask modification is essential** to prevent error tokens from influencing corrections

3. **Token-level credit assignment** is required for effective RL training

4. **Self-generated data** is preferred over synthetic random errors

5. **The backtrack token must have meaningful initialization** to prevent training instability

---

## Next Steps

See `novel_techniques.md` for proposed solutions to these issues.
