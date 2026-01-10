# Summary: Backtrack Token Training Research

**Author**: Claude (Research Summary)  
**Date**: January 9, 2026  
**Project**: Halluc - Backtrack Token Training for LLMs

---

## Project Goal

Train LLMs to perform **on-the-fly self-correction** using a special **backtrack token** (`b`) that functions as a "backspace" to delete previously generated tokens.

**Example**:
```
Generated: x x x x x b b b b x x b x
           ↓ ↓ ↓ ↓ ↓       ↓ ↓   ↓
Final:     x                x x   x  → "x x x"
```

---

## Problem Statement

Despite using SFT and RL training approaches, **performance has not improved**. This research analyzes why and proposes solutions.

---

## Research Deliverables

Four documents have been created in `docs/research/claude/`:

| Document | Purpose |
|----------|---------|
| `literature_review.md` | Comprehensive review of 20+ relevant papers |
| `analysis_why_not_working.md` | Diagnosis of 7 fundamental training issues |
| `novel_techniques.md` | 10 techniques with implementation code |
| `implementation_guide.md` | Step-by-step priority implementation |
| `summary.md` | This summary document |

---

## Key Findings

### Why Current Training Fails

| Issue | Severity | Description |
|-------|----------|-------------|
| **Negative Learning** | 🔴 Critical | SFT teaches model to generate errors before correcting them |
| **Attention Pollution** | 🔴 Critical | Error tokens remain in context, polluting corrections |
| **No Causal Trigger** | 🟠 High | Model doesn't learn *when* to backtrack, only patterns |
| **Sequence-Level Rewards** | 🟠 High | RL can't identify which tokens caused success/failure |
| **Token Initialization** | 🟡 Medium | Random embedding causes training instability |
| **Distribution Mismatch** | 🟡 Medium | Random errors ≠ model's actual error distribution |
| **No Curriculum** | 🟡 Medium | Complex cases overwhelm simple learning |

### Key Insight

> **Standard SFT fundamentally cannot work** for this task because it explicitly trains the model to generate the error tokens.

---

## Literature Highlights

### Most Relevant Papers

1. **SequenceMatch** (Cundy & Ermon, 2023)
   - Frames backtracking as imitation learning with MDP
   - Shows backspace action can mitigate compounding errors

2. **SCoRe** (DeepMind, 2024)
   - Multi-turn RL for self-correction
   - +15.6% on MATH without external oracles

3. **Process Reward Models** (OmegaPRM, 2024)
   - Step-level supervision for reasoning
   - Gemini Pro: 51% → 69.4% on MATH500

4. **Backtracking for Safety** (2024)
   - RESET token for undoing unsafe generations
   - 4x safety improvement without helpfulness loss

---

## Recommended Techniques

### Priority Order

| Priority | Technique | Effort | Impact |
|----------|-----------|--------|--------|
| 🥇 1 | Masked-Error SFT | Easy | High |
| 🥇 1 | Semantic Token Init | Easy | Medium |
| 🥈 2 | Curriculum Learning | Easy | Medium |
| 🥈 2 | Hard Attention Mask | Medium | High |
| 🥉 3 | Self-Generated Errors | Medium | High |
| 🥉 3 | Token-Level DPO | Medium | High |
| Advanced | Process Reward Model | Hard | Very High |
| Advanced | MCTS Data Generation | Hard | Very High |
| Inference | KV-Cache Rewinding | Medium | High |
| Research | Verifier Head | Hard | Very High |

### Quick Wins (Day 1-2)

1. **Don't train on errors**: Mask error tokens in loss (`labels = -100`)
2. **Initialize properly**: Use semantic embedding for `<|BACKTRACK|>`
3. **Start simple**: Use curriculum (1-2 tokens → 5 → 10)

### Medium-Term (Week 1-2)

4. **Block attention**: Corrections shouldn't see errors
5. **Use real errors**: Generate from model, not random
6. **Preference learning**: DPO between backtrack vs. continue

### Long-Term (Month 1+)

7. **Process Reward Model**: Learn *when* to backtrack
8. **MCTS**: Search for optimal backtrack strategies
9. **Verifier Head**: Built-in error detection

---

## Implementation Starting Points

### Masked-Error SFT (Critical Fix)

```python
# In DataCollator
labels[error_token_indices] = -100  # Don't train on errors!
```

### Semantic Token Initialization

```python
# Initialize with similar word embeddings
similar_words = ["delete", "remove", "undo", "erase"]
similar_ids = [tokenizer.convert_tokens_to_ids(w) for w in similar_words]
embeddings.weight[backtrack_id] = embeddings.weight[similar_ids].mean(dim=0)
```

### Hard Attention Mask

```python
# Corrections cannot attend to errors
# Tokens: [P P P] [E E] [B B] [C C]
# C row:  [1 1 1   0 0   0 0   1 1]  ← Errors blocked!
```

### KV-Cache Rewinding (Inference)

```python
# When backtrack token generated:
generated_tokens.pop()  # Remove last token
past_key_values = rewind_kv_cache(past_key_values, num_tokens=2)
```

---

## Expected Improvements

With Priority 1-4 implementations:

| Metric | Current | Expected |
|--------|---------|----------|
| GSM8K Accuracy | ~45% | 55-65% |
| Training Stability | Poor | Stable |
| Backtrack Precision | Low | High |
| Correction Quality | Low | High |

---

## Files Created

```
docs/research/claude/
├── literature_review.md       (12 KB) - 20+ papers reviewed
├── analysis_why_not_working.md (8 KB) - 7 issues diagnosed  
├── novel_techniques.md        (18 KB) - 10 techniques with code
├── implementation_guide.md    (15 KB) - Step-by-step guide
└── summary.md                 (5 KB)  - This file
```

---

## Next Steps

1. **Review** the Implementation Guide
2. **Implement** Priority 1 fixes (Masked-Error SFT + Token Init)
3. **Run** training with new configuration
4. **Evaluate** on GSM8K with backtrack generation
5. **Iterate** based on results
6. **Consider** advanced techniques if plateau reached

---

## Quick Reference

### The Core Fix

```
Before: Loss on [Prompt] [Error] [Backtrack] [Correction]
After:  Loss on [Prompt] [MASKED] [Backtrack] [Correction]
                         ^^^^^^^^
                    Set to -100 (ignored)
```

### The Core Principle

**Don't teach the model to make mistakes. Teach it to recognize and fix them.**

---

## Contact

For questions about this research:
- Review the detailed documents in `docs/research/claude/`
- The existing analysis in `docs/research/backtrack_analysis.md` provides additional context
- See code examples in the implementation guide for integration patterns
