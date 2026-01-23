# GRPO Reward Function Design for Non-Monotonic Sequence Modeling

**Author**: AI Research Assistant  
**Date**: January 11, 2026  
**Project**: N-MAR: Non-Monotonic Autoregressive Modeling via `<UNDO>`  
**Objective**: Design an effective and efficient reward function for training LLMs with the `<UNDO>` token using GRPO

---

## Executive Summary

This document provides a comprehensive guide for designing reward functions to train Large Language Models (LLMs) with a **Non-Monotonic Autoregressive (N-MAR)** framework using Group Relative Policy Optimization (GRPO). The backtracking token (`<UNDO>`) enables LLMs to prune divergent generation paths and self-correct during inference.

**Key Findings:**
1.  **Multi-Component Reward Design is Essential**: A single reward metric is insufficient for the complex behavior of learning *when* to undo and *how* to correct.
2.  **Process Rewards Outperform Outcome Rewards**: Step-level feedback is crucial for teaching the model to identify deviations early.
3.  **Curriculum Learning is Critical**: Gradually increasing task difficulty prevents reward hacking (e.g., unnecessary backtracking) and improves learning efficiency.
4.  **Reward Shaping Must Incentivize Efficient Pruning**: Bonus rewards for successfully pruning divergent paths are essential.

**Recommended Architecture**: A weighted multi-objective reward function combining:
-   **Outcome Accuracy** (final answer correctness after undo operations)
-   **Process Quality** (validity of intermediate steps)
-   **Backtrack Efficiency** (penalizing excessive or unnecessary use of `<UNDO>`)
-   **Format Compliance** (structured output requirements)

---

## 1. Literature Review & Theoretical Foundation

### 1.1 Group Relative Policy Optimization (GRPO)

#### 1.1.1 Core Mechanism

GRPO (DeepSeekMath, 2024) is a critic-free RL algorithm ideal for this task because it computes advantages based on a group of outputs for the same prompt. This allows the model to explore different "undo" strategies (e.g., pruning early vs. late) and learn which is most effective relative to the group mean.

**Relevance to N-MAR**:
-   **Exploration**: Multiple generations allow the model to try different "branching" points for the `<UNDO>` token.
-   **Relative Feedback**: If one generation prunes a deviation effectively and another doesn't, GRPO naturally reinforces the former.

### 1.2 Self-Correction & Non-Monotonicity

#### 1.2.1 The N-MAR Paradigm

Standard autoregressive models suffer from **error propagation**: one bad token pollutes the context for all future tokens. N-MAR breaks this monotonicity.

**Key Insights**:
1.  **Pruning vs. Editing**: Unlike "editing" approaches that rewrite entire blocks, N-MAR uses a token-level `<UNDO>` operator to strictly *remove* the tail of the sequence.
2.  **Efficiency**: The reward must balance *capability* (can it fix errors?) with *efficiency* (does it fix them with minimal tokens?).

#### 1.2.2 Process vs. Outcome Rewards

-   **Outcome Rewards**: Did the model eventually get the right answer? (Necessary but sparse).
-   **Process Rewards**: Did the model detect the deviation immediately? (Crucial for efficiency).

**Recommendation**: Use **Outcome Rewards** as the primary signal but weight **Efficiency Rewards** heavily to force early detection of deviations.

---

## 2. Reward Function Design Recommendations

### 2.1 Proposed Multi-Component Reward Architecture

```
R_total = w₁ * R_outcome + w₂ * R_process + w₃ * R_efficiency + w₄ * R_format
```

### 2.2 Component 1: Outcome Accuracy Reward

**Purpose**: Evaluate the correctness of the sequence *after* all `<UNDO>` operations are applied.

**Logic**:
1.  Parse the raw generation.
2.  Apply the `<UNDO>` logic (stack pop).
3.  Compare the resulting final answer to the ground truth.

**Why it matters**: Ultimately, the non-monotonic path must yield a correct result.

### 2.3 Component 2: Process Quality Reward

**Purpose**: Evaluate the validity of reasoning steps.

**Logic**:
-   If the model generates a valid step, reward it.
-   If the model generates an invalid step *and immediately follows it with `<UNDO>`*, reward it (successful detection).
-   If the model generates an invalid step *and continues*, penalize it.

### 2.4 Component 3: Backtrack Efficiency Reward

**Purpose**: Discourage "gaming" the system (e.g., writing good text, undoing it, and rewriting it just to get a 'correction' bonus).

**Sub-Components**:
1.  **Correction Bonus**: High reward if: `Accuracy(Final) > Accuracy(Pre-Undo)`.
2.  **Unnecessary Undo Penalty**: Negative reward if: `Accuracy(Final) == Accuracy(Pre-Undo)` (the undo didn't improve anything).
3.  **Parsimony Bonus**: Higher reward for using fewer `<UNDO>` tokens to achieve the same correction.

### 2.5 Component 4: Format Compliance

**Purpose**: Ensure the `<UNDO>` tokens are generated in valid clusters (not interleaved randomly like `A <UNDO> B <UNDO>`).

---

## 3. Training Strategy

### 3.1 Curriculum Learning

**Stage 1: Mechanics (The "How")**
-   **Task**: Simple synthetic sequences where an error is explicitly injected, and the model must output `<UNDO>` immediately.
-   **Reward**: Heavily weighted towards `R_format` and `R_efficiency`.

**Stage 2: Detection (The "When")**
-   **Task**: Standard problems where the model might naturally drift.
-   **Reward**: Balanced `R_outcome` and `R_process`.

**Stage 3: Optimization (The "Why")**
-   **Task**: Hard reasoning problems.
-   **Reward**: Heavy penalty on unnecessary undos; emphasis on "measure twice, cut once".

---

## 4. Implementation Notes

### 4.1 Applying the `<UNDO>` Logic

```python
def apply_undo(token_ids: list[int], undo_token_id: int) -> list[int]:
    """
    Apply <UNDO> operations to get the effective sequence.
    """
    result = []
    for token_id in token_ids:
        if token_id != undo_token_id:
            result.append(token_id)
        elif result:  # Undo: remove last token
            result.pop()
    return result
```

### 4.2 Detecting Improvement

```python
def compute_improvement(pre_undo_seq, post_undo_seq, ground_truth):
    """
    Did the undo operation actually move the state closer to the truth?
    """
    score_pre = evaluate(pre_undo_seq, ground_truth)
    score_post = evaluate(post_undo_seq, ground_truth)
    return score_post - score_pre
```

---

## 5. Conclusion

By shifting the perspective from "hallucination mitigation" to "non-monotonic modeling," we frame the problem as one of **trajectory optimization**. The reward function is the critical driver that aligns the model's local token choices (to undo or not to undo) with the global objective of generating correct, efficient sequences.
