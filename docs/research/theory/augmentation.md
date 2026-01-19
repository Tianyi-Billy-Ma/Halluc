# Data Augmentation for Backtrack Token Training

**Author**: Research Analysis  
**Date**: January 2026  
**Project**: Halluc - LLM Backtracking via Reinforcement Learning

---

## Overview

This document formalizes three data augmentation strategies for training autoregressive models with backtracking capability. Each method constructs training sequences that interleave error tokens with backtrack tokens $\langle\text{BK}\rangle$ and corrections, enabling the model to learn error detection and recovery.

---

## 1. Sequence Alignment Augmentation

To train a policy model $p_\theta$ that recognizes and corrects errors via backtracking, we introduce a data augmentation procedure based on sequence alignment (Myers, 1986).

### 1.1 Generation

Given a prompt sequence $x$ and reference completion $y = (y_1, \ldots, y_n)$, we first sample a completion from the policy model:

$$\hat{y} = (\hat{y}_1, \hat{y}_2, \ldots, \hat{y}_m) \sim p_\theta(\cdot | x)$$

### 1.2 Matching Block Decomposition

We compute the maximal matching blocks between the reference and generated sequences:

$$\mathcal{M}(y, \hat{y}) = \{(a_i, b_i, \ell_i)\}_{i=1}^{K}, \quad \text{s.t. } y_{a_i : a_i + \ell_i} = \hat{y}_{b_i : b_i + \ell_i}$$

where the decomposition is constrained to be:
- **Maximal**: No block can be extended in either direction
- **Non-overlapping**: $a_i + \ell_i \leq a_{i+1}$ and $b_i + \ell_i \leq b_{i+1}$
- **Order-preserving**: $a_1 < a_2 < \cdots < a_K$ and $b_1 < b_2 < \cdots < b_K$

### 1.3 Error Identification

From this decomposition, we identify error indices as positions in $\hat{y}$ not covered by any matching block:

$$\mathcal{E} = \{1, \ldots, m\} \setminus \bigcup_{i=1}^{K} [b_i, b_i + \ell_i)$$

### 1.4 Augmented Sequence Construction

The augmented training sequence $y^*$ is constructed by processing $\hat{y}$ sequentially, preserving matched tokens while appending backtrack tokens and corrections after each error span. For the segment between matching blocks $i$ and $i+1$:

$$y^* = (\ldots, \underbrace{\hat{y}_{b_i}, \ldots, \hat{y}_{b_i + \ell_i - 1}}_{\text{matched}}, \underbrace{\hat{y}_{b_i + \ell_i}, \ldots, \hat{y}_{b_{i+1} - 1}}_{\text{error}}, \underbrace{\langle\text{BK}\rangle^{k_i}}_{\text{backtrack}}, \underbrace{y_{a_i + \ell_i}, \ldots, y_{a_{i+1} - 1}}_{\text{correction}}, \ldots)$$

where $k_i = b_{i+1} - (b_i + \ell_i)$ denotes the number of error tokens between blocks $i$ and $i+1$.

### 1.5 Properties

This augmentation procedure exhibits several desirable properties:

1. **Naturalistic errors**: Error patterns arise from the model's own generation, reflecting realistic failure modes
2. **Automatic alignment**: No manual annotation required; errors and corrections are derived algorithmically
3. **Adaptive difficulty**: As the model improves, generated sequences align more closely with references, naturally reducing augmentation intensity

---

## 2. Symbolic Template Augmentation

To expose the model to challenging yet plausible error patterns, we introduce a hard sample augmentation strategy inspired by symbolic reasoning benchmarks (Mirzadeh et al., 2024).

### 2.1 Motivation

Random errors may not reflect the structured mistakes that arise in reasoning tasks. Symbolic template augmentation constructs errors that are *locally coherent* but *globally incorrect*, forcing the model to learn fine-grained error detection.

### 2.2 Formulation

Let $y = (y_1, \ldots, y_n)$ denote a reference sequence derived from a symbolic template $\mathcal{T}$, and let $y' = (y'_1, \ldots, y'_{n'})$ denote an alternative sequence derived from a perturbed template $\mathcal{T}'$ (e.g., with modified numerical values or variable bindings).

We compute the matching block decomposition between $y$ and $y'$:

$$\mathcal{M}(y, y') = \{(a_i, a'_i, \ell_i)\}_{i=1}^{K}$$

The augmented sequence $y^*$ is constructed by interleaving segments from $y'$ (as plausible errors) with backtrack tokens and corrections from $y$:

$$y^* = (\ldots, \underbrace{y_{a_i}, \ldots, y_{a_i + \ell_i - 1}}_{\text{shared}}, \underbrace{y'_{a'_i + \ell_i}, \ldots, y'_{a'_{i+1} - 1}}_{\text{hard error}}, \underbrace{\langle\text{BK}\rangle^{k_i}}_{\text{backtrack}}, \underbrace{y_{a_i + \ell_i}, \ldots, y_{a_{i+1} - 1}}_{\text{correction}}, \ldots)$$

where $k_i = a'_{i+1} - (a'_i + \ell_i)$ denotes the length of the inserted error span.

### 2.3 Properties

1. **Semantic plausibility**: Errors are drawn from valid (but incorrect) reasoning traces, not random tokens
2. **Structural preservation**: Error sequences maintain syntactic coherence with the surrounding context
3. **Controlled difficulty**: The degree of perturbation in $\mathcal{T}'$ can be adjusted to control error subtlety

---

## 3. Stochastic Error Injection

As a baseline augmentation strategy, we consider stochastic injection of random error tokens into the reference sequence.

### 3.1 Formulation

Given a reference sequence $y = (y_1, \ldots, y_n)$, we sample an insertion position $i \in \{1, \ldots, n\}$ and error length $k \in \{1, \ldots, k_{\max}\}$. Error tokens $e = (e_1, \ldots, e_k)$ are sampled independently from the vocabulary:

$$e_j \sim \text{Uniform}(\mathcal{V}), \quad j \in \{1, \ldots, k\}$$

The augmented sequence $y^*$ is constructed by inserting the error tokens followed by $k$ backtrack tokens at position $i$:

$$y^* = (y_1, \ldots, y_i, \underbrace{e_1, \ldots, e_k}_{\text{error}}, \underbrace{\langle\text{BK}\rangle^{k}}_{\text{backtrack}}, y_{i+1}, \ldots, y_n)$$

### 3.2 Generalization to Multiple Insertions

The procedure generalizes to multiple insertion points. Let $\mathcal{I} = \{(i_1, k_1), \ldots, (i_L, k_L)\}$ denote a set of insertion positions and corresponding error lengths, with $i_1 < i_2 < \cdots < i_L$. The augmented sequence becomes:

$$y^* = (y_1, \ldots, y_{i_1}, e^{(1)}, \langle\text{BK}\rangle^{k_1}, y_{i_1+1}, \ldots, y_{i_2}, e^{(2)}, \langle\text{BK}\rangle^{k_2}, \ldots, y_n)$$

where $e^{(j)} = (e^{(j)}_1, \ldots, e^{(j)}_{k_j})$ are independently sampled error sequences.

### 3.3 Properties

1. **Simplicity**: No external data or model inference required
2. **Coverage**: Uniform sampling ensures exposure to diverse error patterns
3. **Limitation**: Random errors lack semantic plausibility, potentially leading to trivially detectable patterns

---

## 4. Comparison of Augmentation Strategies

| Strategy | Error Source | Plausibility | Difficulty | Computational Cost |
|----------|--------------|--------------|------------|-------------------|
| Sequence Alignment | Model generation | High | Adaptive | Requires inference |
| Symbolic Template | Alternative templates | High | Controlled | Template construction |
| Stochastic Injection | Random sampling | Low | Low | Minimal |

### 4.1 Curriculum Considerations

We hypothesize that a curriculum combining these strategies may be beneficial:

1. **Stage 1**: Stochastic injection for initial backtrack token learning
2. **Stage 2**: Symbolic template augmentation for structured error patterns
3. **Stage 3**: Sequence alignment augmentation for naturalistic self-correction

This progression exposes the model to increasingly realistic error distributions as training advances.

---

## References

1. Myers, E. W. (1986). An O(ND) Difference Algorithm and Its Variations. *Algorithmica*, 1(1-4), 251–266.

2. Mirzadeh, I., Alizadeh, K., Shahez, H., Tuzel, O., Bengio, S., & Farajtabar, M. (2024). GSM-Symbolic: Understanding the Limitations of Mathematical Reasoning in Large Language Models. *arXiv:2410.05229*.

3. Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano, R., Hesse, C., & Schulman, J. (2021). Training Verifiers to Solve Math Word Problems. *arXiv:2110.14168*.

---

*This document provides the theoretical foundation for data augmentation strategies in backtrack token training.*
