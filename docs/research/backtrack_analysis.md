# Analysis of Non-Monotonic Autoregressive Modeling via Backtracking

## 1. Problem Definition: The Monotonicity Bottleneck

The fundamental limitation of standard autoregressive (AR) models is **Monotonicity**: sequence generation proceeds in a strictly forward direction ($t \to t+1$). 
$$ P(\mathbf{y}) = \prod_{t=1}^T P(y_t | y_{<t}) $$
Once a token $y_t$ is sampled, it becomes an immutable part of the history $y_{<t+1}$. If $y_t$ is sub-optimal or erroneous (a "deviation"), the model is forced to condition on it forever. This leads to **Error Propagation**: the model tries to make sense of the deviation, often drifting further into low-probability regions (hallucination cascades).

**Proposal**: **Non-Monotonic Autoregressive (N-MAR)** modeling.
We introduce a mechanism to **prune** divergent branches dynamically using a special `<UNDO>` token. A sequence $x_1 x_2 x_3 \langle\text{UNDO}\rangle x_3'$ is functionally equivalent to $x_1 x_2 x_3'$.

---

## 2. Theoretical Analysis of Training Dynamics

### 2.1. The "Negative Learning" Trap in Naive SFT
If we simply train on traces like `[Prompt] -> [Deviation] -> [<UNDO>] -> [Correction]`, standard SFT minimizes the NLL of the *entire* sequence.
$$ \mathcal{L} = - \sum \log P(token_t | history) $$

**The Critical Flaw:** This trains the model to maximize the probability of the **Deviation**.
- The model learns: *"To solve this problem, I must first generate this specific error."*
- This reinforces the generation of deviations rather than their pruning.

**Solution: Masked SFT (mSFT)**
We must mask the loss for the Deviation tokens. The model should only be trained to:
1.  **Detect** deviation (generate `<UNDO>` given deviation context).
2.  **Correct** (generate optimal tokens given pruned context).
3.  **NOT Generate** deviations (gradient = 0).

### 2.2. Context Pollution and "Hard" Backtracking
In a standard Transformer, `<UNDO>` is just another token. The deviation $x_t$ remains in the Attention context.
$$ Attention(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d}})V $$
If $K$ contains keys from the deviation, the model "sees" the mistake even after "undoing" it.

**The "Hard" Backtracking Hypothesis**:
True N-MAR requires the `<UNDO>` token to functionally **erase** the deviation from the model's memory (KV Cache).
- **Inference**: When `<UNDO>` is sampled, physically pop the KV cache stack.
- **Training**: Use a custom Attention Mask where correction tokens cannot attend to pruned tokens.

---

## 3. Literature & Related Work

### 3.1. Non-Monotonic Generation
- **Levenshtein Transformer**: Iterative insertion and deletion. Non-autoregressive.
- **Backspacing in Transformers**: Niche research on "forgetting" mechanisms. Most treat backspace as a vocabulary item, not a structural operator.

### 3.2. Self-Correction
- **SCoRe (DeepMind)**: Training for self-correction via RL.
- **Chain of Hindsight**: Conditioning on "bad outcome" feedback.

---

## 4. Implementation Plan for N-MAR

### Technique A: Masked SFT (The Safety Layer)
**Concept**: Decouple error generation from error correction.
**Implementation**:
Modify `DataCollator` to set `labels` for deviation tokens to `-100`.
- **Loss ON**: `[Prefix]`, `[<UNDO>]`, `[Correction]`
- **Loss OFF**: `[Deviation]`

### Technique B: GRPO for Policy Refinement (The Efficiency Layer)
**Concept**: Use RL to optimize the trade-off between exploration (generating tokens) and pruning (backtracking).
**Reward Function**:
- **Outcome**: +1 for correct final answer.
- **Efficiency**: Penalty for every `<UNDO>` used.
- **Process**: Reward for valid reasoning steps.

### Technique C: Inference-Time Stack Manipulation
**Concept**: Operationalize the N-MAR framework.
**Logic**:
```python
if token == <UNDO>:
    output_ids.pop()
    kv_cache.trim(-2) # Remove UNDO and the token before it
```
This ensures the model's effective context $y_{<t}$ matches the intended pruned sequence.

---

## 5. Conclusion

The N-MAR framework shifts the paradigm from "predicting the next token" to "managing the generation trajectory." By combining **Masked SFT** (to prevent negative learning) with **Stack-Manipulation Inference** (to ensure clean context), we enable robust recovery from autoregressive drift.
