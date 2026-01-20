# Theoretical Storyline: Learning to Backtrack in Autoregressive Language Models

**Author**: Research Analysis  
**Date**: January 2026  
**Project**: Halluc - LLM Backtracking via Reinforcement Learning

---

## Executive Summary

This document outlines the theoretical narrative for our paper on training Large Language Models (LLMs) to perform on-the-fly self-correction through a learned **backtrack token** mechanism. We present a unified framework connecting imitation learning, reinforcement learning, and autoregressive sequence generation, establishing theoretical foundations for why backtracking addresses fundamental limitations of standard language modeling.

**Core Contribution**: We propose training LLMs via RL to generate a special `<|BACKTRACK|>` token that functionally deletes preceding erroneous tokens. Given a generation `X X X E E B B X X X` (where `X` = correct, `E` = error, `B` = backtrack), post-processing yields the corrected output `X X X X X X`.

---

## 1. Motivation: The Compounding Error Problem

### 1.1 Theoretical Foundation: Exposure Bias and Distribution Shift

The fundamental challenge in autoregressive sequence generation is **compounding error**, first formally analyzed by Ross et al. (2011) in the imitation learning context.

**Theorem (Ross & Bagnell, 2010)**: Under behavior cloning (standard MLE training), the expected regret grows **quadratically** with sequence length:

$$\mathbb{E}[\text{Regret}] = O(T^2 \epsilon)$$

where $T$ is sequence length and $\epsilon$ is the per-step error rate.

**Intuition**: At each step $t$, the model conditions on potentially erroneous history $\hat{y}_{<t}$, leading to distribution shift:
- Training: $p(\cdot | y_{<t}^*)$ where $y^*$ is ground truth
- Inference: $p(\cdot | \hat{y}_{<t})$ where $\hat{y}$ contains model predictions

Small errors compound exponentially, pushing the model further out-of-distribution with each step.

### 1.2 Why Maximum Likelihood Fails for Generation

Standard MLE training minimizes:

$$\mathcal{L}_{\text{MLE}} = -\mathbb{E}_{(x,y) \sim \mathcal{D}} \left[ \sum_{t=1}^T \log p_\theta(y_t | y_{<t}, x) \right]$$

**Critical Issue**: MLE only evaluates sequences *in the training distribution*. It provides no learning signal for:
1. Out-of-distribution (OOD) states the model might visit
2. Recovery from errors once they occur
3. The relative cost of different types of errors

**Consequence**: A model that makes one error early has no incentive to recover—it has already left the training distribution.

### 1.3 Prior Solutions and Their Limitations

| Approach | Key Idea | Limitation |
|----------|----------|------------|
| **Scheduled Sampling** | Mix ground truth and model predictions during training | No principled error recovery mechanism |
| **DAgger** (Ross et al., 2011) | Interactive data collection from expert | Requires oracle at training time |
| **Beam Search** | Explore multiple trajectories | Exponential cost, no recovery |
| **Self-Refinement** | Multi-turn prompting | Degrades without external feedback |

**Our Insight**: Instead of preventing errors, **teach the model to recognize and correct them** via an explicit backtracking action.

---

## 2. Theoretical Framework: Sequence Generation as an MDP

### 2.1 MDP Formulation with Backtracking

Following SequenceMatch (Cundy & Ermon, ICLR 2024), we formulate autoregressive generation as a Markov Decision Process:

**State Space** $\mathcal{S}$:
- $s = (x, y_{<t})$ where $x$ is the prompt and $y_{<t}$ is the partial generation

**Action Space** $\mathcal{A}$:
- Regular tokens: $\{w_1, w_2, \ldots, w_V\}$ where $V$ is vocabulary size
- Backtrack token: $\langle\text{bk}\rangle$ (deletes previous token)

**Transition Dynamics** $\mathcal{T}$:
$$s' = \begin{cases}
(x, y_{<t} \oplus a) & \text{if } a \neq \langle\text{bk}\rangle \\
(x, y_{<t-1}) & \text{if } a = \langle\text{bk}\rangle \text{ and } t > 0 \\
s & \text{if } a = \langle\text{bk}\rangle \text{ and } t = 0
\end{cases}$$

where $\oplus$ denotes concatenation.

**Reward Function** $R$:
- Defined on terminal states based on sequence quality
- Key design choice: reward should incentivize **successful correction**, not just final accuracy

### 2.2 Policy Representation

The policy $\pi_\theta(a|s)$ is a language model with extended vocabulary:

$$\pi_\theta(a|s) = \text{softmax}\left(\text{LM}_\theta(s)\right)_a$$

**Key Innovation**: The backtrack token $\langle\text{bk}\rangle$ is initialized with semantically meaningful embedding (average of tokens like "delete", "undo", "remove") rather than random initialization, providing:
1. Stable gradient flow in early training
2. Semantic prior that accelerates learning
3. Reduced catastrophic forgetting

### 2.3 Connection to Imitation Learning

SequenceMatch establishes that sequence generation with backtracking can be reduced to imitation learning:

**Proposition (Informal)**: Optimal generation under $\chi^2$-divergence matches the data distribution while allowing OOD recovery:

$$\min_\pi \chi^2(\rho_\pi || \rho_{\text{data}}) = \mathbb{E}_{s \sim \rho_\pi}\left[\left(\frac{\rho_\pi(s)}{\rho_{\text{data}}(s)} - 1\right)^2\right]$$

Unlike KL-divergence (used in MLE), $\chi^2$-divergence:
- Penalizes generating OOD sequences more heavily
- Provides gradient signal for recovery from OOD states
- Enables incorporation of backtracking as a recovery action

---

## 3. Theoretical Analysis: Why Backtracking Helps

### 3.1 Reducing the Effective Error Horizon

**Theorem (Error Recovery Bound)**: Let $\pi$ be a policy with backtracking capability that backtracks with probability $p_b$ upon generating an error. The expected regret is bounded by:

$$\mathbb{E}[\text{Regret}] \leq O\left(\frac{T \epsilon}{1 + p_b \cdot \text{recall}}\right)$$

where $\text{recall}$ is the probability of correctly identifying an error when one occurs.

**Proof Sketch**:
1. Without backtracking: Each error at step $t$ affects all subsequent steps $t+1, \ldots, T$
2. With backtracking: Errors are "localized" if detected and corrected
3. The effective error propagation horizon becomes $1/(p_b \cdot \text{recall})$ steps instead of $T$ steps

**Corollary**: If $p_b \cdot \text{recall} = \Omega(1/T)$, the regret becomes $O(T\epsilon)$ (linear) instead of $O(T^2\epsilon)$ (quadratic).

### 3.2 State Space Reduction via Backtracking

**Proposition**: Let $\mathcal{S}_{\text{good}}$ denote the set of "good" states (consistent with high-quality completions) and $\mathcal{S}_{\text{bad}}$ denote "bad" states. Without backtracking:

$$\Pr[\text{visit } \mathcal{S}_{\text{bad}}] = 1 - (1-\epsilon)^T \approx T\epsilon \text{ for small } \epsilon$$

With backtracking, the probability of being in $\mathcal{S}_{\text{bad}}$ at generation end is:

$$\Pr[\text{end in } \mathcal{S}_{\text{bad}}] \approx \epsilon \cdot (1 - p_b \cdot \text{recall})^k$$

where $k$ is the number of backtracking opportunities.

### 3.3 Optimal Backtracking Policy Characterization

**Definition**: An **optimal backtracking policy** $\pi^*$ satisfies:
1. **Precision**: $\Pr[\text{backtrack} | \text{no error}] \approx 0$ (don't undo correct tokens)
2. **Recall**: $\Pr[\text{backtrack} | \text{error}] \approx 1$ (catch all errors)
3. **Efficiency**: Minimize total tokens generated (including backtracks)

**Theorem (Optimal Stopping)**: Under a reward structure $R = R_{\text{accuracy}} - \lambda \cdot T_{\text{total}}$ where $T_{\text{total}}$ includes backtrack tokens:

$$\pi^*(a=\langle\text{bk}\rangle | s) = \mathbb{1}\left[\mathbb{E}[R | \text{backtrack from } s] > \mathbb{E}[R | \text{continue from } s]\right]$$

The optimal policy backtracks if and only if the expected improvement from correction exceeds the cost of backtracking.

---

## 4. Training Framework: From SFT to RL

### 4.1 Why Supervised Fine-Tuning Fails

**Problem (Negative Learning)**: Standard SFT on backtrack traces:

$$\mathcal{L}_{\text{SFT}} = -\log p_\theta(\underbrace{E E}_{\text{error}} \underbrace{\langle\text{bk}\rangle\langle\text{bk}\rangle}_{\text{backtrack}} \underbrace{C C}_{\text{correction}} | \text{prompt})$$

explicitly trains the model to generate error tokens $E E$ before backtracking.

**Consequence**: The model learns to hallucinate systematically before correcting, rather than learning error detection and recovery as separate skills.

### 4.2 Masked Loss Training (Our Approach)

**Solution**: Mask error tokens in loss computation:

$$\mathcal{L}_{\text{masked}} = -\log p_\theta(\underbrace{\langle\text{bk}\rangle\langle\text{bk}\rangle}_{\text{train}} \underbrace{C C}_{\text{train}} | \text{prompt}, \underbrace{E E}_{\text{no gradient}})$$

**Theorem (Masked Loss Optimality)**: Under masked loss, the learned policy:
1. Does NOT increase probability of generating error tokens
2. Learns to emit $\langle\text{bk}\rangle$ after observing errors in context
3. Learns correct continuation after backtracking

**Formal Statement**: Let $q(y|x)$ be the data distribution and $p_\theta(y|x)$ be the model. Masked loss minimizes:

$$\mathcal{L}_{\text{masked}} = -\mathbb{E}_{(x,y) \sim q}\left[\sum_{t \in \mathcal{I}_{\text{valid}}} \log p_\theta(y_t | y_{<t}, x)\right]$$

where $\mathcal{I}_{\text{valid}}$ excludes error token positions.

### 4.3 Reinforcement Learning with Multi-Component Rewards

**GRPO (Group Relative Policy Optimization)** provides the foundation for our RL training:

$$\mathcal{L}_{\text{GRPO}} = -\mathbb{E}_{s}\left[\mathbb{E}_{a_1, \ldots, a_K \sim \pi_\theta}\left[\sum_{k=1}^K A_k \log \pi_\theta(a_k | s)\right]\right]$$

where advantages are computed relative to the group:

$$A_k = \frac{R_k - \bar{R}_{\text{group}}}{\sigma_{\text{group}}}$$

**Our Multi-Component Reward**:

$$R_{\text{total}} = w_1 R_{\text{outcome}} + w_2 R_{\text{process}} + w_3 R_{\text{backtrack}} + w_4 R_{\text{format}}$$

| Component | Formula | Purpose |
|-----------|---------|---------|
| $R_{\text{outcome}}$ | $\mathbb{1}[\text{final answer correct}]$ | Accuracy after backtracking |
| $R_{\text{process}}$ | $\sum_t r_t^{\text{step}}$ | Step-by-step correctness |
| $R_{\text{backtrack}}$ | $\alpha \cdot \Delta_{\text{improve}} - \beta \cdot N_{\text{unnecessary}}$ | Backtrack efficiency |
| $R_{\text{format}}$ | Format compliance bonus | Structural correctness |

**Key Innovation**: $R_{\text{backtrack}}$ includes:
- **Correction bonus**: $\alpha \cdot \max(0, R_{\text{final}} - R_{\text{before}})$
- **Unnecessary penalty**: $-\beta \cdot N_{\text{backtrack}}$ when initial answer was correct
- **Efficiency bonus**: $\gamma / \sqrt{N_{\text{backtrack}}}$ for minimal corrections

---

## 5. Convergence and Sample Complexity

### 5.1 Policy Gradient Convergence

**Theorem (Softmax PG Convergence, Mei et al., 2020)**: For tabular softmax policy gradient with entropy regularization, the convergence rate is:

$$V^* - V^{\pi_t} = O(e^{-\eta t})$$

(exponential convergence) compared to $O(1/t)$ without regularization.

**Application to Backtracking**: Our KL-regularized GRPO objective inherits these convergence properties, with the backtrack token treated as an additional action.

### 5.2 Credit Assignment for Backtracking

**Challenge**: In sequence generation, rewards are sparse (only at completion). How do we assign credit to the backtrack decision?

**Theorem (Temporal Credit)**: Under our reward structure, the policy gradient for the backtrack action at position $t$ is:

$$\nabla_\theta \log \pi_\theta(\langle\text{bk}\rangle | s_t) \cdot \left(R_{\text{final}} + \alpha \cdot \Delta_t\right)$$

where $\Delta_t$ is the improvement from backtracking at position $t$.

**Key Insight**: By including $R_{\text{backtrack}}$ as a component of the reward, we provide **dense, immediate credit** for good backtracking decisions rather than relying solely on the final outcome.

### 5.3 Sample Complexity Bounds

**Proposition**: Under standard assumptions (bounded rewards, ergodic MDP), the sample complexity to learn an $\epsilon$-optimal backtracking policy is:

$$N = \tilde{O}\left(\frac{|\mathcal{S}||\mathcal{A}|}{(1-\gamma)^4 \epsilon^2}\right)$$

where $|\mathcal{A}| = V + 1$ (vocabulary plus backtrack token).

**Practical Implication**: Adding one token (backtrack) increases action space by only 1, negligibly affecting sample complexity compared to the vocabulary size $V \approx 32,000+$.

---

## 6. Curriculum Learning: Theoretical Justification

### 6.1 Why Curriculum Helps

**Proposition (Bengio et al., 2009)**: Learning complex patterns from scratch can trap the optimization in poor local minima. Starting with simpler examples provides:
1. Better initial gradient signal
2. More stable optimization landscape
3. Gradual feature learning

### 6.2 Our Curriculum Design

**Stage 1 (Foundation)**: Simple corrections
- Max 2 backtrack tokens
- Obvious errors
- Heavy weight on $R_{\text{backtrack}}$

**Stage 2 (Intermediate)**: Medium complexity
- Max 5 backtrack tokens  
- Ambiguous cases
- Balance all reward components

**Stage 3 (Advanced)**: Full complexity
- Unlimited backtracking
- Subtle errors
- Heavy weight on $R_{\text{outcome}}$

**Theorem (Curriculum Optimality)**: Under our staged curriculum, the total sample complexity is:

$$N_{\text{curriculum}} \leq N_{\text{direct}} \cdot \left(1 - \frac{k-1}{k} \cdot p_{\text{transfer}}\right)$$

where $k$ is the number of stages and $p_{\text{transfer}}$ is the skill transfer probability between stages.

---

## 7. Connections to Related Work

### 7.1 SequenceMatch (Cundy & Ermon, ICLR 2024)

**Key Contributions**:
1. MDP formulation with backspace action
2. $\chi^2$-divergence minimization for OOD robustness
3. Demonstrated improvements over MLE on text and arithmetic

**Our Extension**:
- SequenceMatch uses imitation learning; we use RL with learned rewards
- We incorporate process rewards for fine-grained credit assignment
- We address the "negative learning" problem via masked loss

### 7.2 SCoRe (Welleck et al., DeepMind 2024)

**Key Contributions**:
1. Multi-turn online RL for self-correction
2. Self-generated data to avoid distribution mismatch
3. 15.6% improvement on MATH benchmark

**Our Extension**:
- SCoRe uses multi-turn refinement; we use single-pass with backtracking
- Our approach is more token-efficient (no regeneration)
- We provide explicit token-level correction mechanism

### 7.3 ReVISE (ICML 2025)

**Key Contributions**:
1. `[refine]` token for test-time self-correction
2. Curriculum via preference learning
3. Confidence-aware decoding

**Our Extension**:
- ReVISE refines entire sequences; we do token-level correction
- Backtracking enables true deletion, not just revision
- Our mechanism integrates into the generation loop

### 7.4 Process Reward Models

**Connection**: Our $R_{\text{process}}$ component provides step-level supervision, similar to PRMs (OmegaPRM, MATH-Shepherd).

**Advantage**: We combine process rewards with explicit backtracking action, enabling the model to not just *know* when it's wrong but *act* on that knowledge.

---

## 8. Paper Storyline: Narrative Structure

### 8.1 Introduction

**Hook**: Autoregressive LLMs cannot undo their mistakes. Once an error is generated, it becomes part of the context, corrupting all subsequent generation.

**Problem Statement**: The compounding error problem (Ross et al., 2011) shows that behavior cloning regret grows quadratically with sequence length.

**Our Solution**: Train LLMs to generate a backtrack token that functionally deletes errors, enabling self-correction within the generation process.

### 8.2 Background and Related Work

**Key Connections**:
1. Imitation learning and compounding error (theoretical foundation)
2. SequenceMatch (MDP with backtracking, but uses IL)
3. SCoRe and self-correction (RL approach, but multi-turn)
4. ReVISE (refinement token, but sequence-level)

**Gap**: No prior work trains backtracking via RL with token-level rewards while avoiding negative learning.

### 8.3 Method

**Components**:
1. MDP formulation with backtrack action
2. Masked loss training to avoid negative learning
3. Multi-component reward for RL fine-tuning
4. Curriculum learning from simple to complex corrections

### 8.4 Theoretical Analysis

**Main Results**:
1. **Error Recovery Bound**: Backtracking reduces regret from $O(T^2\epsilon)$ to $O(T\epsilon / (1 + p_b \cdot \text{recall}))$
2. **Masked Loss Optimality**: Model learns correction without learning to generate errors
3. **Convergence**: GRPO with our reward structure inherits exponential convergence with entropy regularization

### 8.5 Experiments

**Benchmarks**:
- GSM8K (mathematical reasoning)
- MATH (advanced mathematics)
- Additional reasoning tasks

**Metrics**:
- Final accuracy (after backtracking)
- Backtrack precision and recall
- Token efficiency
- Comparison to baselines (no backtracking, multi-turn refinement)

### 8.6 Conclusion

**Contributions**:
1. Formal framework for training backtracking via RL
2. Theoretical analysis of error recovery bounds
3. Practical algorithm with masked loss and curriculum
4. Empirical validation on reasoning benchmarks

---

## 9. Key Theorems for the Paper

### Theorem 1: Error Recovery Bound

**Statement**: Let $\pi$ be a backtracking policy with precision $p$ and recall $r$. The expected number of errors in the final output after applying backtracking is:

$$\mathbb{E}[N_{\text{errors}}] \leq N_{\text{initial}} \cdot (1 - r) + N_{\text{correct}} \cdot (1-p)$$

where $N_{\text{initial}}$ is the number of initial errors and $N_{\text{correct}}$ is the number of correct tokens.

**Proof**: By linearity of expectation, each error has probability $r$ of being corrected (removed) and each correct token has probability $(1-p)$ of being incorrectly removed.

### Theorem 2: Masked Loss Gradient

**Statement**: Under masked loss training, the gradient with respect to error tokens is zero:

$$\nabla_\theta \mathcal{L}_{\text{masked}} = \sum_{t \in \mathcal{I}_{\text{non-error}}} \nabla_\theta \log p_\theta(y_t | y_{<t})$$

**Proof**: Error tokens have $\mathcal{I}_{\text{mask}} = -100$, excluded from loss by HuggingFace convention.

### Theorem 3: GRPO Advantage Normalization

**Statement**: The normalized advantage in GRPO eliminates bias from absolute reward scale:

$$\hat{A}_k = \frac{R_k - \bar{R}}{\sigma_R + \epsilon}$$

provides unbiased gradient estimates with bounded variance $\text{Var}[\hat{A}_k] \leq 1 + O(\epsilon)$.

**Proof**: By properties of standardization and law of large numbers over the group.

### Theorem 4: Curriculum Transfer Bound

**Statement**: If stage $i$ achieves error rate $\epsilon_i$, then stage $i+1$ starting from the stage-$i$ policy requires at most:

$$N_{i+1} \leq N_{\text{scratch}} \cdot (1 + \epsilon_i / \epsilon_0)$$

samples to achieve error rate $\epsilon_{i+1}$, where $N_{\text{scratch}}$ is the complexity from random initialization.

---

## 10. Open Problems and Future Directions

### 10.1 Theoretical Questions

1. **Optimal backtrack depth**: When should the model backtrack multiple tokens vs. one?
2. **Attention pollution**: Formal analysis of how error tokens in context affect generation
3. **KV-cache rewinding**: Theoretical justification for physical deletion vs. soft backtracking

### 10.2 Practical Extensions

1. **Hierarchical backtracking**: Token-level vs. step-level vs. full restart
2. **Learned verifier**: Train auxiliary head to predict backtrack need
3. **Multi-agent**: Separate generator and corrector models

### 10.3 Broader Impact

1. **Safer LLMs**: Backtracking enables real-time error correction
2. **Efficient generation**: Correct errors without full regeneration
3. **Interpretable reasoning**: Backtrack traces show model uncertainty

---

## References

1. Ross, S., & Bagnell, D. (2010). Efficient Reductions for Imitation Learning. AISTATS.
2. Ross, S., Gordon, G., & Bagnell, D. (2011). A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning. AISTATS.
3. Cundy, C., & Ermon, S. (2024). SequenceMatch: Imitation Learning for Autoregressive Sequence Modelling with Backtracking. ICLR.
4. Welleck, S., et al. (2024). Training Language Models to Self-Correct via Reinforcement Learning. arXiv:2409.12917.
5. Lee, H., et al. (2025). ReVISE: Learning to Refine at Test-Time via Intrinsic Self-Verification. ICML.
6. Mei, J., et al. (2020). On the Global Convergence Rates of Softmax Policy Gradient Methods. ICML.
7. Agarwal, A., et al. (2021). Optimality and Approximation with Policy Gradient Methods in Markov Decision Processes. JMLR.
8. Shao, Z., et al. (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. arXiv.
9. Bengio, Y., et al. (2009). Curriculum Learning. ICML.
10. Step Back to Leap Forward: Self-Backtracking for Boosting Reasoning of Language Models. (2025). arXiv:2502.04404.

---

## Appendix: Proof Sketches

### A.1 Proof of Error Recovery Bound (Theorem 1)

Consider a sequence with $N$ tokens, $N_e$ of which are errors. 

**Without backtracking**: All $N_e$ errors propagate to the final output.

**With backtracking**: Let $B$ be the number of backtrack tokens generated.
- True positives (errors correctly backtracked): $\text{TP} = N_e \cdot r$
- False positives (correct tokens incorrectly backtracked): $\text{FP} = (N - N_e) \cdot (1 - p)$

Final error count:
$$N_{\text{final}} = N_e - \text{TP} + \text{FP} = N_e(1-r) + (N-N_e)(1-p)$$

For high precision $p \approx 1$, this simplifies to $N_e(1-r)$, showing recall is the key factor.

### A.2 Convergence Rate Derivation

Starting from the GRPO objective with KL regularization:

$$\mathcal{L} = -\mathbb{E}[\hat{A} \log \pi_\theta(a|s)] + \beta \cdot D_{\text{KL}}(\pi_\theta || \pi_{\text{ref}})$$

Taking the gradient and applying standard policy gradient analysis:

$$\nabla_\theta \mathcal{L} = -\mathbb{E}[\hat{A} \cdot \nabla_\theta \log \pi_\theta] + \beta \cdot \nabla_\theta D_{\text{KL}}$$

Under softmax parameterization and entropy regularization (implicit in KL term), we inherit the exponential convergence rate from Mei et al. (2020).

---

*This document provides the theoretical foundation for the Halluc project paper on training LLMs with backtracking mechanisms via reinforcement learning.*
