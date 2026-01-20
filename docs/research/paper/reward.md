# Reward Function Design for Backtracking LLMs

**Author**: Research Team  
**Date**: January 2026  
**Project**: Halluc - LLM Self-Correction via Backtracking  

---

## Executive Summary

This document provides a comprehensive literature review and theoretically-grounded design for the reward function used in training LLMs with the `<|BACKTRACK|>` token via Group Relative Policy Optimization (GRPO). 

**Core Design**: A multi-component reward function combining:
1. **Outcome Accuracy**: Final answer correctness after backtracking
2. **Backtrack Efficiency**: Appropriate and efficient use of backtracking
3. **Format Compliance**: Structural output requirements

**Key Theoretical Contributions**:
- Formalization of backtracking as an editable-prefix MDP
- Potential-based reward shaping for policy-invariant dense feedback
- Incentive-compatibility conditions preventing reward hacking
- State augmentation for Markov reward definitions

---

## 1. Literature Review

### 1.1 Group Relative Policy Optimization (GRPO)

#### DeepSeekMath (Shao et al., 2024)
**Venue**: arXiv 2402.03300

GRPO is an efficient RL algorithm that eliminates the need for a separate value network by using group-based sampling to compute relative advantages:

$$A(s_i) = \frac{r_i - \mu(r)}{\sigma(r)}$$

where $r_i$ is the reward for sample $i$ and $r$ is the vector of rewards for the group.

**Key Properties**:
- Critic-free architecture reduces memory overhead
- Group normalization reduces variance and improves stability
- Well-suited for comparing different backtracking strategies

### 1.2 Process vs. Outcome Reward Models

#### Let's Verify Step by Step (Lightman et al., ICLR 2024)
**Key Finding**: Process supervision (step-level feedback) outperforms outcome supervision (final answer only) by 16.5% error reduction on complex reasoning tasks.

**Mathematical Formulation**:
$$R_{\text{process}}(\pi) = \sum_{t=1}^{T} r_{\text{step}}(\pi_t)$$

versus outcome-only:
$$R_{\text{outcome}}(\pi) = r_{\text{final}}$$

**Implication**: For backtracking, we need dense process rewards to guide *when* to backtrack, not just whether the final answer is correct.

### 1.3 Self-Correction via Reinforcement Learning

#### SCoRe (Kumar et al., ICLR 2025 Oral)
**Key Contributions**:
- Multi-turn online RL framework for self-correction
- Uses self-generated data to avoid distribution mismatch
- Achieves 15.6% improvement on MATH benchmark

**Critical Insight on Negative Learning**:
> "SFT on correction traces trains the model to maximize the probability of the initial incorrect response."

This motivates our Masked SFT approach (see `masked_sft.md`) and informs our reward design to avoid reinforcing errors.

### 1.4 Token-Level Credit Assignment

#### SCAR: Shapley Credit Assignment (Cao et al., ICLR 2026)
Uses Shapley values from cooperative game theory:

$$\phi_i(S) = \sum_{T \subseteq S \setminus \{i\}} \frac{|T|! (|S| - |T| - 1)!}{|S|!} \cdot (R(T \cup \{i\}) - R(T))$$

**Relevance**: Provides principled token-level credit assignment, critical for understanding *which* backtrack action led to improvement.

#### MA-RLHF: Macro Actions (Chai et al., ICLR 2025)
Introduces macro actions (sequences of tokens) to reduce credit assignment complexity.

**Relevance**: Backtracking can be viewed as a macro action that groups "delete previous token" operations.

---

## 2. Theoretical Foundations

### 2.1 Potential-Based Reward Shaping

#### Theorem (Ng, Harada, Russell, ICML 1999)
A shaping reward function $F(s, a, s')$ preserves optimal policies **if and only if** it is potential-based:

$$F(s, a, s') = \gamma \Phi(s') - \Phi(s)$$

for some potential function $\Phi: \mathcal{S} \to \mathbb{R}$.

**Corollary (Advantage Invariance)**:
For potential-based shaping, the advantage function remains invariant:
$$\tilde{A}^\pi(s, a) = A^\pi(s, a)$$

**Implication**: If our shaping rewards can be written as potential differences, they will not distort the optimal policy—they only provide denser feedback to accelerate learning.

### 2.2 Reward Decomposition Theorem

#### Calculus on MDPs (arXiv:2208.09570, 2022)
Every reward function $R$ uniquely decomposes as:

$$R = R_{\text{divergence-free}} + \text{grad}(\Phi)$$

where:
- $R_{\text{divergence-free}}$ is the canonical reward (defines the task)
- $\text{grad}(\Phi) = \gamma\Phi(s') - \Phi(s)$ is the shaping component

**Implication**: We can decompose our multi-component reward into a base task reward and shaping terms, analyzing each for policy invariance.

### 2.3 Multi-Objective RL Convergence

#### Lexicographic Multi-Objective RL (IJCAI 2022)
Under finite state-action spaces and appropriate learning rates, lexicographic policy gradient algorithms converge to lexicographically $\epsilon$-optimal policies.

**Application**: We can prioritize objectives:
1. Outcome accuracy (primary)
2. Backtrack efficiency (secondary)

#### Finite-Time Pareto Convergence (ICML 2024)
MOAC achieves finite-time convergence to $\epsilon$-Pareto-stationary solutions with sample complexity $\tilde{O}(\epsilon^{-2} p_{\min}^{-2})$.

### 2.4 Curriculum Learning Theory

#### CRL Performance Guarantee (ICLR 2026)
The final performance gap decomposes as:

$$\mathcal{E}_K \leq \mathcal{E}_{\text{actor}} + \sum_{k=1}^K \mathcal{E}_{\text{eval},k} + \mathcal{E}_{\text{update}} + \mathcal{E}_{\text{curriculum}}$$

**Implication**: Curriculum learning (easy→hard) reduces total sample complexity when tasks are appropriately decomposed.

#### Curriculum as Optimal Transport (NeurIPS 2022)
Optimal policies between curriculum stages satisfy:

$$V^{\pi_{k+1}^*}(\rho_{k+1}) - V^{\pi_k}(\rho_{k+1}) \leq m \cdot W_d(\rho_k, \rho_{k+1})$$

where $W_d$ is the Wasserstein distance between task distributions.

### 2.5 Reward Hacking Prevention

#### Defining and Characterizing Reward Gaming (NeurIPS 2022)

**Definition (Reward Hacking)**: Hacking occurs if:
$$J(\pi, R) < J(\pi_{\text{ref}}, R) \quad \text{when} \quad J(\pi, \tilde{R}) > J(\pi_{\text{ref}}, \tilde{R})$$

**Theorem**: For "open" policy spaces, only reward functions that are affine transformations can be unhackable.

**Implication**: We cannot guarantee zero hacking, but we can minimize it through:
1. Multiple complementary reward signals
2. Regularization against reference policy
3. Hard constraints on extreme behaviors

#### Occupancy Measure Regularization (ICLR 2025)
Regularizing $\chi^2$ divergence between occupancy measures provides provable lower bounds on true reward improvement:

$$|J(\pi_{\text{safe}}, R) - J(\pi, R)| \leq \|\mu_\pi - \mu_{\pi_{\text{safe}}}\|_1$$

---

## 3. MDP Formalization for Backtracking

### 3.1 Editable-Prefix MDP

We formalize backtracking as a deterministic MDP over editable prefixes:

**State Space**: $s_t = x_t$ where $x_t$ is the current sequence after applying backtracks.

**Action Space**: $\mathcal{A} = \mathcal{V} \cup \{\langle\text{BK}\rangle\}$ (vocabulary plus backtrack token).

**Transition Function**:
$$x_{t+1} = \begin{cases} 
x_t \oplus a_t & \text{if } a_t \in \mathcal{V} \\
\text{pop}(x_t) & \text{if } a_t = \langle\text{BK}\rangle
\end{cases}$$

where $\oplus$ denotes concatenation and $\text{pop}(x)$ removes the last token.

**Terminal Condition**: EOS token or length limit.

**Base Reward**: $r^{\text{base}}_t = 0$ for $t < T$, $r^{\text{base}}_T = \mathbf{1}[\text{final answer correct}]$.

### 3.2 State Augmentation for Markov Rewards

To make "improvement-based" and "unnecessary backtrack" rewards Markov, we augment the state:

$$s_t = (x_t, x^{\text{init}}, m_t)$$

where:
- $x_t$: current sequence (after backtracks)
- $x^{\text{init}}$: initial sequence before any backtracking (or flag "initial was correct")
- $m_t$: backtrack count

This augmentation allows us to define rewards like "correction bonus" and "unnecessary penalty" as legitimate functions of $(s_t, a_t, s_{t+1})$.

---

## 4. Reward Function Design

### 4.1 Multi-Component Architecture

$$R_{\text{total}} = w_1 R_{\text{outcome}} + w_2 R_{\text{backtrack}} + w_3 R_{\text{format}}$$

**Default Weights** (Stage 1 - Foundation):
- $w_1 = 0.5$ (outcome)
- $w_2 = 1.0$ (backtrack)
- $w_3 = 0.3$ (format)

### 4.2 Component 1: Outcome Accuracy

**Purpose**: Evaluate final answer correctness after applying all backtracks.

$$R_{\text{outcome}} = \begin{cases}
1.0 & \text{if } f(x_T) = y^* \text{ (exact match)} \\
0.5 \cdot \text{acc}(f(x_T), y^*) & \text{otherwise (partial credit)}
\end{cases}$$

where $f(x_T)$ is the final sequence after applying backtracks.

### 4.3 Component 2: Backtrack Efficiency

**Purpose**: Incentivize appropriate backtracking while penalizing misuse and "stuttering".

$$R_{\text{backtrack}} = \alpha \cdot \Delta_{\text{improve}} \cdot \mathbf{1}[\text{valid\_attempt}] + \frac{\gamma}{\sqrt{n_{\text{bt}}}} \cdot \mathbf{1}[\Delta > 0] - \beta \cdot n_{\text{bt}} \cdot \mathbf{1}[\text{unnecessary}] - \delta \cdot \mathbf{1}[\text{failed}]$$

where:
- $\Delta_{\text{improve}} = \text{acc}(x_{\text{final}}, y^*) - \text{acc}(x_{\text{init}}, y^*)$
- $\mathbf{1}[\text{valid\_attempt}]$ is a length constraint: $\text{len}(x_{\text{init}}) > 0.5 \cdot \text{len}(y^*)$
  - This prevents rewarding "stuttering" (early backtracks on short prefixes).
- $x_{\text{init}}$: Sequence before the *first* backtrack token.
- $n_{\text{bt}}$: number of backtrack tokens.

**Hyperparameters** (tuned for incentive compatibility):
- $\alpha = 1.0$: Bonus scale (matches Outcome weight).
- $\gamma = 0.25$: Efficiency decay.
- $\beta = 0.1$: Unnecessary penalty (mild deterrent).
- $\delta = 0.5$: Failed correction penalty (strict deterrent).

**Incentive-Compatibility Conditions**:
1. **Prevent Stuttering**: Backtracking on short prefixes yields $R < 0$ (via $\beta$ or $\delta$) because $\mathbf{1}[\text{valid}]$ is 0.
2. **Prevent "Always Backtrack"**: If $\Delta \le 0$, reward is negative ($-\beta n$ or $-\delta$).
3. **Prevent "Never Backtrack"**: If $\Delta > 0$ and attempt is valid, $R = \alpha\Delta + \gamma > 0$.

### 4.3 Component 3: Format Compliance

**Purpose**: Ensure outputs follow expected structure.

$$R_{\text{format}} = \mathbf{1}[\text{has\_boxed}] \cdot (0.5 + 0.5 \cdot \mathbf{1}[\text{valid\_format}])$$

---

## 5. Theoretical Justification

### 5.1 Policy Invariance Analysis

**Claim**: Our reward design can be decomposed into a base reward plus potential-based shaping.

Let $\Phi(s) = V^*(s)$ be the optimal value function of the base MDP (outcome-only reward). Then:

$$R_{\text{total}} = R_{\text{base}} + \underbrace{\sum_k w_k F_k}_{\text{shaping}}$$

For the shaping to be policy-invariant, each $F_k$ should satisfy:
$$F_k(s, a, s') = \gamma \Phi_k(s') - \Phi_k(s)$$

**Process Quality** can be written as potential-based if we define $\Phi_{\text{process}}(s)$ as the expected future step correctness from state $s$. However, we omit this component in our implementation as it requires ground-truth step labels.

**Format Compliance** is terminal-only and does not affect intermediate actions (trivially potential-based with $\Phi = 0$).

**Backtrack Efficiency** is more complex:
- The efficiency bonus $\gamma / \sqrt{n_{\text{bt}}}$ is NOT potential-based (depends on history)
- The correction bonus $\alpha \cdot \Delta$ approximates $\gamma V(s') - V(s)$ when improvement is used

**Conclusion**: Our reward is approximately potential-based for most components, but the efficiency bonus introduces some policy distortion. We accept this as a trade-off for better credit assignment.

### 5.2 Credit Assignment for Backtrack Actions

The backtrack decision at time $t$ is credited via:

$$\nabla_\theta \log \pi_\theta(\langle\text{BK}\rangle | s_t) \cdot A(s_t, \langle\text{BK}\rangle)$$

Our "improvement-based" bonus approximates the counterfactual advantage:

$$A(s_t, \langle\text{BK}\rangle) \approx Q(s_t, \langle\text{BK}\rangle) - V(s_t) \approx \Delta_{\text{improve}}$$

This provides immediate credit proportional to the improvement achieved by backtracking.

### 5.3 Convergence Guarantees

**Claim**: GRPO with our multi-component reward converges to a stationary point.

By standard policy gradient theory (Konda & Tsitsiklis, 2000), under:
1. Smooth policy parameterization
2. Appropriate learning rate schedule
3. Bounded rewards

GRPO converges to stationary points of the shaped objective. The KL regularization in GRPO provides trust-region-like stability (Schulman et al., 2015).

**Note**: We cannot guarantee convergence to the *globally* optimal policy of the base task unless all shaping is potential-based.

---

## 6. Curriculum Learning Strategy

### 6.1 Three-Stage Curriculum

| Stage | Progress | Outcome Weight | Backtrack Weight | Format Weight | Focus |
|-------|----------|----------------|------------------|---------------|-------|
| Foundation | 0-30% | 0.5 | 1.0 | 0.3 | Learn backtrack mechanics |
| Intermediate | 30-70% | 0.8 | 0.7 | 0.3 | Refine when to backtrack |
| Advanced | 70-100% | 1.0 | 0.5 | 0.3 | Optimize final accuracy |

### 6.2 Theoretical Justification

By Theorem 8 (CRL Performance Guarantee), curriculum learning reduces sample complexity when:
1. Tasks are decomposed appropriately (short→long, simple→complex)
2. Stage transitions are smooth (small Wasserstein distance)
3. Policies transfer between stages (Lipschitz reward continuity)

Our curriculum satisfies these conditions by:
- Starting with short sequences and obvious errors
- Gradually increasing sequence length and error subtlety
- Smoothly interpolating weights between stages

---

## 7. Anti-Hacking Mechanisms

### 7.1 Multi-Signal Redundancy

By using four reward components with different sensitivities, we make it harder to game any single metric:
- Gaming outcome alone → penalized by backtrack efficiency
- Gaming backtrack count → penalized by outcome accuracy
- Gaming format → minimal weight, cannot dominate

### 7.2 Hard Constraints

**Maximum Backtrack Limit**: $n_{\text{bt}} \leq 20$
- Violation: $R_{\text{backtrack}} = -1.0$ (severe penalty)
- Prevents "infinite backtrack" loops

**Minimum Improvement Threshold**: $\Delta_{\min} = 0.1$
- Backtrack bonus only applies if improvement exceeds threshold
- Prevents trivial micro-corrections

### 7.3 KL Regularization

GRPO includes KL penalty against reference policy:
$$\mathcal{L} = \mathbb{E}[R] - \beta \cdot D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$$

This prevents extreme exploitation of reward loopholes by keeping the policy close to a known-good baseline.

---

## 8. Implementation

### 8.1 Reward Function Class

```python
@dataclass
class BacktrackRewardFunction(BaseRewardFunction):
    # Component weights
    outcome_weight: float = 1.0
    process_weight: float = 0.7
    backtrack_weight: float = 0.6
    format_weight: float = 0.3
    
    # Backtrack sub-rewards
    correction_bonus: float = 0.4
    unnecessary_penalty: float = 0.2
    efficiency_weight: float = 0.25
    failed_correction_penalty: float = 0.3
    
    # Constraints
    max_backtracks: int = 20
```

### 8.2 Integration with GRPO

```yaml
# Training config
reward_funcs: "backtrack_grpo"
reward_weights: "1.0"

# Initial weights (Stage 1)
outcome_weight: 0.5
process_weight: 1.0
backtrack_weight: 1.0
format_weight: 0.5
```

---

## 9. Open Problems and Future Directions

### 9.1 Theoretical Gaps

1. **Non-Markov Efficiency Bonus**: The $\gamma / \sqrt{n_{\text{bt}}}$ term depends on history. Future work could reformulate as a potential-based term.

2. **Judge Robustness**: The improvement signal depends on the accuracy estimator. Adversarial exploitation is possible if the estimator is weak.

3. **Global Optimality**: We can only guarantee convergence to stationary points, not global optima.

### 9.2 Extensions

1. **Hierarchical Backtracking**: Different tokens for token-level, step-level, and full restart.

2. **Learned Potentials**: Train $\Phi(s)$ as a value network to provide principled shaping.

3. **Adaptive Weights**: Meta-learn component weights during training.

---

## 10. References

### Core Papers

1. Shao, Z., et al. (2024). "DeepSeekMath: Pushing the Limits of Mathematical Reasoning." arXiv:2402.03300.
2. Lightman, H., et al. (2024). "Let's Verify Step by Step." ICLR.
3. Kumar, A., et al. (2025). "Training Language Models to Self-Correct via Reinforcement Learning." ICLR (Oral).
4. Ng, A., Harada, D., Russell, S. (1999). "Policy Invariance Under Reward Transformations." ICML.

### Theoretical Foundations

5. Devlin, S., Kudenko, D. (2012). "Dynamic Potential-Based Reward Shaping." AAMAS.
6. Konda, V., Tsitsiklis, J. (2000). "Actor-Critic Algorithms." NeurIPS.
7. Schulman, J., et al. (2015). "Trust Region Policy Optimization." ICML.
8. "Lexicographic Multi-Objective Reinforcement Learning." IJCAI 2022.
9. "Finite-Time Convergence of Actor-Critic MORL." ICML 2024.

### Credit Assignment

10. Cao, et al. (2026). "SCAR: Shapley Credit Assignment for RLHF." ICLR.
11. Chai, et al. (2025). "MA-RLHF: Macro Actions for Credit Assignment." ICLR.
12. "Expected Eligibility Traces." AAAI 2021.

### Curriculum Learning

13. "Curriculum Reinforcement Learning from Easy to Hard." ICLR 2026.
14. "GRADIENT: Curriculum via Optimal Transport." NeurIPS 2022.
15. "Understanding Complexity Gains of Curriculum." ICML 2023.

### Reward Hacking

16. "Defining and Characterizing Reward Gaming." NeurIPS 2022.
17. "Correlated Proxies and Reward Hacking Mitigation." ICLR 2025.
18. "Calculus on MDPs: Potential Shaping as Gradient." arXiv:2208.09570.

---

*This document provides the theoretical foundation for reward function design in the Halluc project.*
