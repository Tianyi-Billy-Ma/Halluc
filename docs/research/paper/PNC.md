# Problem, Novelty, Contribution (P/N/C) Analysis

## 1. Problem (The "Why")
Standard autoregressive sequence modeling assumes a monotonic generation process where $P(x_t | x_{<t})$ is fixed once sampled. This rigidity makes models susceptible to **irreversible error propagation**, where a single deviation from the optimal manifold conditions all subsequent tokens, leading to covariate shift.
- **The Issue**: Models lack a native mechanism to "undo" local sampling errors.
- **Existing Limitations**:
    - **Standard SFT**: Trains models to mimic monotonic ground truth, offering no guidance on how to recover from inevitable inference-time errors.
    - **Imitation Learning**: Often relies on static datasets or simplistic error models, failing to generalize to complex reasoning failures.

## 2. Novelty (The "New Idea")
We introduce **Autoregressive Sequence Modeling with Undo**, a framework where generation is treated as a revisable process.
- **Mechanism**: A native `<|BACKTRACK|>` token that functionally removes the preceding token from the context window.
- **Training Pipeline**:
    1.  **Sequence Augmentation over True Error Distribution**: We approximate the error distribution by injecting noise into correct trajectories, creating a training set of `(prefix, error, backtrack, correction)` sequences.
    2.  **Masked SFT**: We apply a novel masking strategy during SFT. The loss is computed *only* on the backtrack tokens and the subsequent corrections, explicitly masking the error tokens. This ensures the model learns the *policy of correction* without maximizing likelihood on the *errors themselves*.
    3.  **GRPO with Parsimonious Correction Reward**: We further refine the policy using Group Relative Policy Optimization with a multi-objective reward that:
        - Rewards **Outcome Accuracy** on the resolved sequence.
        - Penalizes **Abundant/Unnecessary Backtracking** to ensure efficiency.

## 3. Contributions (The "Value")
1.  **Mechanism**: We propose a novel autoregressive mechanism that enables models to generate and dynamically revise their outputs during the generation stage.
2.  **Methodology**: We introduce a comprehensive training pipeline featuring **Masked SFT**—which we show theoretically and empirically outperforms standard SFT by focusing gradients on the correction policy—and **GRPO** for optimizing the trade-off between accuracy and efficiency.
3.  **Insights**: Through comprehensive ablation studies, we validate that the "Undo" operator allows models to significantly reduce error propagation and that our masked training objective is crucial for learning robust self-correction.
