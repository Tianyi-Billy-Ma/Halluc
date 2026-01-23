# Research Notes: Peer Review Critique & Revisions

**Date**: 2026-01-23
**Reviewer**: Senior Area Chair (Simulated)
**Topic**: Abstract & P/N/C Rigor

## Summary of Critique
The initial draft was methodologically sound but rhetorically imprecise.
1.  **Overclaim**: "Autonomous Detection" implies a discriminator/monitor.
    *   *Reality*: It's a learned probabilistic policy.
    *   *Fix*: Changed to "internalize a correction policy".
2.  **Weakness**: "Synthetic Error Bootstrapping" using random noise is insufficient for semantic correction.
    *   *Reality*: Random noise teaches the *mechanism* (syntax), RL teaches the *policy* (semantics).
    *   *Fix*: Clarified that synthetic data is a "warm-start" or "mechanical priming" step, not the full solution.
3.  **Terminology**: "Efficiency-Aware Reward" is generic.
    *   *Fix*: Renamed to "Parsimonious Correction Reward" to reflect the specific penalty on unnecessary retractions.

## Revisions (Iteration 2 - "No Hallucination")
*   **Narrative Pivot**: Shifted focus from "Hallucination" (a symptom) to "Error Propagation" and "Monotonicity Constraints" (the root cause).
*   **Framing**: The paper is now about **Distributional Robustness**. The "Undo" token is a mechanism to recover from OOD sampling.

## Revisions (Iteration 3 - "Masked SFT & Pipeline Focus")
*   **Mechanism**: Explicitly defined as "Autoregressive Sequence Modeling with Undo".
*   **Training Pipeline**: 
    *   Added **Masked SFT** as a core contribution (preventing the learning of error generation).
    *   Clarified the GRPO reward structure (Accuracy + Efficiency Penalty).
*   **Theoretical Claim**: Masked SFT > Standard SFT for this task.
*   **Insights**: Added focus on "comprehensive ablation studies" validating the pipeline.

## Future Defense Strategy (for Paper)
*   **Masked SFT**: Emphasize that standard SFT on `(error -> backtrack)` sequences effectively doubles the gradient on the error tokens (once when generating, once when seeing it in history). Masking is essential to break this symmetry.
