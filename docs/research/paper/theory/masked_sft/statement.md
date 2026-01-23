# Theoretical Analysis: Masked SFT vs. Standard SFT

**Author**: Research Analysis  
**Date**: January 2026  
**Project**: N-MAR: Non-Monotonic Autoregressive Sequence Modeling

---

## 1. Problem Statement

Autoregressive models generate sequences monotonically, meaning any sampled token becomes a permanent condition for future steps. This rigidity makes them susceptible to **error propagation**, where a single deviation causes the trajectory to drift irreversibly into low-probability regions.

We propose a **Non-Monotonic Autoregressive (N-MAR)** framework that empowers models to prune these divergent trajectories using a single special token, $\bk$ (denoted as `<UNDO>`).

We consider the task of training a language model on **non-monotonic traces**. A trace $\mathbf{y}$ is a sequence composed of four segments:
$$
\mathbf{y} = (\mathbf{p}, \mathbf{e}, \mathbf{b}, \mathbf{c})
$$
where:
- $\mathbf{p}$: Correct **Prefix**.
- $\mathbf{e}$: **Deviations** (sub-optimal or erroneous tokens to be pruned).
- $\mathbf{b}$: **Undo** tokens (specifically $\langle\text{UNDO}\rangle$).
- $\mathbf{c}$: **Correction** tokens (the optimal continuation).

We compare two training objectives:

1.  **Standard SFT ($\mathcal{L}_{\text{SFT}}$)**: Minimizes negative log-likelihood on the *entire* sequence.
2.  **Masked SFT ($\mathcal{L}_{\text{mSFT}}$)**: Minimizes negative log-likelihood on $\mathbf{p}, \mathbf{b}, \mathbf{c}$ only, masking the loss for $\mathbf{e}$.

---

## 2. Main Result: The Safety Gap Theorem

The core theoretical advantage of Masked SFT is not just that it learns to backtrack, but that it avoids the **Negative Learning** inherent in Standard SFT.

\begin{theorem}[Safety Gap between SFT and Masked SFT]\label{theorem: safety gap}
Let $\mathcal{L}_{\text{SFT}}$ and $\mathcal{L}_{\text{mSFT}}$ be the standard and masked SFT objectives respectively. Let $\mathbf{g}_{\text{SFT}}$ and $\mathbf{g}_{\text{mSFT}}$ be their expected gradient updates. 

The learning dynamics satisfy the following properties:

\begin{enumerate}[label=(\alph*), leftmargin=*, noitemsep]
    \item \textbf{Common Learning (Detection & Correction)}: Both objectives drive the model to generate backtrack and correction tokens given error contexts. The gradients for these behaviors are identical:
    \begin{equation}
        \frac{\partial \mathcal{L}_{\text{SFT}}}{\partial \theta_{\mathbf{b},\mathbf{c}}} = \frac{\partial \mathcal{L}_{\text{mSFT}}}{\partial \theta_{\mathbf{b},\mathbf{c}}}
    \end{equation}
    where $\theta_{\mathbf{b},\mathbf{c}}$ denotes parameters influencing the likelihood of $\mathbf{b}$ and $\mathbf{c}$.

    \item \textbf{Negative Learning in SFT}: Standard SFT explicitly maximizes the probability of generating errors. The gradient difference is non-zero and points in the direction of error generation:
    \begin{equation}
        \mathbf{g}_{\text{SFT}} - \mathbf{g}_{\text{mSFT}} = - \mathbb{E}_{\mathbf{y}} \left[ \sum_{y_t \in \mathbf{e}} \nabla_\theta \log p_\theta(y_t \mid y_{<t}) \right]
    \end{equation}

    \item \textbf{Safety Divergence}: Over $T$ training steps, the probability of spontaneously generating deviations (errors) satisfies:
    \begin{equation}
        p_{\theta_{\text{SFT}}}(\mathbf{e} \mid \mathbf{p}) \gg p_{\theta_{\text{mSFT}}}(\mathbf{e} \mid \mathbf{p})
    \end{equation}
    assuming the deviation generation direction is not orthogonal to the correction direction.
\end{enumerate}
\end{theorem}

This theorem establishes that while both methods teach the model *how to recover*, Standard SFT simultaneously teaches the model *to deviate*, whereas Masked SFT does not.
