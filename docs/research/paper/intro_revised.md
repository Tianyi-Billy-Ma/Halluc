Title: Non-Monotonic Autoregressive Sequence Model

\section{Introduction}

Autoregressive sequence models generate outputs by conditioning each token on all previously generated tokens, a paradigm that has achieved remarkable success across language modeling~\cite{brown2020language, touvron2023llama}, code generation~\cite{chen2021evaluating}, and beyond.
This sequential factorization, however, enforces a fundamental constraint: once a token is sampled, it becomes an immutable part of the conditioning context for all subsequent generation steps.
We call this the \textit{monotonicity constraint}---the generated sequence can only grow, never shrink.

This architectural rigidity creates a critical vulnerability to \textit{error propagation}.
When a model samples an erroneous or suboptimal token, that token permanently conditions all future predictions, potentially steering the generation trajectory into low-probability regions from which recovery is impossible~\cite{zhang2023snowball, bachmann2024pitfalls}.
Theoretical analysis formalizes this concern: under standard maximum likelihood training, the expected regret grows \textit{quadratically} with sequence length~\cite{ross2011reduction}, as small per-step errors compound into cascading failures.
This compounding effect is particularly pronounced in tasks requiring multi-step reasoning, where a single logical misstep early in generation can corrupt the entire downstream chain~\cite{gan2025rethinking, berglund2023reversal}.

Recent efforts to address error propagation have explored external orchestration mechanisms: prompting strategies that encourage deliberation~\cite{wei2022chain}, tree-structured search over candidate continuations~\cite{yao2023tree, besta2024graph}, and post-hoc verification with iterative refinement~\cite{madaan2023selfrefine, lightman2023let}.
While effective, these approaches share a common limitation---they operate \textit{outside} the core autoregressive loop, requiring either expensive inference-time computation or access to external verifiers.
Moreover, empirical studies reveal that models often struggle to self-correct without external oracles, frequently introducing new errors or undoing valid reasoning steps~\cite{huang2023large, kamoi2024can}.

In this work, we propose \textbf{\modelname}, a framework that breaks the monotonicity constraint by introducing a native \texttt{<UNDO>} token directly into the autoregressive vocabulary.
When generated, this token triggers a ``pop'' operation that removes the preceding token from the sequence, effectively allowing the model to prune divergent trajectories and explore alternative continuations within a single generation pass.
Unlike multi-turn refinement or external search, our approach internalizes self-correction as a first-class generative action, enabling efficient, on-the-fly error recovery without additional inference overhead.

Training a model to use the \texttt{<UNDO>} token effectively presents a subtle challenge: standard supervised fine-tuning on error-correction traces would paradoxically \textit{reinforce} the very errors the model should avoid---a phenomenon we term \textit{negative learning}.
To address this, we introduce \textbf{Masked Supervised Fine-Tuning (Masked SFT)}, where the loss is computed exclusively on backtracking and correction tokens while error tokens are masked.
We provide theoretical analysis showing that Masked SFT yields unbiased gradients for the target correction behavior while providing zero gradient signal for error generation.
Following the supervised stage, we apply \textbf{Group Relative Policy Optimization (GRPO)}~\cite{shao2024deepseekmath} with a multi-component reward function that balances outcome accuracy on the resolved sequence against backtracking efficiency, incentivizing the model to correct errors when beneficial while penalizing unnecessary or excessive undoing.

We validate \modelname on mathematical reasoning benchmarks, demonstrating substantial improvements over standard autoregressive baselines.
Comprehensive ablation studies reveal that each component of our training pipeline---sequence augmentation, masked loss computation, and the multi-objective reward design---contributes meaningfully to final performance, and we provide insights into the conditions under which non-monotonic generation offers the greatest advantage.

\paragraph{Contributions.} Our contributions are summarized as follows:
\begin{itemize}[noitemsep, topsep=-1pt, leftmargin=*]
\item \textbf{Non-Monotonic Autoregressive Framework.} We propose a novel sequence modeling paradigm that extends the standard autoregressive vocabulary with a functional \texttt{<UNDO>} token, enabling models to dynamically prune and revise their generation trajectories within a single forward pass.

\item \textbf{Masked SFT with Theoretical Guarantees.} We introduce Masked Supervised Fine-Tuning and prove that it provides unbiased gradients for learning error detection and correction while preventing the model from learning to generate errors---addressing the negative learning problem that undermines naive fine-tuning on correction traces.

\item \textbf{Comprehensive Ablation Studies.} We conduct extensive experiments analyzing the contribution of each pipeline component---data augmentation strategies, loss masking, reward function design, and curriculum learning---providing actionable insights into the design space of non-monotonic sequence models.

\item \textbf{State-of-the-Art Performance.} We demonstrate that \modelname achieves significant improvements over both standard autoregressive baselines and existing self-correction methods on complex reasoning benchmarks, while incurring lower inference-time computational cost than search-based alternatives.
\end{itemize}
