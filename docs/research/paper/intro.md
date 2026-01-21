# Introduction

Autoregressive large language models (LLMs) have demonstrated remarkable capabilities across a wide range of tasks, from code generation~\cite{chen2021evaluating} and question answering~\cite{kamalloo2023evaluating} to complex summarization~\cite{zhang2024benchmarking}. Fundamentally trained on the next-token prediction objective, these models generate text sequentially, appending each new token to the immutable history of previous generations~\cite{bachmann2024pitfalls}. While this forward-only generation paradigm is efficient, it imposes a critical cognitive limitation: the model must commit to a reasoning path step-by-step, without the ability to revise earlier decisions during the generation process.

This "forward-only" constraint renders LLMs particularly susceptible to **error propagation**, where a single incorrect token or logical leap early in the generation corrupts the subsequent context, leading to a cascade of flawed reasoning that the model cannot recover from~\cite{zhang2023snowball, chen2025rethinking}. Unlike human reasoning, which is inherently non-linear and iterative—often involving backtracking, revising assumptions, and correcting errors on the fly—standard LLMs lack a mechanism to "change their mind" once a token is emitted. This limitation is further evidenced by structural failures such as the "Reversal Curse," where models fail to generalize relationships in reverse directions due to the strict left-to-right causal attention mask~\cite{berglund2024reversal, li2025breaking}.

To address these limitations, recent research has explored methods to induce "System 2" or "slow thinking" capabilities in LLMs. Prompting strategies like Chain-of-Thought (CoT)~\cite{wei2022chain} encourage models to decompose problems, but they remain bound by linear generation. Search-based methods such as Tree of Thoughts (ToT)~\cite{yao2023tree} and Graph of Thoughts (GoT)~\cite{besta2024graph} introduce non-linear exploration by sampling multiple reasoning paths and using external heuristics to prune incorrect branches. While effective, these approaches rely on expensive external orchestration and high inference-time compute, making them impractical for latency-sensitive applications. Alternatively, post-hoc self-correction methods attempt to prompt the model to review and fix its own outputs. However, empirical studies have shown that LLMs often struggle to self-correct reasoning errors without external oracles, frequently failing to identify logical inconsistencies or "correcting" valid steps due to a lack of intrinsic verification capabilities~\cite{huang2023cannot, kamoi2024correct, zhang2025darkside}.

In this work, we propose **Intrinsic Backtracking**, a novel mechanism that internalizes the iterative refinement process directly into the autoregressive generation loop. Building on recent explorations of token-level editing~\cite{chen2024sequencematch, zhang2025stepback}, we train the model to generate a special `\backtrack` token that functionally deletes the preceding token, allowing the model to erase and revise its own reasoning on the fly. This effectively gives the model a "backspace key," enabling it to explore reasoning paths, detect potential errors, and correct them within a single generation pass without external intervention.

Our approach distinguishes itself from prior imitation-learning based attempts~\cite{chen2024sequencematch} by leveraging **Group Relative Policy Optimization (GRPO)**~\cite{shao2024deepseekmath} to discover optimal backtracking policies. We introduce a comprehensive training framework comprising:
1.  **Masked Supervised Fine-Tuning (SFT)** with data augmentation strategies based on sequence alignment~\cite{myers1986diff}, which teaches the model to recognize and correct errors without learning to generate them.
2.  **Multi-Component Reward Shaping**: Unlike simple outcome-based rewards, we design a dense reward function that balances outcome accuracy with **backtrack efficiency**. We introduce specific terms to incentivize successful corrections while penalizing "stuttering" (unnecessary backtracking) and failed correction attempts, ensuring the model learns an efficient self-correction policy rather than just memorizing error patterns.

We validate our method on complex reasoning benchmarks, demonstrating that intrinsic backtracking significantly improves reasoning accuracy and robustness compared to standard autoregressive baselines, while incurring a lower computational cost than external search methods.

## References

\bibliographystyle{icml2024}
\bibliography{references}

% References to be included in the .bib file:
% [chen2021evaluating] Chen, M., et al. (2021). Evaluating Large Language Models Trained on Code.
% [kamalloo2023evaluating] Kamalloo, E., et al. (2023). Evaluating Open-Domain Question Answering in the Era of Large Language Models.
% [zhang2024benchmarking] Zhang, T., et al. (2024). Benchmarking Large Language Models for Summarization.
% [bachmann2024pitfalls] Bachmann, G., & Nagarajan, V. (2024). The Pitfalls of Next-Token Prediction.
% [zhang2023snowball] Zhang, M., Press, O., Merrill, W., Liu, A., & Smith, N. A. (2023). How Language Model Hallucinations Can Snowball. arXiv preprint arXiv:2305.13534.
% [chen2025rethinking] Chen, et al. (2025). Rethinking External Slow-Thinking: From Snowball Errors to Probability of Correct Reasoning.
% [berglund2024reversal] Berglund, L., et al. (2024). The Reversal Curse: LLMs trained on "A is B" fail to learn "B is A". ICLR 2024.
% [li2025breaking] Li, et al. (2025). Breaking the Reversal Curse: How Masked Diffusion Models Achieve Reverse Inference.
% [wei2022chain] Wei, J., et al. (2022). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. NeurIPS 2022.
% [yao2023tree] Yao, S., et al. (2023). Tree of Thoughts: Deliberate Problem Solving with Large Language Models. NeurIPS 2023.
% [besta2024graph] Besta, M., et al. (2024). Graph of Thoughts: Solving Elaborate Problems with Large Language Models. AAAI 2024.
% [huang2023cannot] Huang, J., et al. (2023). Large Language Models Cannot Self-Correct Reasoning Yet. ICLR 2024.
% [kamoi2024correct] Kamoi, R., et al. (2024). When Can LLMs Actually Correct Their Own Mistakes? A Critical Survey. Transactions of the ACL.
% [zhang2025darkside] Zhang, et al. (2025). Understanding the Dark Side of LLMs' Intrinsic Self-Correction.
% [chen2024sequencematch] Chen, et al. (2024). SequenceMatch: Imitation Learning for Autoregressive Sequence Modeling with Backtracking. ICLR 2024.
% [zhang2025stepback] Zhang, et al. (2025). Step Back to Leap Forward: Self-Backtracking for Boosting Reasoning of Language Models.
% [shao2024deepseekmath] Shao, Z., et al. (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.
% [myers1986diff] Myers, E. W. (1986). An O(ND) difference algorithm and its variations. Algorithmica.
