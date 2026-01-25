TItle: Non-Monotonic Autoregressive Sequence Model

\section{Introduction}
Autoregressive models, such as series of GPT models, have shown capabilities to perform various downstream tasks, from code generation~\cite{chen2021evaluating}, question-answering~\cite{kamalloo2023evaluating}, to complex summarizations~\cite{zhang2024benchmarking, tang2023evaluating}.
Fundamentally trained on the next-token prediction objective, these models generate text in monotonic manners, where a new token is appended to the immutable sequence~\cite {bachmann2024pitfalls, thoppilan2022lamda}.
While this monotonic generation method has become a fundamental paradigm, it is susceptible to error propagation, where a single incorrect token or logical leap early in the generation corrupts the subsequent context, leading to a cascade of flawed reasoning that the model cannot recover from~\cite{zhang2023snowball, gan2025rethinking, berglund2023reversal, zhu2024towards}.

To address these limitations, recent research has explored methods to induce slow thinking capabilities in LLMs~\cite{lin2023swiftsage}, such as prompting strategies~\cite{wei2022chain}, search-based methods~\cite{yao2023tree, besta2024graph}, and post-hoc verifications~\cite{madaan2023selfrefine, lightman2023let, wang2022self}.
While effective, these approaches rely on expensive external orchestration or high inference-time computation, making them impractical in real-world tasks.
Moreover, empirical studies have shown that LLMs often struggle to self-correct without external oracles, frequently hallucinating new errors or correcting valid answers due to a lack of intrinsic verification capabilities~\cite{huang2023large, kamalloo2023evaluating, kamoi2024can}.

In this work, we propose a novel autoregressive mechanism, called \modelname, that internalizes the iterative refinement process directly into the autoregressive generation loop.
Specifically, instead of relying on external mechanisms or post-hoc critique, we train the model with an erase action, which serves as a token \bk in generation, allowing the model to refine the previous generated token on the fly.
This effectively gives the model ablilty to explore reasoning paths, detect potential errors, and correct them within a single generation pass.
To train model samples in a non-monotonic manner, we introduce a training framework that incorporates a masked supervised fine-tuning stage based on sequence alignment among true error distributions that aims to enable the model recognize and correct errors without learning them.
Afterward, we continue training the model via group relative policy optimization (GRPO)~\cite{shao2024deepseekmath} with a carefully designed reward function to incentivize the model to correct revisions while penalizing stuttering, i.e., unnecessary backtracking.
We validate our method on complex reasoning benchmarks, demonstrating that \modelname significantly improves reasoning accuracy compared to standard autoregressive baselines, while incurring a lower computational cost.

Our contributions are summarized as follows:
\begin{itemize}[noitemsep, topsep=-1pt, leftmargin=*]
\item We
\item \textbf{}
\end{itemize}
