# Mathematical Preliminaries: Autoregressive Language Models and Supervised Fine-Tuning

**Author**: Research Analysis  
**Date**: January 2026  
**Project**: Halluc - LLM Backtracking via Reinforcement Learning

---

## 1. Notation

Throughout this document, we use the following notation:

| Symbol | Description |
|--------|-------------|
| $\mathcal{V}$ | Vocabulary (set of all tokens), $\|\mathcal{V}\| = V$ |
| $x = (x_1, \ldots, x_n)$ | Input prompt sequence of $n$ tokens |
| $y = (y_1, \ldots, y_T)$ | Output/completion sequence of $T$ tokens |
| $y_{<t}$ | Prefix $(y_1, \ldots, y_{t-1})$ |
| $y_{\leq t}$ | Prefix including $t$: $(y_1, \ldots, y_t)$ |
| $\theta$ | Model parameters |
| $\pi_\theta$ | Policy (language model) parameterized by $\theta$ |
| $p_\theta(\cdot)$ | Probability distribution under model $\theta$ |
| $\mathcal{D}$ | Training dataset |
| $\mathbb{E}[\cdot]$ | Expectation |

---

## 2. Autoregressive Language Models

### 2.1 Definition

An **autoregressive language model** defines a probability distribution over sequences by factorizing the joint probability using the chain rule of probability:

**Definition 1 (Autoregressive Model)**: An autoregressive model $p_\theta$ is characterized by factorizing the joint distribution over a sequence $y = (y_1, y_2, \ldots, y_T)$ as:

$$p_\theta(y | x) = \prod_{t=1}^{T} p_\theta(y_t | y_{<t}, x)$$

where each factor $p_\theta(y_t | y_{<t}, x)$ represents the probability of element $y_t$ conditioned on all preceding elements $y_{<t} = (y_1, \ldots, y_{t-1})$ and optional context $x$.

**Remark**: The autoregressive property ensures that generation proceeds sequentially, with each element depending only on previously generated elements (causal dependency). This follows directly from the chain rule of probability.

### 2.2 Policy Notation

In the reinforcement learning literature, the language model is often denoted as a **policy** $\pi_\theta$:

$$\pi_\theta(y_t | s_t) = p_\theta(y_t | y_{<t}, x)$$

where the **state** $s_t = (x, y_{<t})$ encapsulates the prompt and all previously generated tokens. This framing is useful when applying RL algorithms to language model training.

### 2.3 Output Distribution

At each generation step $t$, the model produces a probability distribution over the vocabulary $\mathcal{V}$:

**Definition 2 (Next-Token Distribution)**: The conditional distribution over the next token is:

$$p_\theta(y_t = v | y_{<t}, x) = \frac{\exp(z_v / \tau)}{\sum_{v' \in \mathcal{V}} \exp(z_{v'} / \tau)}$$

where:
- $z = (z_1, \ldots, z_V) \in \mathbb{R}^V$ are the **logits** (unnormalized log-probabilities)
- $\tau > 0$ is the **temperature** parameter
- The softmax function converts logits to a valid probability distribution

**Logit Computation**: For transformer-based models:

$$z = W_{\text{head}} \cdot h_t + b$$

where $h_t \in \mathbb{R}^d$ is the final hidden state at position $t$, and $W_{\text{head}} \in \mathbb{R}^{V \times d}$ is the output projection (often called the "language model head").

### 2.4 Temperature Scaling

The temperature parameter $\tau$ controls the sharpness of the output distribution:

$$p_\theta^\tau(y_t = v | \cdot) = \frac{\exp(z_v / \tau)}{\sum_{v'} \exp(z_{v'} / \tau)}$$

| Temperature | Effect |
|-------------|--------|
| $\tau \to 0^+$ | Deterministic (argmax), selects highest probability token |
| $\tau = 1$ | Standard softmax, unmodified distribution |
| $\tau > 1$ | Flatter distribution, more randomness/diversity |
| $\tau \to \infty$ | Uniform distribution over vocabulary |

**Proposition 1**: As $\tau \to 0^+$, the distribution converges to a point mass on the maximum logit:

$$\lim_{\tau \to 0^+} p_\theta^\tau(y_t = v | \cdot) = \begin{cases} 1 & \text{if } v = \arg\max_{v'} z_{v'} \\ 0 & \text{otherwise} \end{cases}$$

---

## 3. Transformer Architecture Fundamentals

### 3.1 Self-Attention Mechanism

The core computation in transformers is **scaled dot-product attention**:

**Definition 3 (Scaled Dot-Product Attention)**: Given query $Q$, key $K$, and value $V$ matrices:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

where:
- $Q \in \mathbb{R}^{n \times d_k}$ (queries)
- $K \in \mathbb{R}^{m \times d_k}$ (keys)
- $V \in \mathbb{R}^{m \times d_v}$ (values)
- $d_k$ is the key dimension (scaling factor prevents gradient vanishing)

For a sequence of hidden states $H = [h_1, \ldots, h_n]^\top \in \mathbb{R}^{n \times d}$:

$$Q = HW_Q, \quad K = HW_K, \quad V = HW_V$$

where $W_Q, W_K \in \mathbb{R}^{d \times d_k}$ and $W_V \in \mathbb{R}^{d \times d_v}$ are learned projections.

### 3.2 Multi-Head Attention

**Definition 4 (Multi-Head Attention)**: Multiple attention heads capture different aspects of relationships:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W_O$$

where each head is:

$$\text{head}_i = \text{Attention}(QW_Q^i, KW_K^i, VW_V^i)$$

and $W_O \in \mathbb{R}^{hd_v \times d}$ is the output projection. Typically $d_k = d_v = d/h$.

### 3.3 Causal (Autoregressive) Masking

For autoregressive generation, we must ensure position $t$ cannot attend to future positions $t' > t$:

**Definition 5 (Causal Mask)**: The causal attention mask $M \in \{0, -\infty\}^{n \times n}$ is:

$$M_{ij} = \begin{cases} 0 & \text{if } i \geq j \\ -\infty & \text{if } i < j \end{cases}$$

The masked attention becomes:

$$\text{CausalAttention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} + M\right) V$$

Adding $-\infty$ to future positions ensures $\text{softmax}$ assigns them zero weight.

### 3.4 Position Encoding

Since self-attention is permutation-invariant, positional information must be explicitly injected:

**Sinusoidal Encoding** (Vaswani et al., 2017):

$$\text{PE}_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right), \quad \text{PE}_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

**Rotary Position Embedding (RoPE)** (Su et al., 2021): Encodes relative positions by rotating query and key vectors:

$$\tilde{q}_m = R_{\Theta, m} q_m, \quad \tilde{k}_n = R_{\Theta, n} k_n$$

where $R_{\Theta, m}$ is a rotation matrix determined by position $m$.

### 3.5 Transformer Block

A single transformer layer consists of:

$$h' = h + \text{MultiHead}(\text{LN}(h))$$
$$h'' = h' + \text{FFN}(\text{LN}(h'))$$

where:
- $\text{LN}(\cdot)$ is layer normalization
- $\text{FFN}(\cdot)$ is a feed-forward network: $\text{FFN}(x) = \sigma(xW_1 + b_1)W_2 + b_2$
- $\sigma$ is typically GELU or SiLU activation

---

## 4. Decoding Strategies

### 4.1 Greedy Decoding

Select the most probable token at each step:

$$y_t = \arg\max_{v \in \mathcal{V}} p_\theta(v | y_{<t}, x)$$

**Properties**:
- Deterministic
- Fast (single forward pass per token)
- May miss globally optimal sequences

### 4.2 Beam Search

Maintain $B$ candidate sequences (beams) and expand each:

**Algorithm**: Let $\mathcal{H}_t$ be the set of $B$ partial hypotheses at step $t$. Each hypothesis $h \in \mathcal{H}_t$ has a prefix $y_{<t}$ and a score:

$$\text{score}(h) = \sum_{\tau=1}^{t-1} \log p_\theta(y_\tau | y_{<\tau}, x) + \lambda \cdot \text{len}(y_{<t})$$

where $\lambda$ is a **length normalization** factor to prevent bias toward shorter sequences.

At step $t$, for each hypothesis $h \in \mathcal{H}_t$, extend by all tokens $v \in \mathcal{V}$:

$$\mathcal{H}'_{t+1} = \bigcup_{h \in \mathcal{H}_t} \{(h \cdot v, \text{score}(h) + \log p_\theta(v | h_{\text{prefix}}, x)) \mid v \in \mathcal{V}\}$$

Then select the top $B$ hypotheses:

$$\mathcal{H}_{t+1} = \arg\max_{S \subset \mathcal{H}'_{t+1}, |S|=B} \sum_{h \in S} \text{score}(h)$$

**Properties**:
- Explores multiple hypotheses
- Computational cost: $O(B \times V)$ per step
- Tends toward high-probability but potentially repetitive outputs (Holtzman et al., 2020)

### 4.3 Sampling with Temperature

Sample from the temperature-scaled distribution:

$$y_t \sim p_\theta^\tau(\cdot | y_{<t}, x)$$

### 4.4 Top-$k$ Sampling

Restrict sampling to the $k$ most probable tokens:

**Definition 6 (Top-$k$ Sampling)**: Let $\mathcal{V}^{(k)} \subset \mathcal{V}$ be the set of $k$ tokens with highest probability. The truncated distribution is:

$$p_\theta^{(k)}(v | \cdot) = \begin{cases} \frac{p_\theta(v | \cdot)}{\sum_{v' \in \mathcal{V}^{(k)}} p_\theta(v' | \cdot)} & \text{if } v \in \mathcal{V}^{(k)} \\ 0 & \text{otherwise} \end{cases}$$

### 4.5 Top-$p$ (Nucleus) Sampling

Restrict to the smallest set whose cumulative probability exceeds $p$:

**Definition 7 (Nucleus Sampling)**: Let $\mathcal{V}^{(p)}$ be the smallest set such that:

$$\sum_{v \in \mathcal{V}^{(p)}} p_\theta(v | \cdot) \geq p$$

where tokens are added in decreasing probability order. Sample from the renormalized distribution over $\mathcal{V}^{(p)}$.

**Advantage**: Adaptively adjusts the candidate set size based on the distribution's entropy.

---

## 5. Supervised Fine-Tuning (SFT)

### 5.1 Problem Setup

Given:
- A pre-trained language model with parameters $\theta_0$
- A dataset $\mathcal{D} = \{(x^{(i)}, y^{(i)})\}_{i=1}^N$ of prompt-completion pairs

**Goal**: Adapt the model to generate completions $y$ given prompts $x$ by minimizing a supervised loss.

### 5.2 Maximum Likelihood Estimation (MLE) Objective

**Definition 8 (SFT Loss)**: The standard SFT objective is the negative log-likelihood:

$$\mathcal{L}_{\text{SFT}}(\theta) = -\mathbb{E}_{(x, y) \sim \mathcal{D}}\left[\log p_\theta(y | x)\right]$$

Expanding using the autoregressive factorization:

$$\mathcal{L}_{\text{SFT}}(\theta) = -\mathbb{E}_{(x, y) \sim \mathcal{D}}\left[\sum_{t=1}^{|y|} \log p_\theta(y_t | y_{<t}, x)\right]$$

**Equivalently** (per-token formulation):

$$\mathcal{L}_{\text{SFT}}(\theta) = -\frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{|y^{(i)}|} \log p_\theta(y_t^{(i)} | y_{<t}^{(i)}, x^{(i)})$$

### 5.3 Relationship to Cross-Entropy Loss

The per-token loss is equivalent to the cross-entropy between the one-hot target distribution and the model's predicted distribution:

$$\mathcal{L}_t = -\sum_{v \in \mathcal{V}} \mathbb{1}[v = y_t] \cdot \log p_\theta(v | y_{<t}, x) = -\log p_\theta(y_t | y_{<t}, x)$$

**Proposition 2**: Minimizing $\mathcal{L}_{\text{SFT}}$ is equivalent to minimizing the KL divergence between the empirical data distribution and the model distribution:

$$\mathcal{L}_{\text{SFT}}(\theta) = D_{\text{KL}}(p_{\text{data}} \| p_\theta) + H(p_{\text{data}})$$

where $H(p_{\text{data}})$ is the entropy of the data distribution (constant w.r.t. $\theta$).

### 5.4 Loss Masking for Prompt Tokens

In instruction-tuning, we typically only compute loss on the completion tokens, not the prompt:

**Definition 9 (Completion-Only Loss)**: Let $|x|$ denote the prompt length. The masked loss is:

$$\mathcal{L}_{\text{masked}}(\theta) = -\sum_{t=|x|+1}^{|x|+|y|} \log p_\theta(y_{t-|x|} | x, y_{<t-|x|})$$

**Implementation**: Set labels for prompt positions to $-100$ (ignore index):

$$\ell_t = \begin{cases} -100 & \text{if } t \leq |x| \text{ (prompt token)} \\ y_{t-|x|} & \text{if } t > |x| \text{ (completion token)} \end{cases}$$

### 5.5 Gradient Computation

The gradient of the SFT loss with respect to parameters $\theta$:

$$\nabla_\theta \mathcal{L}_{\text{SFT}} = -\mathbb{E}_{(x,y) \sim \mathcal{D}}\left[\sum_{t=1}^{|y|} \nabla_\theta \log p_\theta(y_t | y_{<t}, x)\right]$$

For a single token with logits $z = f_\theta(y_{<t}, x)$, the gradient with respect to logits is:

$$\frac{\partial \mathcal{L}_t}{\partial z_v} = p_\theta(v | y_{<t}, x) - \mathbb{1}[v = y_t]$$

This is the difference between the predicted probability and the target (one-hot).

### 5.6 Causal Language Modeling

SFT on autoregressive models is a form of **causal language modeling** (CLM):

**Definition 10 (Causal Language Modeling)**: Given a sequence $(w_1, w_2, \ldots, w_n)$, predict each token from its left context:

$$\mathcal{L}_{\text{CLM}} = -\sum_{t=1}^{n} \log p_\theta(w_t | w_1, \ldots, w_{t-1})$$

**Contrast with Masked Language Modeling** (BERT-style):
- CLM: Predict $w_t$ given $w_{<t}$ (left context only)
- MLM: Predict masked $w_t$ given $w_{\neq t}$ (bidirectional context)

CLM is required for generation since we cannot access future tokens at inference time.

### 5.7 Training Considerations

**Learning Rate**: Typically lower than pre-training (e.g., $10^{-5}$ to $10^{-6}$) to avoid catastrophic forgetting.

**Sequence Packing**: Multiple short sequences can be packed into a single training example with attention masks preventing cross-sequence attention:

$$[\text{seq}_1, \text{EOS}, \text{seq}_2, \text{EOS}, \ldots, \text{seq}_k, \text{EOS}, \text{PAD}, \ldots]$$

**Gradient Accumulation**: Simulate larger batch sizes by accumulating gradients over multiple forward passes:

$$\theta \leftarrow \theta - \eta \cdot \frac{1}{K} \sum_{k=1}^{K} \nabla_\theta \mathcal{L}^{(k)}$$

### 5.8 Optimizer and Learning Rate Schedule

**AdamW Optimizer** (Loshchilov & Hutter, 2019): The standard optimizer for fine-tuning LLMs:

$$\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \\
\hat{m}_t &= \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t} \\
\theta_t &= \theta_{t-1} - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_{t-1} \right)
\end{aligned}$$

where $g_t = \nabla_\theta \mathcal{L}(\theta_{t-1})$ and $\lambda$ is the weight decay coefficient.

**Cosine Annealing with Warmup**: For warmup steps $W$ and total steps $T$:

$$\eta_t = \begin{cases}
\eta_0 \cdot \frac{t}{W} & \text{if } t \leq W \\
\eta_0 \cdot \frac{1}{2}\left(1 + \cos\left(\pi \frac{t - W}{T - W}\right)\right) & \text{if } t > W
\end{cases}$$

where $\eta_0$ is the peak learning rate.

---

## 6. Generation Process

### 6.1 Autoregressive Decoding Algorithm

Given a trained model $\pi_\theta$ and prompt $x$:

```
Algorithm: Autoregressive Generation
Input: prompt x, model π_θ, max_length T, decoding_strategy
Output: completion y = (y_1, ..., y_T')

1. Initialize: y ← []
2. For t = 1, 2, ..., T:
   a. Compute logits: z_t = f_θ(x, y_{<t})
   b. Apply decoding_strategy to z_t → select y_t
   c. Append y_t to y
   d. If y_t = EOS: break
3. Return y
```

### 6.2 KV-Cache for Efficient Generation

During generation, recomputing attention over all previous tokens is redundant. The **KV-cache** stores key and value projections:

$$K_{\text{cache}} = [K_1, K_2, \ldots, K_{t-1}], \quad V_{\text{cache}} = [V_1, V_2, \ldots, V_{t-1}]$$

At step $t$, only compute $K_t, V_t$ for the new token and concatenate with cache:

$$K_t^{\text{full}} = [K_{\text{cache}}; K_t], \quad V_t^{\text{full}} = [V_{\text{cache}}; V_t]$$

**Complexity reduction**: From $O(t^2)$ per token to $O(t)$ per token.

---

## 7. Exposure Bias and Compounding Errors

### 7.1 The Exposure Bias Problem

**Definition 11 (Exposure Bias)**: The discrepancy between training (conditioning on ground-truth $y^*_{<t}$) and inference (conditioning on model predictions $\hat{y}_{<t}$):

- **Training**: $p_\theta(y_t | y^*_{<t}, x)$ — always sees correct history
- **Inference**: $p_\theta(y_t | \hat{y}_{<t}, x)$ — may see erroneous history

### 7.2 Compounding Error Bound

**Theorem (Ross & Bagnell, 2010)**: Under behavior cloning (SFT), the expected regret grows quadratically with sequence length:

$$\mathbb{E}[\text{Regret}] = O(T^2 \epsilon)$$

where $T$ is sequence length and $\epsilon$ is the per-step error rate.

**Implication**: Small errors early in generation propagate and amplify, pushing the model into out-of-distribution states.

### 7.3 Motivation for Backtracking

This compounding error problem motivates mechanisms for **self-correction**. Rather than preventing all errors (impossible), we can train models to:
1. **Detect** when an error has occurred
2. **Backtrack** to remove erroneous tokens
3. **Correct** by generating the right continuation

This reduces the effective error propagation horizon, improving generation quality.

---

## References

1. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention Is All You Need. *NeurIPS*.

2. Radford, A., Narasimhan, K., Salimans, T., & Sutskever, I. (2018). Improving Language Understanding by Generative Pre-Training. *OpenAI Technical Report* (GPT-1).

3. Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). Language Models are Unsupervised Multitask Learners. *OpenAI Technical Report* (GPT-2).

4. Brown, T., Mann, B., Ryder, N., et al. (2020). Language Models are Few-Shot Learners. *NeurIPS* (GPT-3).

5. Holtzman, A., Buys, J., Du, L., Forbes, M., & Choi, Y. (2020). The Curious Case of Neural Text Degeneration. *ICLR* (Nucleus Sampling).

6. Fan, A., Lewis, M., & Dauphin, Y. (2018). Hierarchical Neural Story Generation. *ACL* (Top-k Sampling).

7. Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., & Liu, Y. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding. *arXiv:2104.09864* (RoPE).

8. Ross, S., & Bagnell, D. (2010). Efficient Reductions for Imitation Learning. *AISTATS*.

9. Ouyang, L., Wu, J., Jiang, X., et al. (2022). Training Language Models to Follow Instructions with Human Feedback. *NeurIPS* (InstructGPT).

10. Wei, J., Bosma, M., Zhao, V., et al. (2022). Finetuned Language Models are Zero-Shot Learners. *ICLR* (FLAN).

11. Chung, H. W., Hou, L., Longpre, S., et al. (2022). Scaling Instruction-Finetuned Language Models. *arXiv:2210.11416* (FLAN-T5/Flan-PaLM).

12. Bengio, S., Vinyals, O., Jaitly, N., & Shazeer, N. (2015). Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks. *NeurIPS*.

13. Loshchilov, I., & Hutter, F. (2019). Decoupled Weight Decay Regularization. *ICLR* (AdamW).

14. Zekri, A., et al. (2024). Large Language Models as Markov Chains. *arXiv* (Formal treatment of LLMs as Markov processes).

---

*This document provides the mathematical foundations for understanding autoregressive language models and supervised fine-tuning, serving as preliminaries for the Halluc backtracking research.*
