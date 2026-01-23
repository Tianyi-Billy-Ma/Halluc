# Proof of Safety Gap Theorem

**Theorem Reference**: Theorem \ref{theorem: safety gap} (Safety Gap between SFT and Masked SFT)

## Preliminaries

Let $\mathcal{D}$ be the distribution of backtrack traces. Each trace is a sequence $\mathbf{y} = (\mathbf{p}, \mathbf{e}, \mathbf{b}, \mathbf{c})$ consisting of:
- Prefix $\mathbf{p} = (y_1, \dots, y_{T_p})$
- Errors $\mathbf{e} = (y_{T_p+1}, \dots, y_{T_p+T_e})$
- Backtracks $\mathbf{b} = (y_{T_p+T_e+1}, \dots, y_{T_p+T_e+T_b})$ where $y_t = \langle\text{UNDO}\rangle$.
- Corrections $\mathbf{c} = (y_{T_p+T_e+T_b+1}, \dots, y_T)$

We define the set of error indices as $\mathcal{I}_{\text{error}} = \{T_p+1, \dots, T_p+T_e\}$.
The set of target indices (prefix, backtrack, correction) is $\mathcal{I}_{\text{target}} = \{1, \dots, T\} \setminus \mathcal{I}_{\text{error}}$.

The two objectives are:
$$
\mathcal{L}_{\text{SFT}}(\theta) = - \mathbb{E}_{\mathbf{y}} \left[ \sum_{t=1}^T \log p_\theta(y_t \mid y_{<t}) \right]
$$
$$
\mathcal{L}_{\text{mSFT}}(\theta) = - \mathbb{E}_{\mathbf{y}} \left[ \sum_{t \in \mathcal{I}_{\text{target}}} \log p_\theta(y_t \mid y_{<t}) \right]
$$

---

## Proof

We address the three claims of the theorem.

### Part (a): Common Learning (Detection & Correction)

**Statement**: Both objectives drive the model to generate backtrack and correction tokens given error contexts.

**Proof**:
Consider the partial derivative of the loss with respect to parameters $\theta_{\mathbf{b},\mathbf{c}}$ that specifically influence the generation of backtracks and corrections (e.g., the embeddings for $\langle\text{UNDO}\rangle$ or the correct tokens).

For $\mathcal{L}_{\text{SFT}}$, the sum runs over all $t$. We can split it:
$$
\mathcal{L}_{\text{SFT}} = \underbrace{- \sum_{t \in \mathcal{I}_{\text{target}}} \log p_\theta(y_t \mid y_{<t})}_{\mathcal{L}_{\text{mSFT}}} \underbrace{- \sum_{t \in \mathcal{I}_{\text{error}}} \log p_\theta(y_t \mid y_{<t})}_{\mathcal{L}_{\text{error}}}
$$

Taking the gradient with respect to the likelihood of a target token $y_k$ (where $k \in \mathcal{I}_{\text{target}}$):
$$
\nabla_{\theta} \mathcal{L}_{\text{SFT}} = \nabla_{\theta} \mathcal{L}_{\text{mSFT}} + \nabla_{\theta} \mathcal{L}_{\text{error}}
$$

If we assume the parameters controlling specific target tokens are largely independent of those controlling error tokens (e.g., different rows in the output embedding matrix), or if we simply look at the *contribution* from the target tokens themselves:
$$
\frac{\partial \mathcal{L}_{\text{SFT}}}{\partial \log p(y_k \mid \cdot)} = -1 = \frac{\partial \mathcal{L}_{\text{mSFT}}}{\partial \log p(y_k \mid \cdot)}
$$
Thus, both objectives provide the exact same gradient signal for learning to generate $\mathbf{b}$ and $\mathbf{c}$ given the context. The model learns $P(\langle\text{UNDO}\rangle \mid \mathbf{p}, \mathbf{e})$ equally well under both regimes. \qed

---

### Part (b): Negative Learning in SFT

**Statement**: The gradient difference points in the direction of error generation.

**Proof**:
From the decomposition in Part (a):
$$
\nabla_\theta \mathcal{L}_{\text{SFT}} = \nabla_\theta \mathcal{L}_{\text{mSFT}} + \nabla_\theta \mathcal{L}_{\text{error}}
$$
where $\mathcal{L}_{\text{error}} = - \sum_{y_t \in \mathbf{e}} \log p_\theta(y_t \mid y_{<t})$.

The difference in expected gradient updates is:
$$
\mathbf{g}_{\text{SFT}} - \mathbf{g}_{\text{mSFT}} = \mathbb{E} [\nabla_\theta \mathcal{L}_{\text{error}}]
$$
$$
\mathbf{g}_{\text{SFT}} - \mathbf{g}_{\text{mSFT}} = - \mathbb{E} \left[ \sum_{y_t \in \mathbf{e}} \nabla_\theta \log p_\theta(y_t \mid y_{<t}) \right]
$$

Since $\nabla \log p$ points in the direction of increasing probability, the term $-\nabla \log p$ points in the direction of *decreasing* probability. However, in gradient *descent* ($\theta \leftarrow \theta - \eta \nabla \mathcal{L}$), we move against the gradient.
The update contribution from SFT for errors is:
$$
\Delta \theta_{\text{error}} = - \eta \nabla \mathcal{L}_{\text{error}} = \eta \sum_{y_t \in \mathbf{e}} \nabla_\theta \log p_\theta(y_t \mid y_{<t})
$$
This vector explicitly increases the probability of error tokens. Masked SFT has this term identically equal to zero. \qed

---

### Part (c): Safety Divergence

**Statement**: $p_{\theta_{\text{SFT}}}(\mathbf{e} \mid \mathbf{p}) \gg p_{\theta_{\text{mSFT}}}(\mathbf{e} \mid \mathbf{p})$

**Proof**:
Let $\mathbf{v}_{\text{error}} = \mathbb{E}[\nabla_\theta \log p_\theta(\mathbf{e} \mid \mathbf{p})]$ be the average direction that increases error probability.
Let $\mathbf{v}_{\text{correct}} = \mathbb{E}[\nabla_\theta \log p_\theta(\mathbf{c} \mid \mathbf{p})]$ be the average direction that increases correction probability.

Under SFT, the parameter update is roughly:
$$
\Delta \theta_{\text{SFT}} \propto \mathbf{v}_{\text{correct}} + \mathbf{v}_{\text{error}} + \mathbf{v}_{\text{undo}}
$$
Under mSFT, the update is:
$$
\Delta \theta_{\text{mSFT}} \propto \mathbf{v}_{\text{correct}} + \mathbf{v}_{\text{undo}}
$$

After $T$ steps, the parameters diverge by $T \cdot \mathbf{v}_{\text{error}}$.
The probability of generating an error is monotonic in the projection of $\theta$ onto $\mathbf{v}_{\text{error}}$.
$$
\theta_{\text{SFT}} \cdot \mathbf{v}_{\text{error}} \approx (\theta_{\text{mSFT}} + T \mathbf{v}_{\text{error}}) \cdot \mathbf{v}_{\text{error}} = \theta_{\text{mSFT}} \cdot \mathbf{v}_{\text{error}} + T \|\mathbf{v}_{\text{error}}\|^2
$$
Since $\|\mathbf{v}_{\text{error}}\|^2 > 0$, the alignment with error generation grows linearly with training time for SFT, but remains constant (or follows a random walk) for mSFT.
Therefore, $p_{\theta_{\text{SFT}}}(\text{Error})$ grows significantly larger than $p_{\theta_{\text{mSFT}}}(\text{Error})$. \qed
