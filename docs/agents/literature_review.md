# Literature Review: Reasoning-Based Classification and Verification

**Generated:** 2026-01-25

This document summarizes papers on approaches where LLMs use reasoning quality/confidence to make classification or selection decisions.

---

## 1. Inference to the Best Explanation (IBE)

### IBE-Eval: Inference to the Best Explanation in Large Language Models
**Authors:** Dalal et al.  
**Venue:** ACL 2024  
**URL:** https://arxiv.org/abs/2402.10767

**Core Mechanism:**
- Framework inspired by philosophical Inference to the Best Explanation (IBE)
- Evaluates LLM-generated explanations using features: **consistency, parsimony, coherence, uncertainty**
- Uses neuro-symbolic approach: formalizes explanations into Prolog rules, verifies logical consistency
- Selects classification based on which explanation scores highest on IBE criteria

**Key Results:**
- Identifies best explanation with up to **77% accuracy** on Causal QA
- Outperforms GPT-3.5-as-a-Judge baseline
- Linguistic uncertainty found to be the best single predictor

**Relevance:** Directly implements "reasoning confidence for classification" - generates explanations for each candidate, selects based on explanation quality.

---

### Cycles of Thought: Measuring LLM Confidence through Stable Explanations
**Authors:** Becker et al.  
**Venue:** arXiv 2024  
**URL:** https://arxiv.org/abs/2406.03441

**Core Mechanism:**
- Measures LLM uncertainty by analyzing **distribution of generated explanations**
- Interprets each (model + explanation) pair as a test-time classifier
- Uses **explanation entailment** as classifier likelihood to calculate posterior answer distribution
- Stable explanations → higher confidence; inconsistent explanations → lower confidence

**Key Results:**
- Improves AURC and AUROC metrics over baselines across 5 datasets
- Effective for quantifying uncertainty without access to logits

**Relevance:** Uses reasoning/explanation stability as a proxy for classification confidence.

---

## 2. Noisy Channel / Reverse Probability Approaches

### Noisy Channel Language Model Prompting for Few-Shot Text Classification
**Authors:** Sewon Min, Mike Lewis, Hannaneh Hajishirzi, Luke Zettlemoyer  
**Venue:** ACL 2022  
**URL:** https://arxiv.org/abs/2108.04106

**Core Mechanism:**
- Uses **channel models**: P(input|label) instead of direct P(label|input)
- Forces model to "explain every word in the input" given each label
- Applies Bayes rule: P(label|input) ∝ P(input|label) × P(label)
- Selection based on which label best explains the input

**Key Results:**
- Significantly outperforms direct models in few-shot settings
- **Lower variance, higher worst-case accuracy**
- Preferred when: few examples, imbalanced labels, unseen label generalization

**Relevance:** The channel model essentially asks "how well can I generate this input assuming this label?" - a form of reasoning-based classification.

---

## 3. Self-Consistency and Voting Methods

### Self-Consistency Improves Chain of Thought Reasoning in Language Models
**Authors:** Xuezhi Wang et al.  
**Venue:** ICLR 2023  
**URL:** https://arxiv.org/abs/2203.11171

**Core Mechanism:**
- Replaces greedy decoding with **sampling diverse reasoning paths**
- Generates multiple chain-of-thought solutions for same problem
- **Majority voting** over final answers to select most consistent answer
- Intuition: correct answers reached via multiple distinct reasoning paths

**Key Results:**
- **5-25% accuracy improvement** on reasoning tasks (GSM8K, SVAMP, AQuA)
- Simple to implement (no fine-tuning needed)
- Works across different LLM sizes

**Relevance:** Uses reasoning path diversity as implicit confidence measure - answers supported by more reasoning paths are selected.

---

### Ranked Voting based Self-Consistency
**Authors:** Wang et al.  
**Venue:** ACL Findings 2025  
**URL:** https://arxiv.org/abs/2505.10772

**Core Mechanism:**
- Extension of self-consistency that generates **ranked answer lists** per reasoning path
- Uses ranked voting methods:
  - Instant-runoff voting
  - Borda count voting
  - Mean reciprocal rank voting
- Captures alternative answers that standard majority voting ignores

**Key Results:**
- Outperforms baseline majority voting across 6 datasets
- Works on both multiple-choice and open-ended QA

**Relevance:** More sophisticated aggregation of reasoning-based confidence.

---

## 4. Process Reward Models (PRM) and Verifiers

### Let's Verify Step by Step
**Authors:** OpenAI (Lightman et al.)  
**Venue:** arXiv 2023 / OpenAI Research  
**URL:** https://arxiv.org/abs/2305.20050

**Core Mechanism:**
- Compares **outcome supervision** (reward final answer only) vs **process supervision** (reward each reasoning step)
- Process Reward Model (PRM): trained on 800K step-level human feedback labels (PRM800K dataset)
- Verifier scores each step of reasoning, selects solutions with highest process scores

**Key Results:**
- PRM achieves **78% solve rate** on MATH dataset subset
- Significantly outperforms Outcome Reward Model (ORM)
- Process supervision provides better credit assignment

**Relevance:** Verifier evaluates reasoning quality step-by-step to select correct solutions.

---

### Generative Verifiers: Reward Modeling as Next-Token Prediction
**Authors:** Zhang, Hosseini et al.  
**Venue:** arXiv 2024 (ICLR 2025)  
**URL:** https://arxiv.org/abs/2408.15240

**Core Mechanism:**
- Reframes reward modeling as **next-token prediction** (not classification)
- **GenRM**: Verifier generates "Yes"/"No" tokens, score derived from P("Yes")
- **GenRM-CoT**: Generates reasoning rationale before verification decision
- Enables chain-of-thought reasoning within the reward model itself

**Key Results:**
- **16-40% improvement** over discriminative verifiers on Best-of-N selection
- GSM8K: 73% → 93.4% with Best-of-N using GenRM
- Scales with model size and inference compute

**Relevance:** Verifier uses reasoning (CoT) to evaluate solution quality - the verification process itself involves generating explanations.

---

### V-STaR: Training Verifiers for Self-Taught Reasoners
**Authors:** Hosseini et al.  
**Venue:** COLM 2024  
**URL:** https://arxiv.org/abs/2402.06457

**Core Mechanism:**
- Iterative self-improvement: use both correct AND incorrect self-generated solutions
- Train verifier using **DPO** on (correct, incorrect) solution pairs
- Verifier selects best solution from candidates at inference time
- Iterative training improves both generator and verifier

**Key Results:**
- **4-17% accuracy improvement** over prior self-improvement methods
- 7B V-STaR surpasses base LLaMA2 70B (8-shot) on GSM8K
- DPO training more effective than ORM approach

**Relevance:** Verifier trained to distinguish correct from incorrect reasoning, used for solution selection.

---

## 5. Confidence Calibration Surveys

### A Survey of Confidence Estimation and Calibration in Large Language Models
**Authors:** Jiahui Geng et al.  
**Venue:** NAACL 2024  
**URL:** https://arxiv.org/abs/2311.08298

**Core Mechanism (Survey):**
- Comprehensive overview of confidence estimation and calibration in LLMs
- Categorizes methods for generation vs classification tasks
- Discusses challenges: large output space, prompt sensitivity
- Applications: hallucination detection, ambiguity detection

**Key Findings:**
- LLMs often miscalibrated (confident when wrong, uncertain when right)
- Consistency-based methods (related to self-consistency) promising for calibration
- Fine-tuning can improve calibrated uncertainty expressions

**Relevance:** Background on the confidence calibration problem and existing approaches.

---

### A Survey on Uncertainty Quantification of Large Language Models
**Authors:** Various  
**Venue:** arXiv 2024  
**URL:** https://arxiv.org/abs/2412.05563

**Core Mechanism (Survey):**
- Taxonomy of UQ methods by computational efficiency and uncertainty dimensions
- Dimensions include: **input, reasoning, parameter, prediction uncertainty**
- Covers both white-box (logit access) and black-box methods

**Key Findings:**
- Reasoning uncertainty is a distinct dimension worth quantifying
- Multi-hop reasoning particularly challenging for uncertainty estimation

**Relevance:** Provides framework for understanding reasoning uncertainty.

---

## Summary Table

| Paper | Year | Core Idea | Selection Mechanism |
|-------|------|-----------|---------------------|
| IBE-Eval | 2024 | Evaluate explanations with IBE criteria | Highest IBE score (consistency, parsimony, coherence) |
| Cycles of Thought | 2024 | Explanation distribution stability | Stable explanations → high confidence |
| Noisy Channel LM | 2022 | P(input\|label) instead of P(label\|input) | Best "explainer" of input |
| Self-Consistency | 2023 | Multiple reasoning paths | Majority vote over answers |
| Let's Verify Step by Step | 2023 | Step-level verification | Highest process reward score |
| Generative Verifiers | 2024 | Reward as next-token prediction + CoT | Verifier with reasoning selects best |
| V-STaR | 2024 | DPO-trained verifier on correct/incorrect pairs | Verifier ranks candidates |

---

## Relation to N-MAR

The N-MAR approach (Non-Monotonic Autoregressive Modeling with `<UNDO>` token) relates to these works in several ways:

1. **Error Detection**: Like PRM/verifiers, N-MAR implicitly learns to detect when reasoning goes wrong (triggering `<UNDO>`)

2. **Self-Correction**: Unlike post-hoc verification (V-STaR, GenRM), N-MAR enables **online correction** during generation

3. **Reasoning Quality Signal**: The `<UNDO>` token acts as an implicit signal of low reasoning quality at a particular step

4. **Key Differentiator**: These works use verification/scoring AFTER generation is complete. N-MAR allows pruning DURING generation, potentially more efficient for avoiding divergent trajectories.

**Potential Synthesis:**
- Could train PRM/verifier to predict when `<UNDO>` should be triggered
- Could use IBE-style evaluation to score backtrack decisions
- GenRM-CoT approach could inform how to make `<UNDO>` decisions via reasoning
