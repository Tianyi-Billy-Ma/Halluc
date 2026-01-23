# Research Status: N-MAR Framework

**Project**: N-MAR (Non-Monotonic Autoregressive Modeling)  
**Previous Focus**: Hallucination Mitigation (Pivoted)

---

## 1. Current Research Focus
We have shifted from a pure "hallucination mitigation" perspective to a broader structural innovation: **Non-Monotonic Autoregressive Modeling (N-MAR)** via the `<UNDO>` token.

**Core Hypothesis**: The rigidity of monotonic autoregressive generation (where $y_t$ permanently conditions $y_{>t}$) is the root cause of error propagation. Enabling the model to *prune* divergent trajectories via `<UNDO>` solves this structural flaw.

---

## 2. Research Roadmap

### Phase 1: Foundation (Completed)
- [x] **Codebase Analysis**: Established current limitations of SFT on error traces.
- [x] **Pivot Definition**: Formulated N-MAR framework and `<UNDO>` mechanism.
- [x] **Theoretical Proof**: Proved "Safety Gap" between Masked SFT and Standard SFT (see `docs/research/paper/theory/masked_sft/`).

### Phase 2: Implementation (In Progress)
- [ ] **Data Pipeline**: 
    - [ ] Augment error traces with `<UNDO>`.
    - [ ] Implement `MaskedSFTCollator` to zero-out loss on errors.
- [ ] **Model Training**:
    - [ ] SFT Baseline (Monotonic).
    - [ ] N-MAR SFT (Masked).
    - [ ] GRPO Refinement (Efficiency).

### Phase 3: Evaluation (Planned)
- [ ] **Benchmarks**: GSM8K (Reasoning), Hallucination Eval (Factuality).
- [ ] **Metrics**:
    - [ ] Accuracy (Outcome).
    - [ ] Pruning Efficiency (Precision of `<UNDO>`).
    - [ ] Context Pollution Analysis (Does `<UNDO>` truly clean the state?).

---

## 3. Key Concepts

| Term | Definition |
| :--- | :--- |
| **Monotonicity** | The standard AR constraint where output length only increases. |
| **N-MAR** | Non-Monotonic AR framework allowing token deletion. |
| **Divergence** | A trajectory drifting into low-probability/erroneous regions. |
| **Pruning** | The act of using `<UNDO>` to remove a divergent branch. |
| **Masked SFT** | Training technique that masks loss on deviations to prevent negative learning. |

---

## 4. Documentation Status
- `README.md`: Updated to reflect N-MAR focus.
- `AGENTS.md`: Updated system context.
- `docs/research/GRPO_Reward_Design.md`: Aligned reward logic with N-MAR efficiency goals.
- `docs/research/backtrack_analysis.md`: Reframed problem as "Monotonicity Bottleneck".
