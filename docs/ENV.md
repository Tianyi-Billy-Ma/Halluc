# Environment Compatibility (llamafactory vs. lm_eval)

## Overview
- `llamafactory` (training) installs `LLaMA-Factory` from `LLaMA-Factory/requirements.txt`.
- `lm_eval` (evaluation) installs `lm-evaluation-harness` via `lm-evaluation-harness/pyproject.toml`.
- The two stacks currently live in separate conda environments because their dependency constraints cannot be satisfied simultaneously.

## Conflict Inventory
| Area | LLaMA-Factory requirement | lm-evaluation-harness requirement | Impact |
| --- | --- | --- | --- |
| Python runtime | `python >= 3.9 (recommended 3.10)` per `LLaMA-Factory/README.md` (Requirements table) | `requires-python >= 3.10` in `lm-evaluation-harness/pyproject.toml` | Training can run on Python 3.9, but evaluation refuses to install there. Teams keeping training on 3.9 must split environments. |
| `peft` version | Core pin `0.14.0 – 0.17.1` in `LLaMA-Factory/requirements.txt` and documentation (`LLaMA-Factory/README.md`) | Hard floor `peft >= 0.2.0` in `lm-evaluation-harness/pyproject.toml`, plus runtime check for `>= 0.4.0` when `load_in_4bit` is used (`lm_eval/models/huggingface.py`) | There is no overlapping version range, so a single `pip` solve fails. Evaluation features that rely on adapters/4-bit loading expect APIs that shipped after 0.2.x, while training code still targets the older PEFT API surface. |

## Detailed Notes & Mitigations

### 1. Python runtime floor mismatch
**Evidence**
- `LLaMA-Factory/README.md` lists Python 3.9 as the minimum supported version with 3.10 recommended.
- `lm-evaluation-harness/pyproject.toml` declares `requires-python = ">=3.10"`, so pip/uv will abort on 3.9.

**Why it matters**
- Older training boxes that were provisioned with Python 3.9 cannot reuse the same interpreter for evaluation.

**Paths to fix**
1. **Standardize on Python ≥3.10** (preferred)  
   - Recreate the `llamafactory` conda env with `conda create -n llamafactory python=3.10` (or 3.11).  
   - Reinstall `LLaMA-Factory` (`pip install -e LLaMA-Factory`), verifying unit tests and a short training dry run.  
   - Once training is confirmed on ≥3.10, both environments can share the same base interpreter, reducing divergence.
2. **Keep 3.9 for legacy runs**  
   - If hardware constraints force Python 3.9, keep the split but call it out explicitly in `bash/sys/init_env.sh` so that automation knows which env pairs are compatible.

### 2. `peft` version pin mismatch
**Evidence**
- Training pins `peft>=0.14.0,<=0.17.1` (`LLaMA-Factory/requirements.txt`) and reiterates 0.14–0.15 in the README requirements table.
- Evaluation declares `peft>=0.2.0` in `pyproject.toml` and enforces `>=0.4.0` when `load_in_4bit=True` (`lm_eval/models/huggingface.py` lines 681–688).

**Why it matters**
- `pip/uv` cannot install both projects together because the solver cannot find a `peft` release that satisfies `[0.14.0, 0.17.1] ∩ [0.2.0, ∞)`.  
- Even if the dependency file were relaxed, runtime checks in `lm_eval` demand APIs introduced in PEFT 0.4+ for adapter/4-bit evaluation.

**Paths to fix**
1. **Upgrade training to PEFT ≥0.4.0**  
   - Audit the usages under `LLaMA-Factory/src/llamafactory/model/**` for breaking changes (notably LoRA config fields and `PeftModel.from_pretrained`).  
   - Update `LLaMA-Factory/requirements.txt` to `peft>=0.4.0` (or whichever version is validated) and run smoke tests for SFT, DPO, and PPO configs.  
   - This brings training in line with evaluation and allows a single environment once dependencies are resynced.
2. **Relax LM Eval’s requirement when adapters are not needed**  
   - Fork `lm-evaluation-harness/pyproject.toml`, moving `peft` to an optional extra (e.g., `peft>=0.4.0` under `[project.optional-dependencies.peft]`).  
   - Install with `pip install -e lm-evaluation-harness --config-settings editable_mode=compat --no-deps` inside the unified env, then rely on the training version of PEFT (0.17.1).  
   - Document that adapter evaluation and `load_in_4bit` are unavailable until PEFT is upgraded.
3. **Maintain split environments (stopgap)**  
   - Continue activating `llamafactory` for training and `lm_eval` for evaluation via `bash/sys/init_env.sh`.  
   - Explicitly freeze `peft==0.17.1` in the training env and `peft==0.6.x` (or newer) in the evaluation env to avoid accidental cross-contamination when developers install packages globally.

## Recommended Next Steps
1. Decide whether upgrading `LLaMA-Factory` to PEFT ≥0.4.0 is feasible in the near term; if yes, prioritize that path so the dual-env requirement goes away.
2. If not, capture the adapter/4-bit limitation in `docs/lm_eval/README.md` and keep the environments isolated until the training code is ported.

