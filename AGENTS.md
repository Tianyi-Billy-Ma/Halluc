# HALLUC - LLM Backtracking Research

**Generated:** 2026-01-14 | **Commit:** 39d0acb1 | **Branch:** main

## IMPORTANT: RESEARCH-ONLY CODEBASE

This is a **research-only** codebase. It is NOT released publicly and is NOT intended for production use.

**Implications:**
- **No backward compatibility**: APIs and configs may change without migration support
- **No semantic versioning**: Breaking changes happen freely
- **No deprecation warnings**: Old code is removed, not deprecated
- **Experimental features**: Code quality prioritizes research velocity over stability

When modifying code, focus on correctness and clarity. Do not add compatibility shims or legacy support.

## OVERVIEW

LLM hallucination mitigation via **backtracking mechanisms** - models learn to self-correct by generating `<|BACKTRACK|>` tokens that functionally delete preceding tokens. Built on HuggingFace ecosystem (transformers, trl, peft) with lm-evaluation-harness for eval.

## STRUCTURE

```
Halluc/
├── llmhalluc/           # Main package
│   ├── run_train.py     # Entry: python -m llmhalluc.run_train
│   ├── run_eval.py      # Entry: python -m llmhalluc.run_eval
│   ├── run_exp.py       # Combined train→eval pipeline
│   ├── train/           # SFT, DPO, GRPO executors
│   ├── data/            # Dataset converters (backtrack injection)
│   ├── reward/          # GRPO reward functions (bt.py = core)
│   ├── models/          # Model loading, patcher, PEFT
│   ├── hparams/         # Argument dataclasses + config parsing
│   ├── eval/            # Custom metrics (ROUGE, filters)
│   ├── prompts/         # Prompt templates (Math, QA)
│   └── extras/          # Constants, templates
├── configs/
│   ├── llmhalluc/       # Training configs (sft.yaml, grpo.yaml, dpo.yaml)
│   ├── lm_eval/tasks/   # Custom eval tasks (gsm8k_simple.yaml + .py)
│   └── deepspeed/       # ZeRO stage configs
├── scripts/             # Cluster-specific (crc/, psu/, delta/, amz/)
├── docs/                # Research notes, implementation guides
└── lm-evaluation-harness/  # Submodule/vendored eval framework
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Add training method | `llmhalluc/train/` | Subclass `BaseExecutor`, register in `__init__.py` |
| New dataset converter | `llmhalluc/data/` | Subclass `DatasetConverter`, register in `manager.py` |
| Modify reward function | `llmhalluc/reward/bt.py` | `BacktrackRewardFunction` - outcome/process/efficiency |
| Add special tokens | `llmhalluc/models/patcher.py` | + update `extras/constant.py` |
| New eval task | `configs/lm_eval/tasks/` | YAML + Python pair (see gsm8k_simple) |
| Training hyperparams | `llmhalluc/hparams/train_args.py` | Dataclass fields |
| Cluster job scripts | `scripts/{crc,psu,delta}/` | Platform-specific SLURM/SGE |

## CODE MAP

| Symbol | Type | Location | Role |
|--------|------|----------|------|
| `BaseExecutor` | Class | train/base.py | Training lifecycle (tokenizer→model→dataset→trainer) |
| `GRPOExecutor` | Class | train/grpo.py | Group Relative Policy Optimization |
| `BacktrackRewardFunction` | Class | reward/bt.py | Multi-component reward (outcome + efficiency + process) |
| `BacktrackDatasetConverter` | Class | data/backtrack.py | Injects error→backtrack→correction sequences |
| `_apply_backtracking` | Func | models/helper.py | Token-level backtrack logic (pop on BACKTRACK token) |
| `run_train` | Func | train/__init__.py | Dispatcher to stage-specific executors |
| `run_eval` | Func | eval/base.py | Wraps lm_eval.simple_evaluate |

## CONVENTIONS

### Environment Separation (CRITICAL)
- **Training**: `llamafactory` conda env (or `llmhalluc` for local dev)
- **Evaluation**: `lm_eval` conda env
- Reason: Dependency conflicts between trl and lm-evaluation-harness

### Activation
```bash
source ~/.activate_conda  # If conda not active
conda activate llmhalluc  # Local development
```

### Execution Pattern
Always use Python scripts, NOT CLI tools:
```bash
python -m llmhalluc.run_train --config configs/llmhalluc/grpo.yaml
python -m llmhalluc.run_eval --config configs/lm_eval/eval.yaml
```

### Config Loading
YAML configs merged with CLI overrides via `omegaconf`. Check `llmhalluc/hparams/parser.py`.

### Eval Task Pattern
Custom lm-eval tasks use **YAML+Python pairs**:
- `gsm8k_simple.yaml` - Task definition
- `gsm8k_simple.py` - Custom `process_results` for hybrid metrics

## ANTI-PATTERNS

| DO NOT | WHY |
|--------|-----|
| Modify KV-cache or attention mask for backtracking | Research proved ineffective - use token-based approach |
| Train model to generate error tokens | Only train to DETECT errors (backtrack) and CORRECT |
| Use easily-gamed reward structures | Risk of "always backtrack" exploitation |
| Skip curriculum learning in GRPO | Required to prevent reward hacking |
| Run training in `lm_eval` env (or vice versa) | Dependency conflicts will break |
| Use `@ts-ignore` / `as any` equivalents | N/A Python project |

## BACKTRACKING MECHANISM

Core innovation - `<|BACKTRACK|>` token acts as functional backspace:

1. **Token Logic** (`models/helper.py`): When model generates BACKTRACK, pop previous token from sequence
2. **Data Augmentation** (`data/backtrack.py`): Insert synthetic error→backtrack→correction sequences
3. **Reward System** (`reward/bt.py`):
   - `outcome_accuracy`: Correctness AFTER applying backtracks
   - `backtrack_efficiency`: Penalize unnecessary, reward successful corrections
   - `process_quality`: Step-by-step reasoning evaluation
4. **Curriculum**: Shift from process→outcome focus as training progresses

## COMMANDS

We use `uv` for local development and `conda` for training on computing cluster.
For conda environment, we use `llmhalluc` or `billy`.

### macOS Local Development (LSP/Type-Checking Only)

On macOS, we install a minimal set of packages for LSP support and early bug detection.
**We do NOT run training on macOS** - this is only for code editing and type checking.

```bash
# Create venv and install macos dependencies
uv venv
source .venv/bin/activate
uv pip install -e ".[macos,dev]"
```

The `macos` optional dependency excludes CUDA-specific packages (bitsandbytes, deepspeed)
that won't work on macOS. Some packages like deepspeed may fail to import on newer Python
versions - this is expected and does not affect type checking.

### Training Commands (Linux/Cluster Only)

```bash
# Training (uses accelerate)
python -m llmhalluc.run_train --config configs/llmhalluc/grpo.yaml

# Evaluation
python -m llmhalluc.run_eval --config configs/lm_eval/eval.yaml

# End-to-end (train + eval)
python -m llmhalluc.run_exp --config configs/llmhalluc/e2e.yaml

# Clean caches
./scripts/sys/clean_pycache.sh
./scripts/sys/clean_logs.sh
```

## NOTES

- **Missing install script**: `scripts/env/install.sh` referenced in docs but directory doesn't exist
- **Empty " 2" directories**: Artifact dirs (`configs/llamafactory 2`, etc.) - safe to delete
- **No CI/CD**: No GitHub Actions. Tests via lm-evaluation-harness, not pytest
- **Cluster scripts reference `./bash/`**: May need path fixes to `./scripts/sys/`
- **Constants**: `llmhalluc/extras/constant.py` is single source of truth for special tokens, paths
- **Model-specific tokens**: Llama3 uses `<|reserved_special_token_0|>`, Qwen3 uses `<|BACKTRACK|>`


## AGENTS
You have full access to the `docs/agents/` directory which contains documentation and notes that might related to your tasks. 
Feel free to explore the files within that directory to gather the information you need.

### Memories 
The file `docs/agents/memories.md` contains important memories that you should be aware of.
When the user ask you to memnorize something, please store that information in the `docs/agents/memories.md` file for future reference.