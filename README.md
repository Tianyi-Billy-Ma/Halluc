# Halluc: Non-Monotonic Autoregressive Sequence Modeling

**Paper Title**: Non-Monotonic Autoregressive Sequence Modeling via `<UNDO>` Token  
**Project**: N-MAR (Non-Monotonic Autoregressive Modeling)

## Overview

Autoregressive models generate sequences monotonically, where any sampled token—even if erroneous or sub-optimal—becomes a permanent condition for all subsequent steps. This rigidity makes them susceptible to **error propagation**, where a single deviation causes the generation trajectory to drift irreversibly into low-probability regions.

We propose a **Non-Monotonic Autoregressive (N-MAR)** sequence modeling framework that empowers autoregressive models to sample sequences non-monotonically via a single `<UNDO>` token.

### Key Mechanism
1.  **`<UNDO>` Token**: A special token that functionally deletes the preceding token from the sequence, allowing the model to prune divergent trajectories.
2.  **Training Pipeline**:
    *   **Augmentation**: Synthesize trajectories that demonstrate recovery from deviations (Error $\to$ `<UNDO>` $\to$ Correction).
    *   **Masked SFT**: Train on these traces while masking the loss for error tokens to prevent negative learning.
    *   **GRPO Refinement**: Optimize the policy to use `<UNDO>` efficiently (pruning bad paths without excessive backtracking).

## Installation

### Quick Start

```bash
# 1. Create environment (choose one)

# Option A: Conda
conda create -n llmhalluc python=3.11 -y && conda activate llmhalluc

# Option B: uv
uv venv && source .venv/bin/activate

# 2. Run install script (auto-detects CUDA)
./scripts/env/install.sh
```

### What Gets Installed

1. **PyTorch** - Auto-detects your CUDA version
2. **Core dependencies** - transformers, trl, peft, accelerate, etc.
3. **lm-evaluation-harness** - Cloned and installed in editable mode

### Optional: Flash Attention

```bash
pip install flash-attn  # Only if needed
```

## Project Structure

```
Halluc/
├── llmhalluc/          # Main package (N-MAR implementation)
├── configs/            # Training configs (SFT, GRPO)
├── lm-evaluation-harness/  # Eval framework
└── scripts/            # Installation & Cluster scripts
```

### Scripts

The script folder `./scripts` contains various utility scripts for model training from difference sources.

- `delta/`: Scripts for using the Delta supercomputer at NCSA.
- `crc/`: Scripts for using the CRC cluster at University of Notre Dame.
- `pcs/`: Scripts for using the PCS cluster at Pitt Supercomputer Center.
- `amz/`: Scripts for using Amazon Web Services (AWS) for training.
