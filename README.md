# Halluc

LLM Hallucination Research with Backtracking Mechanisms.

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
├── llmhalluc/          # Main package
├── configs/            # Training configs
├── lm-evaluation-harness/  # Eval framework
└── scripts/            # Installation scripts
```