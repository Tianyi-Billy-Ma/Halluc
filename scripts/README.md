# Scripts

This folder contains utility scripts organized by purpose.

## Directory Structure

```
scripts/
├── env/                 # Environment installation
│   ├── install.sh       # Main entry point (run this!)
│   ├── install_torch.sh # CUDA-aware PyTorch install
│   └── install_lm_eval.sh # lm-evaluation-harness setup
├── sys/                 # System utilities
│   ├── clean_logs.sh    # Clean log files
│   ├── notify.sh        # Notifications
│   └── tunnel/          # SSH tunneling
└── [cluster dirs]       # Cluster-specific scripts (crc, psu, delta, amz)
```

## Installation

See `env/install.sh` - the main installation script:

```bash
# Option A: Conda
conda create -n llmhalluc python=3.11 -y && conda activate llmhalluc
./scripts/env/install.sh

# Option B: uv
uv venv && source .venv/bin/activate
./scripts/env/install.sh
```
