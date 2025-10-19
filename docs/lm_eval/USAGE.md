# LM Eval Runner - Usage Guide

## Overview

The `llmhalluc/run_eval.py` script provides a YAML-driven interface to run lm_eval evaluations while maintaining full backwards compatibility with the original CLI interface.

## Quick Start

### Using YAML Configuration

```bash
# Using default configuration
python llmhalluc/run_eval.py --config configs/lm_eval/run/default.yaml --tasks gsm8k

# Using example GSM8K configuration
python llmhalluc/run_eval.py --config configs/lm_eval/run/example_gsm8k.yaml

# Quick test run (small model, limited examples)
python llmhalluc/run_eval.py --config configs/lm_eval/run/test_run.yaml
```

### Using CLI Arguments Only

```bash
# Traditional CLI usage (still supported)
python llmhalluc/run_eval.py --tasks gsm8k --model hf \
  --model_args "pretrained=meta-llama/Llama-2-7b-hf,dtype=float16" \
  --num_fewshot 5 --batch_size auto --device cuda:0
```

### Mixing YAML and CLI (CLI overrides YAML)

```bash
# Load config but override specific parameters
python llmhalluc/run_eval.py --config configs/lm_eval/run/example_gsm8k.yaml \
  --num_fewshot 8 \
  --limit 100 \
  --wandb_args "project=my-project,name=custom-run"
```

## Directory Structure

```
configs/lm_eval/
├── tasks/                    # Custom task definitions
│   └── gsm8k_custom.yaml    # Example custom task
└── run/                      # Run configurations
    ├── default.yaml         # Default configuration template
    ├── example_gsm8k.yaml   # Example GSM8K run config
    └── test_run.yaml        # Quick test configuration
```

## Custom Tasks

Custom tasks placed in `configs/lm_eval/tasks/` are automatically discovered. The directory is added to the task search path by default.

### Creating a Custom Task

1. Create a YAML file in `configs/lm_eval/tasks/` (e.g., `my_task.yaml`)
2. Define the task following lm_eval task format (see `gsm8k_custom.yaml` for example)
3. Reference it in your run config: `tasks: "my_task"`

## Configuration File Format

See `configs/lm_eval/run/default.yaml` for a comprehensive template with all available parameters. Key sections:

- **Model Configuration**: `model`, `model_args`
- **Tasks**: `tasks` (required)
- **Few-shot**: `num_fewshot`
- **Batching**: `batch_size`, `max_batch_size`
- **Device**: `device` (e.g., "cuda:0", "cpu")
- **Output**: `output_path`, `log_samples`
- **Logging**: `wandb_args`, `wandb_config_args`
- **Reproducibility**: `seed`

## Examples

### Example 1: Evaluate with Custom Model

```yaml
# configs/lm_eval/run/my_model.yaml
model: "hf"
model_args: "pretrained=/path/to/my/model,dtype=bfloat16"
tasks: "gsm8k,arc_easy,arc_challenge"
num_fewshot: 5
batch_size: "auto"
device: "cuda:0"
output_path: "results/my_model_eval"
log_samples: true
wandb_args: "project=my-evals,name=my-model-v1"
```

Run: `python llmhalluc/run_eval.py --config configs/lm_eval/run/my_model.yaml`

### Example 2: Quick Debug Run

```bash
python llmhalluc/run_eval.py \
  --config configs/lm_eval/run/default.yaml \
  --tasks gsm8k \
  --model_args "pretrained=gpt2" \
  --limit 10 \
  --device cpu \
  --log_samples false
```

### Example 3: Multiple Tasks with W&B Logging

```yaml
# configs/lm_eval/run/benchmark.yaml
model: "hf"
model_args: "pretrained=meta-llama/Llama-2-13b-hf,dtype=float16"
tasks: "gsm8k,mmlu,hellaswag,arc_challenge"
num_fewshot: 5
batch_size: "auto"
device: "cuda"
output_path: "results/benchmark_llama2_13b"
log_samples: true
wandb_args: "project=llm-benchmarks,entity=my-team,name=llama2-13b-full"
seed: "42,42,42,42"
```

## Testing

Verify the setup works:

```bash
# List all available tasks (including custom ones)
python llmhalluc/run_eval.py --tasks list | grep custom

# Test with minimal config
python llmhalluc/run_eval.py --config configs/lm_eval/run/test_run.yaml
```

## Tips

1. **Start with test config**: Use `test_run.yaml` to validate your setup before running full evaluations
2. **Use auto batch size**: Set `batch_size: "auto"` to automatically find optimal batch size
3. **CLI overrides**: Any CLI argument will override the YAML config value
4. **Custom tasks**: Place task definitions in `configs/lm_eval/tasks/` for automatic discovery
5. **W&B integration**: Set `wandb_args` to automatically log results to Weights & Biases

## Troubleshooting

### Task not found
- Ensure task YAML is in `configs/lm_eval/tasks/` or another included path
- Check task name matches the `task:` field in the YAML
- Use `--tasks list` to see all available tasks

### Config not loading
- Verify YAML syntax is correct
- Check file path is correct relative to working directory
- Use absolute paths if needed

### Model loading issues
- Verify `model_args` format (comma-separated or JSON)
- Check model path exists
- Ensure sufficient GPU memory if using CUDA
