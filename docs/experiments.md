# Experiments

## Experiment Results

| Experiment ID | Model | Dataset | Method | EM (strict) | EM (flex) | BLEU | ROUGE-L | Notes |
|---------------|-------|---------|--------|--------------|-----------|------|---------|-------|
| EXP-001 | qwen3-4b | gsm8k | vanilla | 79.30 ± 1.12 | 83.09 ± 1.03 | - | - | Baseline evaluation |
| EXP-002 | qwen3-0.6b | gsm8k | vanilla | 10.69 ± 0.85 | 29.34 ± 1.25 | - | - | Smaller model comparison |
| EXP-003 | qwen3-4b | gsm8k | sft | - | - | 49.76 | 59.07 | SFT with LoRA training |
| EXP-004 | qwen3-4b | gsm8k | sft | - | - | 34.08 | 39.27 | SFT with LoRA + backtrack |
| EXP-005 | llama-3.2-3b | gsm8k | vanilla | 21.68 ± 1.14 | 63.31 ± 1.33 | - | - | Llama baseline |
| EXP-006 | qwen3-0.6b | gsm8k | sft | - | - | 30.36 | 33.80 | SFT with LoRA + backtrack (0.6B) |

### Metrics Legend
- **EM (strict)**: Exact Match with strict matching (extracts answer after ####)
- **EM (flex)**: Exact Match with flexible extraction (extracts any number)
- **BLEU**: BLEU-4 score for text generation quality
- **ROUGE-L**: Longest common subsequence based ROUGE score

## Experiment Tracking Template

| Experiment ID | Model | Dataset | Method | Config | Status | Start Date | End Date | Results | Notes |
|---------------|-------|---------|--------|--------|--------|------------|----------|---------|-------|
| EXP-001 | qwen3-4b | gsm8k | vanilla | fewshot_8 | completed | 2025-01-03 | 2025-01-03 | [results](outputs/qwen3-4b/gsm8k/vanilla/lm_eval/fewshot_8/results_2025-10-03T16-58-48.072096.json) | Baseline evaluation |
| EXP-002 | qwen3-0.6b | gsm8k | vanilla | fewshot_8 | completed | 2025-01-03 | 2025-01-03 | [results](outputs/qwen3-0.6b/gsm8k/vanilla/lm_eval/fewshot_8/results_2025-10-03T16-29-12.982645.json) | Smaller model comparison |
| EXP-003 | qwen3-4b | gsm8k | sft | lora/train | completed | 2025-01-03 | 2025-01-03 | [predictions](outputs/qwen3-4b/gsm8k/sft/lora/train/generated_predictions.jsonl) | SFT with LoRA training |
| EXP-004 | qwen3-4b | gsm8k | sft | lora/train_bt | completed | 2025-01-03 | 2025-01-03 | [predictions](outputs/qwen3-4b/gsm8k/sft/lora/train_bt/generated_predictions.jsonl) | SFT with LoRA + backtrack |
| EXP-005 | llama-3.2-3b | gsm8k | vanilla | fewshot_8 | completed | 2025-01-03 | 2025-01-03 | [results](outputs/llama-3.2-3b/gsm8k/vanilla/lm_eval/fewshot_8/results_2025-09-29T16-48-49.286255.json) | Llama baseline |
| EXP-006 | qwen3-0.6b | gsm8k | sft | lora/train_bt | completed | 2025-01-03 | 2025-01-03 | [predictions](outputs/qwen3-0.6b/gsm8k/sft/lora/train_bt/generated_predictions.jsonl) | SFT with LoRA + backtrack (0.6B) |

## Experiment Details

### EXP-001: Qwen3-4B Vanilla Evaluation
- **Model**: qwen3-4b
- **Dataset**: gsm8k
- **Method**: vanilla (no fine-tuning)
- **Configuration**: fewshot_8
- **Status**: completed
- **Results**: [View results](outputs/qwen3-4b/gsm8k/vanilla/lm_eval/fewshot_8/results_2025-10-03T16-58-48.072096.json)
- **Samples**: [View samples](outputs/qwen3-4b/gsm8k/vanilla/lm_eval/fewshot_8/samples_gsm8k_2025-10-03T16-58-48.072096.jsonl)

### EXP-002: Qwen3-0.6B Vanilla Evaluation
- **Model**: qwen3-0.6b
- **Dataset**: gsm8k
- **Method**: vanilla (no fine-tuning)
- **Configuration**: fewshot_8
- **Status**: completed
- **Results**: [View results](outputs/qwen3-0.6b/gsm8k/vanilla/lm_eval/fewshot_8/results_2025-10-03T16-29-12.982645.json)

### EXP-003: Qwen3-4B SFT with LoRA
- **Model**: qwen3-4b
- **Dataset**: gsm8k
- **Method**: supervised fine-tuning (SFT)
- **Configuration**: lora/train
- **Status**: completed
- **Predictions**: [View predictions](outputs/qwen3-4b/gsm8k/sft/lora/train/generated_predictions.jsonl)

### EXP-004: Qwen3-4B SFT with LoRA + Backtrack
- **Model**: qwen3-4b
- **Dataset**: gsm8k
- **Method**: supervised fine-tuning (SFT) with backtrack
- **Configuration**: lora/train_bt
- **Status**: completed
- **Predictions**: [View predictions](outputs/qwen3-4b/gsm8k/sft/lora/train_bt/generated_predictions.jsonl)

## Status Legend
- **pending**: Experiment not started
- **running**: Experiment in progress
- **completed**: Experiment finished successfully
- **failed**: Experiment failed
- **cancelled**: Experiment was cancelled

## Configuration Types
- **vanilla**: No fine-tuning, direct model evaluation
- **sft**: Supervised fine-tuning
- **fewshot_8**: 8-shot evaluation
- **lora**: Low-rank adaptation
- **train**: Training configuration
- **train_bt**: Training with backtrack

## Results Format
- **results_*.json**: Evaluation metrics and scores
- **samples_*.jsonl**: Individual sample predictions
- **generated_predictions.jsonl**: Model predictions during training/inference
