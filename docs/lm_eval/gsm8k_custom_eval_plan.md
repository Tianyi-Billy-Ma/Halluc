# GSM8K Custom Evaluation Plan

## Objectives
- Provide a reusable evaluation entry point for GSM8K experiments that wraps `lm-evaluation-harness`.
- Support both standard GSM8K and the symbolic/backtrack variants produced by `llmhalluc.data.gsm8k`.
- Ensure evaluations can target locally fine-tuned adapters (e.g., LLaMA-Factory LoRA checkpoints) with consistent logging.

## Current Artifacts Review
- Dataset metadata lives in `data/dataset_info.json` (`gsm8k_eval`, `gsm8k_bt_eval`) and converters under `llmhalluc/data/gsm8k.py`.
- Existing evaluation configs (e.g., `configs/qwen3/4b/gsm8k_eval.yaml`) trigger LLaMA-Factory evaluation but stop short of harness-based metrics.
- `docs/lm_eval/task_guide.md` documents harness YAML usage; no project-specific runner exists yet.

## Implementation Phases

1. Baseline Harness Integration
   - Add a Python script under `llmhalluc/scripts/` (`run_lm_eval.py`) that imports `lm_eval.evaluator.simple_evaluate`.
   - Expose CLI arguments for model URI (`--model_args`), task variant (`--task`, default `gsm8k_cot`), split, output directory, batch size, and optional self-consistency sampling.
   - Enforce execution via `uv run` in documentation and sample commands to align with project standards.

2. Custom Task Bridging
   - Register internal helper(s) in `llmhalluc` to load symbolic/backtrack datasets without modifying `lm-evaluation-harness` (e.g., via `custom_dataset` callable or temporary JSON export).
   - Provide prompt templates that mirror `llmhalluc/prompts/MathPrompt.py` to keep reasoning style consistent.
   - Allow toggling between vanilla GSM8K and backtrack-enhanced prompts through CLI switches.

3. Output Management and Metrics
   - Configure metric list to report exact match and optional reasoning-quality metrics; ensure filters extract final numerical answers (reuse harness regex filters where possible).
   - Save aggregated metrics and per-sample generations under `Experiments/<model>/gsm8k_eval/` with timestamped filenames.
   - Offer optional integrations for W&B or CSV exports while keeping defaults filesystem-only.

4. Validation and Automation
   - Add lightweight unit tests (PyTest) to validate CLI argument parsing and harness invocation with a mocked mini dataset.
   - Document usage in `docs/lm_eval/README.md` (new section) with step-by-step commands, including how to point at LoRA checkpoints exported from LLaMA-Factory.
   - Integrate script with existing bash automation (optional follow-up) for repeatable nightly runs.

## Open Questions
- Do we need bespoke answer post-processing beyond regex extraction for backtrack completions?
- Should evaluations support mixed-precision or distributed launches, or remain single-node initially?
