# Agent Memories

## Argument Hierarchy & Best Practices

**Concept**:
In this codebase, `TrainArguments` acts as the **master configuration** schema for the initial CLI/YAML parsing pipeline (via `patch_configs`). However, the actual fine-tuning executors (SFT, DPO, GRPO) operate on arguments derived from **`FTArguments`** (which `SFTArguments`, etc. inherit from).

**Rule**:
When you need to add a new argument relevant to training:
1.  **MUST** add it to `FTArguments` (`llmhalluc/hparams/ft_args.py`). This ensures it is available to the specific executors (`SFTExecutor`, `GRPOExecutor`).
2.  **SHOULD** add it to `TrainArguments` (`llmhalluc/hparams/train_args.py`) if it needs to be parsed from the main `train_config.yaml` or CLI entry point before dispatching to specific stages.

**Why**: If you only add it to `TrainArguments`, the `SFTExecutor` (which expects `SFTArguments`) won't see it because `SFTArguments` doesn't inherit from `TrainArguments`. If you only add it to `FTArguments`, the initial configuration loading might miss it if strict parsing is enabled on `TrainArguments`.

## File System Constraints

**Rule**:
**NEVER** save `.tex` (LaTeX) files or other intermediate research artifacts into the codebase directory unless the user **explicitly** asks you to.

**Protocol**:
- Generate LaTeX content directly in the chat response.
- Only write code files (`.py`, `.sh`, `.yaml`, `.md`) to the disk.
- Keep the repository clean of temporary write-ups.
