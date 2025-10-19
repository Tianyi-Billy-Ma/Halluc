# Large Language Model Training and Evaluation 

## Overview 
This project contains the source code for training and evaluation of large language models.


## Project Specific Standards

- You should not change anything under the folders `LLaMA-Factory` and `lm-evaluation-harness` as these are thrid-party packages. But if needed, you should inform me and I will update the packages. 
- For python codes, you should put under the folder `llmhalluc`.
- After implementation, you should [Recent Changes](#recent-changes) to reflect the recent changes (Keep the order of the changes and should only keep five recent changes). Moreover, if all the tasks are completed, you should change the [Current Tasks](#current-tasks) to "N/A".

## Standard Git Workflow

- **Branch from `main`** with a descriptive name: `feature/<short-description>` or `bugfix/<short-description>`.
- **Install/update dependencies** with `uv sync` at current working directory.
- **Develop incrementally**, keeping commits focused; use `ruff format` and `ruff check --fix` before staging changes.
- **Run validation**: `pytest` (or targeted `pytest tests/<suite>`), plus any scenario-specific scripts in `tests/scripts/`.
- **Review diffs** with `git status` / `git diff`, then commit using conventional messages: `git commit -m "feat: add browser HTML cache"`.
- **Push and open a PR** once CI passes locally: `git push --set-upstream origin feature/<short-description>`; include testing notes in the PR description.

## Environment Setup

We use conda to manage the python environment. Specifically, we use the environment `llamafactory` for the LLaMA-Factory and `lm_eval` for the lm-evaluation-harness. 

## Recent Changes

- Implemented `llmhalluc/run_eval.py` with YAML configuration support for lm_eval evaluations. The script supports both YAML configs and CLI arguments with proper override behavior.
- Created comprehensive configuration files:
  - `configs/lm_eval/run/default.yaml` - Default configuration template with all parameters
  - `configs/lm_eval/run/example_gsm8k.yaml` - Example run configuration for GSM8K
  - `configs/lm_eval/run/test_run.yaml` - Test configuration for quick validation
  - `configs/lm_eval/tasks/gsm8k_custom.yaml` - Custom task definition template
- Automatic inclusion of `configs/lm_eval/tasks/` directory for custom task definitions.
- Tested and validated: YAML config loading, CLI override, custom task discovery, and end-to-end evaluation runs.
- Setup the project and relevant packages.

## Current Tasks

N/A

## Next Steps

- Add unit coverage for `llmhalluc/scripts/run_lm_eval.py` using a stub dataset.
- Extend `docs/lm_eval/README.md` with usage instructions for the new runner.
- Integrate backtrack-aware decoding logic ahead of GSM8K evaluations.
