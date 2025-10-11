# Project Memory

## User Preferences
- Always use `uv run` for executing Python scripts instead of `python` or `python3`
- Prefer absolute paths over relative paths when possible
- llmhalluc is a local package - do not install it, use PYTHONPATH instead

## Project Structure
- Halluc project contains multiple subdirectories:
  - LLaMA-Factory/ - LLaMA fine-tuning framework
  - lm-evaluation-harness/ - Language model evaluation toolkit
  - configs/ - Configuration files
  - bash/ - Bash scripts for automation
  - docs/ - Documentation
  - llmhalluc/ - Main project code

## Git Configuration
- Remote repository: https://github.com/Tianyi-Billy-Ma/Halluc.git
- Main branch: main
- Successfully pushed initial commit with all project files

## LM-Evaluation-Harness
- Located in `/users/tma2/Projects/Halluc/lm-evaluation-harness/`
- Uses YAML configuration files for task definitions
- Supports various metrics including F1, accuracy, exact_match, etc.
- F1 metric can be configured with different aggregation methods
- CLI arguments are defined in `lm_eval/__main__.py` in the `setup_parser()` function
- Thinking functionality is already implemented with `think_end_token` and `enable_thinking` parameters

## Backtracking Dataset Processing
- Main script: `llmhalluc/scripts/process_backtrack_dataset.py`
- Run with: `PYTHONPATH=/users/tma2/Projects/Halluc:$PYTHONPATH uv run python llmhalluc/scripts/process_backtrack_dataset.py`
- Located in `llmhalluc/data/gsm8k.py` - GSM8KSymbolicDatasetConverter class
- Purpose: Create training data that teaches models to self-correct using backtrack tokens
- **Output behavior**: By default, pushes processed dataset to HuggingFace Hub (no local files created)
- Message "No files have been modified since last commit" is normal - it means HF Hub has no changes to commit
- Algorithm flow:
  1. Tokenize both symbolic (correct) and original responses
  2. Find matching segments using `cs_alg()` from `llmhalluc/utils/alg_utils.py`
  3. Randomly select divergence points between matching segments
  4. At chosen points: follow original path → insert backtrack tokens → continue with symbolic path
  5. Verify by simulating backtrack execution (backtrack tokens delete previous tokens)
- Key function: `cs_alg()` returns only matching segments as anchor points (gaps are divergences)
- Edge cases handled:
  - Identical responses: returns original without backtracking
  - Empty matching segments: initializes with boundary pairs
  - No divergence chosen: ensures at least one is selected
  - Backtrack token validation: must encode to exactly one token ID
- Important: Always use `is None` check for backtrack_idx, not truthiness check
- Important: Calculate backtrack_idx using cumulative length, not segment length



