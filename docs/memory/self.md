# Self Memory

## Git Operations

### Successfully pushed Halluc project to remote
- Added all untracked files including LLaMA-Factory, lm-evaluation-harness, configs, bash scripts, and documentation
- Committed with descriptive message: "Initial commit: Add all project files including LLaMA-Factory, lm-evaluation-harness, configs, bash scripts, and documentation"
- Successfully pushed to origin/main branch
- Remote repository: https://github.com/Tianyi-Billy-Ma/Halluc.git

## Script Modifications

### Enhanced init_env.bash script
- Added environment name parameter with default value "llamafactory"
- Implemented automatic environment activation for both conda and virtual environments
- Added comprehensive error handling and informative output messages
- Script now supports: `./init_env.bash [env_name]` where env_name defaults to "llamafactory"

### Created log_yaml.sh script
- Simple script to display YAML file contents using cat
- Requires YAML file path as argument (no default value)
- Includes error handling for missing files and helpful usage information
- Lists available YAML files when file not found or no argument provided
- Usage: `./log_yaml.sh yaml_file_path`

### Fixed init_env.sh syntax error
- Issue: Missing closing `fi` statement for the `if command -v conda` block
- Error: "syntax error: unexpected end of file" on line 26
- Fix: Added missing `fi` statement at the end of the script to properly close the if block
- Script now has proper bash syntax and should execute without errors

## LLaMA-Factory Training Errors

### tqdm ZeroDivisionError in distributed training
- Issue: ZeroDivisionError in tqdm progress bar during dataset loading in distributed training
- Error: `ZeroDivisionError: integer division or modulo by zero` at line 253 in tqdm/std.py
- Root cause: The `nsyms` variable (number of symbols) becomes zero, causing division by zero in bar formatting
- Context: Occurs during dataset mapping in LLaMA-Factory when loading GSM8K dataset
- Location: `/users/tma2/miniconda3/envs/llamafactory/lib/python3.11/site-packages/tqdm/std.py:253`
- **LIKELY CAUSE**: `TQDM_ASCII=1` environment variable can cause issues in distributed training environments
- Potential fixes:
  1. **Remove TQDM_ASCII=1**: `unset TQDM_ASCII` or don't set it
  2. Set environment variable to disable progress bars: `TQDM_DISABLE=1`
  3. Update tqdm to latest version
  4. Use `disable=True` parameter in tqdm calls
  5. Check for terminal width issues in distributed environment

## LM-Evaluation-Harness Knowledge

### F1 Metric Configuration
- F1 is a natively supported metric in lm-evaluation-harness
- Can be used with different aggregation functions: `mean`, `f1`, or custom functions
- Common usage patterns:
  - `aggregation: mean` - simple mean aggregation (CORRECT for generate_until tasks)
  - `aggregation: f1` - uses sklearn's f1_score with max value
  - `aggregation: f1_micro` - micro-averaged F1 for multi-class classification (WRONG for generate_until)
  - `aggregation: !function utils.weighted_f1_score` - custom weighted F1
- F1 works with both `multiple_choice` and `generate_until` output types
- **CRITICAL**: For `generate_until` tasks, use `aggregation: mean`, NOT `f1_micro`
- `f1_micro` is designed for classification tasks with multiple classes, not text generation
- Can be combined with other metrics like `acc` in the same metric_list
- Supports additional parameters like `ignore_case`, `ignore_punctuation`, `regexes_to_ignore`

### F1 vs Exact Match Filter Behavior
- **CRITICAL DIFFERENCE**: F1 and exact_match handle filters differently
- **exact_match**: Uses `filtered_resps` (post-filter output) for evaluation
- **f1**: Uses `resps` (raw model output) for evaluation - ignores filters
- **Root cause**: F1 is retrieved from HF Evaluate library, not native lm-eval metric
- **Impact**: For tasks like GSM8K with answer extraction filters:
  - exact_match compares extracted number vs target number
  - f1 compares full reasoning text vs full target text
- **Solutions**:
  1. Use custom F1 implementation that respects filters
  2. Remove F1 metric if only final answer matters
  3. Accept current behavior if reasoning quality evaluation is desired

### Save File Functions
- **Primary save functions** are located in `lm_eval/loggers/evaluation_tracker.py`:
  - `save_results_aggregated()` - saves aggregated evaluation results to JSON files
  - `save_results_samples()` - saves per-sample results to JSONL files using append mode ("a")
- **Cache functions** in `lm_eval/caching/cache.py`:
  - `save_to_cache()` - saves objects to pickle cache files using dill
  - `load_from_cache()` - loads cached objects
- **Task-specific save functions**:
  - `save_results()` in `lm_eval/tasks/noreval/ask_gec/errant.py` - saves evaluation results
  - Various `_generate_configs.py` files use `with open()` for YAML config generation
- **File formats**: JSON for aggregated results, JSONL for samples, pickle for cache, YAML for configs
- **Append mode usage**: JSONL files use append mode ("a") in `save_results_samples()` at line 337

### LM-Evaluation-Harness CLI Arguments
- **CLI argument definition**: Located in `lm_eval/__main__.py` in the `setup_parser()` function
- **Thinking functionality**: Already implemented in model classes with `think_end_token` and `enable_thinking` parameters
- **Adding new CLI args**: Add to `setup_parser()` function around line 293, then pass to `evaluator.simple_evaluate()` around line 459-487
- **Existing thinking support**: HFLM class supports `think_end_token` (str/int) and `enable_thinking` (bool) parameters
- **Post-processing**: `postprocess_generated_text()` in `lm_eval/models/utils.py` handles stripping thinking content

## Algorithm Fixes

### cs_alg function in llmhalluc/utils/alg_utils.py
- **Issue**: Original implementation collected ALL common subarrays including overlapping ones, causing index logic errors downstream
- **Problem**: The function returned overlapping segments which broke the assumption in gsm8k.py that segments are sequential
- **Correct behavior**: Function should return ONLY the matching segments (anchor points), not diverging segments
- **Solution**: Use Python's `difflib.SequenceMatcher` to find matching blocks efficiently
- **Example**: For A=[1,2,3,4,5] and B=[1,9,8,4,5], should return ([(0,1), (3,5)], [(0,1), (3,5)]) - only matches
- **Key insight**: The gaps between matching segments are the diverging regions, which gsm8k.py handles separately

### GSM8KSymbolicDatasetConverter edge cases in llmhalluc/data/gsm8k.py
- **Bug #1**: Line 134 used `if not backtrack_idx:` which fails when `backtrack_idx=0`. **Fix**: Use `if backtrack_idx is None:`
- **Bug #2**: Line 135 calculated `backtrack_idx = len(modified_sub_ids)` incorrectly (should be cumulative). **Fix**: Use `len(modified_token_ids) + len(modified_sub_ids)`
- **Bug #3**: Line 143 accessed `sym_next_start` outside loop causing `NameError`. **Fix**: Move inside loop, use `sym_lcs_pairs[-1][0]` for final segment
- **Edge case #1**: Identical responses cause `num_candidates=0`, leading to `np.random.choice` error. **Fix**: Early return when identical
- **Edge case #2**: Empty cs_alg results cause index errors. **Fix**: Initialize with boundary pairs `[(0,0), (len,len)]` if empty
- **Edge case #3**: All `chosen_candidates` are False leaves `backtrack_idx=None`. **Fix**: Ensure at least one divergence is chosen
- **Edge case #4**: Backtrack token encoding to multiple tokens breaks logic. **Fix**: Validate in `__post_init__` that it encodes to exactly one token
- **Best practice**: Enhanced assertion messages with actual vs expected values for better debugging


