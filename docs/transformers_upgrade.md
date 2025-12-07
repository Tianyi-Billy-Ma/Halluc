# Updating LLaMA-Factory to Newer Hugging Face `transformers`

## Context
- `LLaMA-Factory/requirements.txt` currently pins `transformers>=4.49.0,<=4.57.1` to preserve API compatibility with our PPO stack and the surrounding RLHF tooling.
- The PPO integration directly subclasses both Hugging Face `Trainer` and TRL's PPO trainer, so breaking changes in either project ripple into our fork.
```1:39:/users/tma2/Projects/LLaMA-Factory/src/llamafactory/train/ppo/trainer.py
# ... inspired by TRL v0.8.0
from transformers import GenerationConfig, Trainer
from trl import PPOConfig, PPOTrainer
```
- Upgrading `transformers` typically requires coordinating bumps for `accelerate`, `peft`, `trl`, and sometimes `safetensors`. Treat the upgrade as a bundled effort.

## Upgrade Strategy
1. **Pick a target release window.** Decide which `transformers` minor version you need (e.g., `4.59.x`). Review its release notes for `Trainer`, `GenerationConfig`, and tokenizer changes.
2. **Mirror upstream LLaMA-Factory support.** Check the upstream repo to see which commit already supports your target version. Rebasing onto that commit is usually easier than hand-porting dozens of fixes.
3. **Relax dependency pins in stages.** Bump `transformers` first, then align `accelerate`, `peft`, and `trl` to the versions officially supported by that upstream commit.
4. **Patch local customizations.** Revisit files we override (e.g., `src/llamafactory/train/ppo/*`, custom collators, dataset loaders) to ensure new APIs are respected.
5. **Regenerate lockfiles/environments** (`uv lock`, `uv sync`, `conda env update`) and re-run smoke tests before rolling into long training jobs.

## Detailed Steps
### 1. Inventory the current surface
- Record the versions in `LLaMA-Factory/requirements.txt` and any environment files.
- Note custom integrations that touch `transformers` internals:
  - Custom PPO trainer and utilities in `src/llamafactory/train/ppo/`.
  - Trainer mixins (`trainer_utils.py`, `callbacks/`, `extras/`).
  - Generation helpers that rely on specific `GenerationConfig` or `logits_processor` behaviors.

### 2. Choose the upgrade baseline
- If upstream already supports the target version, cherry-pick or merge those commits first.
- Otherwise, create an internal branch, tag the current working state, and outline the API gaps you must close (see "Compatibility considerations").

### 3. Update dependency pins
- Edit `LLaMA-Factory/requirements.txt` with the new version ranges for:
  - `transformers`
  - `accelerate`
  - `peft`
  - `trl` (≥0.10.x if you need the latest `Trainer` contract changes)
  - `safetensors` (newer `transformers` may require ≥0.5.4)
- Regenerate `uv.lock` or the relevant lockfile to ensure downstream consumers get the exact versions.
- Announce the change in `docs/AGENTS.md` so everyone knows to rebuild their environments.

### 4. Refactor our PPO + Trainer glue
- If moving to TRL ≥0.10.x, update `CustomPPOTrainer` so it extends the new `trl.trainer.ppo_trainer.PPOTrainer` signature and logging hooks.
- Review `PPOTrainer.__init__` call sites—later TRL versions expect `dataset=None` plus manual dataloaders, so reuse their new helpers where practical.
- Validate the `unwrap_model_for_generation` import path (`trl.models.utils`) because it was relocated in TRL 0.10.

### 5. Align Hugging Face `Trainer` usage
- Confirm that callbacks, `TrainerState`, and checkpoint utilities we import still live under the same modules.
- Revisit generation code paths: newer `transformers` may change default EOS handling, attention masks, or KV-cache flags.
- Update any deprecated kwargs (e.g., `use_cache`, `pad_token_id` behavior) flagged by `transformers` warnings.

### 6. Rebuild and test environments
- Recreate the `llamafactory` conda env or run `uv sync --extra gpu` (or whichever extra you use) to install the new stack.
- Run:
  1. `python llamafactory/train/sft/trainer.py --help` (ensures CLI wiring still imports).
  2. A short supervised fine-tuning dry run (`scripts/test.py` or similar) on a single GPU.
  3. PPO smoke test with a tiny dataset to confirm TRL integration still works.
  4. Any custom evaluation harnesses you rely on (especially if you upgraded tokenizers).

### 7. Communicate and iterate
- Document the new minimum versions, migration notes, and known issues in `README.md` or this doc.
- Encourage teammates to wipe cached checkpoints built with older safetensor formats if loading fails.

## Compatibility Considerations
- **TRL:** `trl>=0.10` aligns with `transformers>=4.38`, but it breaks our dual-inheritance trick. Factor time to adopt the new trainer base classes.
- **Accelerate:** Keep `accelerate` within the range endorsed by both `transformers` and `trl`. DeepSpeed/FSDP plugins often lag one release behind.
- **PEFT:** Check whether LoRA/Q-LoRA modules changed signature (`AutoPeftModelForCausalLM`) and update import paths accordingly.
- **Tokenizers:** Regenerate tokenizer JSONs if the new `transformers` version expects updated `added_tokens_decoder` fields.
- **Inference scripts:** Validate `GenerationConfig` serialization/deserialization for saved checkpoints so legacy runs do not crash.

## Validation Checklist
- [ ] `python -m compileall LLaMA-Factory/src` (fast syntax sanity check)
- [ ] One SFT epoch on a toy dataset
- [ ] One PPO mini-run hitting reward model + reference model swapping
- [ ] `bash/bash/crc/e2e.sh` dry run to ensure orchestration scripts pass new env vars
- [ ] Regression test any custom callbacks (value head fix, processor saver, etc.)

## Rollback Plan
1. Tag the repo before the upgrade (`git tag transformers-pre-upgrade`).
2. Keep the old `uv.lock` or `requirements.txt` in a `rollback/` folder for quick restore.
3. If training jobs fail after the upgrade, revert the dependency pin commit, redeploy the old environment, and file an issue detailing the failure so we can patch forward safely.