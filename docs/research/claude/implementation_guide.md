# Implementation Guide: Priority Actions for Backtrack Training

**Author**: Claude (Implementation Guide)  
**Date**: January 9, 2026  
**Project**: Halluc - Backtrack Token Training for LLMs

---

## Overview

This document provides step-by-step implementation instructions for the highest-priority fixes. Follow these in order for maximum impact with minimum effort.

---

## Priority 1: Masked-Error SFT

### Step 1.1: Modify Data Converter

**File**: `llmhalluc/data/backtrack.py`

Add tracking of error token indices:

```python
@dataclass
class BacktrackDatasetConverter(DatasetConverter):
    """Converter for backtrack dataset processing."""
    
    tokenizer: PreTrainedTokenizer
    max_tokens: int = 10
    no_spc_vocab: list[int] | None = None
    split: str = "train"
    key_mapping: dict[str, str] | None = None
    
    # NEW: Return loss mask info
    return_loss_mask_info: bool = True

    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        """Convert an example to backtrack format with loss masking info."""
        
        # ... existing code for prompt, query, response extraction ...
        
        # Tokenize response
        response_token_ids = self.tokenizer.encode(response, add_special_tokens=False)
        backtrack_id = self.tokenizer.encode(BACKTRACK_TOKEN, add_special_tokens=False)[0]
        
        random_int = (
            np.random.randint(0, self.max_tokens) if self.split == "train" else 0
        )
        
        if random_int == 0:
            return {
                "prompt": prompt,
                "query": query,
                "original_response": response,
                "response": response,
                "backtrack_content": "",
                # NEW: No masking needed
                "error_token_count": 0,
                "error_start_offset": 0,
            }
        
        # Generate backtrack content
        random_split = np.random.randint(0, len(response_token_ids))
        np.random.shuffle(self.no_spc_vocab)
        
        error_tokens = self.no_spc_vocab[:random_int]
        
        backtrack_token_ids = (
            response_token_ids[:random_split] + error_tokens
        )
        
        curr_response_token_ids = [backtrack_id] * random_int + response_token_ids[
            random_split:
        ]
        
        # Decode modified content
        backtrack_content = self.tokenizer.decode(backtrack_token_ids)
        modified_response = self.tokenizer.decode(curr_response_token_ids)
        
        return {
            "prompt": prompt,
            "query": query,
            "response": modified_response,
            "backtrack_content": backtrack_content,
            # NEW: Info for loss masking
            "error_token_count": random_int,
            "error_start_offset": random_split,  # Offset in response where errors start
            "prefix_token_count": random_split,  # Tokens before error
        }
```

### Step 1.2: Create Custom Data Collator

**File**: `llmhalluc/data/collators.py` (new file)

```python
"""Custom data collators for backtrack training."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorMixin


@dataclass
class BacktrackDataCollator(DataCollatorMixin):
    """
    Data collator that masks error tokens in the loss computation.
    
    This prevents the model from learning to generate error tokens,
    while still training it to recognize errors and generate backtrack tokens.
    """
    
    tokenizer: PreTrainedTokenizerBase
    mlm: bool = False
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate features and create masked labels.
        """
        # First, do standard collation
        batch = self._collate_standard(features)
        
        # Then, mask error tokens in labels
        batch = self._mask_error_tokens(features, batch)
        
        return batch
    
    def _collate_standard(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Standard tokenization and padding."""
        
        # Extract text fields
        texts = []
        for feature in features:
            # Construct full text for training
            prompt = feature.get("prompt", "")
            query = feature.get("query", "")
            backtrack_content = feature.get("backtrack_content", "")
            response = feature.get("response", "")
            
            # Combine into full sequence
            # Format: <prompt><query><backtrack_content><response>
            full_text = f"{prompt}{query}{backtrack_content}{response}"
            texts.append(full_text)
        
        # Tokenize
        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors=self.return_tensors,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        
        # Create initial labels (copy of input_ids, with padding masked)
        labels = batch["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        
        return batch
    
    def _mask_error_tokens(
        self, 
        features: List[Dict[str, Any]], 
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Mask error tokens in labels (set to -100).
        
        The model should NOT be trained to generate error tokens.
        Only train on: prompt (optional), backtrack tokens, and correction tokens.
        """
        labels = batch["labels"]
        
        for i, feature in enumerate(features):
            error_count = feature.get("error_token_count", 0)
            
            if error_count == 0:
                continue
            
            # Find where error tokens are in the tokenized sequence
            # This requires knowing the structure of the sequence
            
            backtrack_content = feature.get("backtrack_content", "")
            
            # Tokenize just the backtrack_content to find error token indices
            if backtrack_content:
                backtrack_tokens = self.tokenizer.encode(
                    backtrack_content, 
                    add_special_tokens=False
                )
                
                # The last `error_count` tokens in backtrack_content are the errors
                # Find their positions in the full sequence
                
                # Get full text up to and including backtrack_content
                prompt = feature.get("prompt", "")
                query = feature.get("query", "")
                prefix_text = f"{prompt}{query}{backtrack_content}"
                
                # Tokenize prefix to find end of errors
                prefix_tokens = self.tokenizer.encode(prefix_text, add_special_tokens=False)
                
                # Error tokens are the last `error_count` tokens of the prefix
                error_end_idx = len(prefix_tokens)
                error_start_idx = error_end_idx - error_count
                
                # Mask these positions in labels
                # Account for any special tokens added by tokenizer
                special_token_offset = 0
                if self.tokenizer.bos_token_id is not None:
                    special_token_offset = 1
                
                error_start_idx += special_token_offset
                error_end_idx += special_token_offset
                
                # Clamp to valid range
                seq_len = labels.size(1)
                error_start_idx = max(0, min(error_start_idx, seq_len))
                error_end_idx = max(0, min(error_end_idx, seq_len))
                
                # Set error token labels to -100 (ignored in loss)
                labels[i, error_start_idx:error_end_idx] = -100
        
        batch["labels"] = labels
        return batch
```

### Step 1.3: Integrate with Trainer

**File**: `llmhalluc/train/sft.py`

Modify to use custom collator:

```python
from llmhalluc.data.collators import BacktrackDataCollator

class SFTExecutor(BaseExecutor):
    # ... existing code ...
    
    def setup_trainer(self):
        """Setup SFTTrainer with custom data collator for backtrack training."""
        train_dataset = self.dataset.get("train")
        eval_dataset = self.dataset.get("eval")
        
        # Create custom data collator
        data_collator = BacktrackDataCollator(
            tokenizer=self.tokenizer,
        )
        
        self.trainer = SFTTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,  # Use custom collator
            args=self.args,
        )
```

---

## Priority 2: Semantic Token Initialization

### Step 2.1: Create Initialization Utility

**File**: `llmhalluc/utils/token_init.py` (new file)

```python
"""Utilities for initializing special tokens."""

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
import logging

logger = logging.getLogger(__name__)


def initialize_backtrack_token(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    backtrack_token: str = "<|BACKTRACK|>",
    similar_words: list[str] = None,
) -> None:
    """
    Initialize backtrack token embedding with semantic meaning.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        backtrack_token: The backtrack token string
        similar_words: List of semantically similar words for initialization
    """
    # Default similar words for backtrack concept
    if similar_words is None:
        similar_words = [
            "delete", "remove", "undo", "erase", "back", "cancel",
            "retry", "revert", "reset", "clear", "redo"
        ]
    
    # Get backtrack token ID
    backtrack_id = tokenizer.convert_tokens_to_ids(backtrack_token)
    
    if backtrack_id == tokenizer.unk_token_id:
        logger.warning(f"Backtrack token '{backtrack_token}' not found in vocabulary")
        return
    
    # Get model embeddings
    embeddings = model.get_input_embeddings()
    
    # Find valid similar token IDs
    similar_ids = []
    for word in similar_words:
        # Try the word as-is
        token_id = tokenizer.convert_tokens_to_ids(word)
        if token_id != tokenizer.unk_token_id:
            similar_ids.append(token_id)
        
        # Also try subword tokens
        subword_ids = tokenizer.encode(word, add_special_tokens=False)
        for sid in subword_ids:
            if sid != tokenizer.unk_token_id:
                similar_ids.append(sid)
    
    # Remove duplicates
    similar_ids = list(set(similar_ids))
    
    with torch.no_grad():
        if similar_ids:
            # Average embeddings of similar tokens
            similar_embeddings = embeddings.weight[similar_ids]
            mean_embedding = similar_embeddings.mean(dim=0)
            
            # Set backtrack token embedding
            embeddings.weight[backtrack_id] = mean_embedding
            
            logger.info(
                f"Initialized '{backtrack_token}' from {len(similar_ids)} similar tokens"
            )
        else:
            # Fallback: use mean of all embeddings
            mean_embedding = embeddings.weight.mean(dim=0)
            embeddings.weight[backtrack_id] = mean_embedding
            
            logger.warning(
                f"No similar tokens found. Initialized '{backtrack_token}' with mean embedding"
            )
    
    # Optionally: also initialize output embeddings if not tied
    if hasattr(model, 'lm_head') and not model.config.tie_word_embeddings:
        output_embeddings = model.get_output_embeddings()
        with torch.no_grad():
            output_embeddings.weight[backtrack_id] = mean_embedding
        logger.info("Also initialized output embedding")


def add_and_initialize_backtrack_token(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    backtrack_token: str = "<|BACKTRACK|>",
) -> None:
    """
    Add backtrack token to tokenizer and initialize its embedding.
    
    This is a convenience function that handles the full process.
    """
    # Add special token if not already present
    if backtrack_token not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({
            "additional_special_tokens": [backtrack_token]
        })
        
        # Resize model embeddings
        model.resize_token_embeddings(len(tokenizer))
        
        logger.info(f"Added '{backtrack_token}' to vocabulary")
    
    # Initialize embedding
    initialize_backtrack_token(model, tokenizer, backtrack_token)
```

### Step 2.2: Integrate with Model Setup

**File**: `llmhalluc/train/base.py`

Add initialization after model loading:

```python
from llmhalluc.utils.token_init import add_and_initialize_backtrack_token

class BaseExecutor(ABC):
    # ... existing code ...
    
    def setup_model(self):
        model_kwargs = {
            "low_cpu_mem_usage": True,
            "trust_remote_code": getattr(self.args, "trust_remote_code", True),
        }
        if getattr(self.args, "bf16", False):
            model_kwargs["torch_dtype"] = torch.bfloat16
        elif getattr(self.args, "fp16", False):
            model_kwargs["torch_dtype"] = torch.float16

        self.model = get_model(self.args.model_name_or_path, **model_kwargs)
        
        # NEW: Initialize backtrack token after model loading
        self._initialize_special_tokens()
    
    def _initialize_special_tokens(self):
        """Initialize any special tokens needed for training."""
        # Check if backtrack training is enabled
        if getattr(self.args, "use_backtrack", False):
            add_and_initialize_backtrack_token(
                self.model,
                self.tokenizer,
                backtrack_token="<|BACKTRACK|>"
            )
```

---

## Priority 3: Curriculum Learning

### Step 3.1: Create Curriculum Dataset Wrapper

**File**: `llmhalluc/data/curriculum.py` (new file)

```python
"""Curriculum learning utilities for backtrack training."""

from dataclasses import dataclass
from typing import Optional
import logging

from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning."""
    
    # Maximum backtrack tokens at each phase
    phase_1_max_tokens: int = 2   # Epochs 0-2
    phase_2_max_tokens: int = 5   # Epochs 3-5
    phase_3_max_tokens: int = 10  # Epochs 6+
    
    # Phase boundaries (by epoch)
    phase_1_end_epoch: int = 3
    phase_2_end_epoch: int = 6


class CurriculumBacktrackDataset(Dataset):
    """
    Dataset wrapper that filters examples based on curriculum phase.
    """
    
    def __init__(
        self,
        base_dataset: Dataset,
        config: Optional[CurriculumConfig] = None,
    ):
        self.base_dataset = base_dataset
        self.config = config or CurriculumConfig()
        self.current_epoch = 0
        
        # Build index of valid examples for each phase
        self._build_indices()
    
    def _build_indices(self):
        """Pre-compute which examples are valid at each max_token level."""
        self.indices_by_max_tokens = {}
        
        for max_tokens in [
            self.config.phase_1_max_tokens,
            self.config.phase_2_max_tokens,
            self.config.phase_3_max_tokens,
        ]:
            valid_indices = []
            for i in range(len(self.base_dataset)):
                example = self.base_dataset[i]
                error_count = example.get("error_token_count", 0)
                if error_count <= max_tokens:
                    valid_indices.append(i)
            
            self.indices_by_max_tokens[max_tokens] = valid_indices
        
        logger.info(f"Curriculum index built: {len(self.indices_by_max_tokens)} phases")
    
    def set_epoch(self, epoch: int):
        """Set current epoch for curriculum progression."""
        self.current_epoch = epoch
        logger.info(f"Curriculum epoch set to {epoch}, max_tokens={self.get_max_tokens()}")
    
    def get_max_tokens(self) -> int:
        """Get maximum backtrack tokens for current epoch."""
        if self.current_epoch < self.config.phase_1_end_epoch:
            return self.config.phase_1_max_tokens
        elif self.current_epoch < self.config.phase_2_end_epoch:
            return self.config.phase_2_max_tokens
        else:
            return self.config.phase_3_max_tokens
    
    def _get_current_indices(self):
        """Get valid indices for current curriculum phase."""
        max_tokens = self.get_max_tokens()
        return self.indices_by_max_tokens[max_tokens]
    
    def __len__(self):
        return len(self._get_current_indices())
    
    def __getitem__(self, idx):
        real_idx = self._get_current_indices()[idx]
        return self.base_dataset[real_idx]


class CurriculumCallback:
    """
    Callback for updating curriculum phase during training.
    
    Use this with HuggingFace Trainer.
    """
    
    def __init__(self, dataset: CurriculumBacktrackDataset):
        self.dataset = dataset
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        """Update curriculum at start of each epoch."""
        self.dataset.set_epoch(state.epoch)
```

### Step 3.2: Integrate with Training

**File**: `llmhalluc/train/sft.py`

```python
from llmhalluc.data.curriculum import CurriculumBacktrackDataset, CurriculumCallback

class SFTExecutor(BaseExecutor):
    
    def setup_dataset(self):
        # ... existing dataset loading code ...
        
        # Wrap with curriculum if enabled
        if getattr(self.args, "use_curriculum", False):
            self.dataset["train"] = CurriculumBacktrackDataset(
                self.dataset["train"]
            )
            self.curriculum_callback = CurriculumCallback(self.dataset["train"])
    
    def setup_trainer(self):
        # ... existing code ...
        
        # Add curriculum callback
        callbacks = []
        if hasattr(self, "curriculum_callback"):
            callbacks.append(self.curriculum_callback)
        
        self.trainer = SFTTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            args=self.args,
            callbacks=callbacks,  # Add callbacks
        )
```

---

## Priority 4: KV-Cache Rewinding Inference

### Step 4.1: Create Custom Generation Function

**File**: `llmhalluc/inference/backtrack_generate.py` (new file)

```python
"""Backtrack-aware generation with KV-cache rewinding."""

from typing import Optional, List, Tuple
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer


def rewind_kv_cache(
    past_key_values: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
    num_tokens: int = 1,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
    """
    Remove the last N tokens from the KV cache.
    
    Args:
        past_key_values: Tuple of (key, value) for each layer.
                        Each tensor has shape (batch, heads, seq_len, head_dim)
        num_tokens: Number of tokens to remove
    
    Returns:
        Rewound past_key_values
    """
    if past_key_values is None:
        return None
    
    new_past = []
    for layer_past in past_key_values:
        key, value = layer_past
        
        # Remove last num_tokens from sequence dimension
        new_key = key[:, :, :-num_tokens, :]
        new_value = value[:, :, :-num_tokens, :]
        
        new_past.append((new_key, new_value))
    
    return tuple(new_past)


def generate_with_backtrack(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    max_length: int = 512,
    max_backtracks: int = 50,  # Prevent infinite backtrack loops
    backtrack_token: str = "<|BACKTRACK|>",
    temperature: float = 1.0,
    do_sample: bool = False,
    top_p: float = 1.0,
    top_k: int = 50,
) -> Tuple[str, List[dict]]:
    """
    Generate text with physical backtracking via KV-cache rewinding.
    
    When the model generates a backtrack token, we:
    1. Remove the backtrack token from output
    2. Remove the previous token from output
    3. Rewind the KV-cache by 2 positions
    4. Continue generation from the rewound state
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: Input prompt
        max_length: Maximum tokens to generate
        max_backtracks: Maximum number of backtracks allowed
        backtrack_token: The backtrack token string
        temperature: Sampling temperature
        do_sample: Whether to sample or use greedy decoding
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
    
    Returns:
        Tuple of (generated_text, generation_log)
    """
    device = model.device
    
    # Tokenize prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    prompt_length = input_ids.shape[1]
    
    # Get backtrack token ID
    backtrack_id = tokenizer.convert_tokens_to_ids(backtrack_token)
    eos_id = tokenizer.eos_token_id
    
    # Track generation
    generated_tokens = []
    past_key_values = None
    backtrack_count = 0
    generation_log = []
    
    for step in range(max_length):
        # Forward pass
        outputs = model(
            input_ids=input_ids[:, -1:] if past_key_values else input_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
        
        past_key_values = outputs.past_key_values
        logits = outputs.logits[:, -1, :]
        
        # Apply temperature
        if temperature != 1.0:
            logits = logits / temperature
        
        # Sample or greedy
        if do_sample:
            # Apply top-k
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(logits, top_k)
                logits = torch.full_like(logits, float('-inf'))
                logits.scatter_(1, top_k_indices, top_k_logits)
            
            # Apply top-p
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    torch.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')
            
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        
        next_token_id = next_token.item()
        
        # Log step
        log_entry = {
            "step": step,
            "token_id": next_token_id,
            "token": tokenizer.decode([next_token_id]),
            "action": "generate"
        }
        
        # Check for backtrack
        if next_token_id == backtrack_id:
            log_entry["action"] = "backtrack"
            
            if len(generated_tokens) > 0 and backtrack_count < max_backtracks:
                # Remove last generated token
                removed_token = generated_tokens.pop()
                log_entry["removed_token_id"] = removed_token
                log_entry["removed_token"] = tokenizer.decode([removed_token])
                
                # Rewind KV cache by 2 (backtrack + removed token)
                past_key_values = rewind_kv_cache(past_key_values, num_tokens=2)
                
                # Update input_ids to reflect removal
                input_ids = input_ids[:, :-1]
                
                backtrack_count += 1
                generation_log.append(log_entry)
                continue
            else:
                # Can't backtrack (nothing to remove or max reached)
                log_entry["action"] = "backtrack_failed"
                generation_log.append(log_entry)
                continue
        
        # Check for EOS
        if next_token_id == eos_id:
            generation_log.append(log_entry)
            break
        
        # Normal token - add to generated
        generated_tokens.append(next_token_id)
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        generation_log.append(log_entry)
    
    # Decode final output
    output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return output_text, generation_log


# Example usage
if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
    
    prompt = "Question: What is 2 + 2?\nAnswer:"
    
    output, log = generate_with_backtrack(
        model, tokenizer, prompt,
        max_length=100,
        temperature=0.7,
        do_sample=True
    )
    
    print(f"Output: {output}")
    print(f"\nGeneration log:")
    for entry in log:
        print(entry)
```

---

## Quick Start Checklist

### Day 1: Essential Fixes

- [ ] Create `llmhalluc/data/collators.py` with `BacktrackDataCollator`
- [ ] Modify `llmhalluc/data/backtrack.py` to track error positions
- [ ] Modify `llmhalluc/train/sft.py` to use custom collator
- [ ] Create `llmhalluc/utils/token_init.py` with initialization utility
- [ ] Integrate token initialization in `llmhalluc/train/base.py`

### Day 2: Training Improvements

- [ ] Create `llmhalluc/data/curriculum.py` with curriculum learning
- [ ] Add curriculum callback to trainer
- [ ] Run training with masked-error SFT

### Day 3: Inference Improvements

- [ ] Create `llmhalluc/inference/backtrack_generate.py`
- [ ] Test inference with KV-cache rewinding
- [ ] Run evaluation

---

## Configuration Examples

### Training Config (`configs/backtrack_sft.yaml`)

```yaml
# Model
model_name_or_path: meta-llama/Llama-3.1-8B
tokenizer_name_or_path: meta-llama/Llama-3.1-8B

# Dataset
dataset: gsm8k_backtrack
converter: backtrack

# Training
output_dir: ./outputs/backtrack_sft
num_train_epochs: 10
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
learning_rate: 2e-5
warmup_ratio: 0.1
bf16: true

# Backtrack-specific
use_backtrack: true
use_curriculum: true

# Logging
logging_steps: 10
save_steps: 500
eval_steps: 500
```

---

## Expected Results

After implementing these fixes:

| Metric | Before | After (Expected) |
|--------|--------|------------------|
| GSM8K Accuracy | ~45% | 55-65% |
| Backtrack Usage | Random | Contextual |
| Training Stability | Poor | Stable |
| Correction Quality | Low | High |

---

## Next Steps

After implementing these priority fixes:

1. **Monitor training logs** for convergence
2. **Evaluate on GSM8K** with and without backtracking
3. **Analyze generation logs** to understand backtrack patterns
4. **Iterate on curriculum** based on results
5. **Consider advanced techniques** (PRM, MCTS) if plateau reached

For detailed implementation of advanced techniques, see `novel_techniques.md`.
