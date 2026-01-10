# Novel Techniques for Backtrack Token Training

**Author**: Claude (Research & Implementation Guide)  
**Date**: January 9, 2026  
**Project**: Halluc - Backtrack Token Training for LLMs

---

## Executive Summary

This document presents **10 novel techniques** to solve the backtrack token training problem, ranging from quick fixes to advanced research directions. Each technique includes theoretical motivation, implementation details, and expected outcomes.

---

## Technique Overview

| # | Technique | Difficulty | Impact | Recommended Order |
|---|-----------|-----------|--------|-------------------|
| 1 | Masked-Error SFT | Easy | High | 🥇 First |
| 2 | Semantic Token Initialization | Easy | Medium | 🥇 First |
| 3 | Curriculum Learning | Easy | Medium | 🥈 Second |
| 4 | Hard Backtrack Attention Mask | Medium | High | 🥈 Second |
| 5 | Self-Generated Error Data | Medium | High | 🥉 Third |
| 6 | Token-Level DPO | Medium | High | 🥉 Third |
| 7 | Process Reward Model (PRM) | Hard | Very High | Advanced |
| 8 | MCTS-Based Data Generation | Hard | Very High | Advanced |
| 9 | KV-Cache Rewinding Inference | Medium | High | Inference |
| 10 | Verifier Head Architecture | Hard | Very High | Research |

---

## Technique 1: Masked-Error SFT 🥇

### Concept
Do NOT train the model to generate error tokens. Only train it to:
1. **Detect** errors (generate backtrack token)
2. **Correct** errors (generate correct continuation)

### Implementation

```python
# In your DataCollator or preprocessing

def create_masked_labels(input_ids, error_start, error_end, backtrack_start):
    """
    Create labels where error tokens are masked (set to -100).
    
    Args:
        input_ids: Full tokenized sequence
        error_start: Start index of error tokens
        error_end: End index of error tokens  
        backtrack_start: Start index of backtrack tokens
    
    Returns:
        labels: Same as input_ids, but error tokens set to -100
    """
    labels = input_ids.clone()
    
    # Mask error tokens - model should NOT learn to generate these
    labels[error_start:error_end] = -100
    
    # Optionally mask prompt tokens too
    # labels[:prompt_end] = -100
    
    return labels
```

### Modifications to `backtrack.py`

```python
@dataclass
class BacktrackDatasetConverter(DatasetConverter):
    # ... existing code ...
    
    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        # ... existing processing ...
        
        # NEW: Track indices for loss masking
        result = {
            "prompt": prompt,
            "query": query,
            "response": modified_response,
            "backtrack_content": backtrack_content,
            # NEW FIELDS:
            "error_token_count": random_int,  # Number of error tokens to mask
            "error_position": random_split,    # Where errors were inserted
        }
        return result
```

### Custom Data Collator

```python
from transformers import DataCollatorForLanguageModeling

class BacktrackDataCollator(DataCollatorForLanguageModeling):
    def __call__(self, features):
        batch = super().__call__(features)
        
        # For each example, mask the error tokens in labels
        for i, feature in enumerate(features):
            error_count = feature.get("error_token_count", 0)
            error_pos = feature.get("error_position", 0)
            
            if error_count > 0:
                # Find where error tokens are in the tokenized sequence
                # and set those label positions to -100
                # (Implementation depends on your tokenization scheme)
                pass
        
        return batch
```

### Expected Outcome
- Model stops learning to hallucinate
- Model retains ability to "recover" when errors appear in context
- **Estimated improvement**: 5-15% accuracy gain

---

## Technique 2: Semantic Token Initialization 🥇

### Concept
Initialize the `<|BACKTRACK|>` token embedding meaningfully, not randomly.

### Implementation

```python
def initialize_backtrack_token(model, tokenizer):
    """Initialize backtrack token with semantically meaningful embedding."""
    
    # Get the backtrack token ID
    backtrack_token = "<|BACKTRACK|>"
    backtrack_id = tokenizer.convert_tokens_to_ids(backtrack_token)
    
    # Option 1: Average of semantically similar tokens
    similar_tokens = ["delete", "remove", "undo", "erase", "back", "cancel"]
    similar_ids = [tokenizer.convert_tokens_to_ids(t) for t in similar_tokens]
    similar_ids = [i for i in similar_ids if i != tokenizer.unk_token_id]
    
    if similar_ids:
        # Get embeddings of similar tokens
        embeddings = model.get_input_embeddings()
        similar_embeddings = embeddings.weight[similar_ids]
        
        # Average them
        mean_embedding = similar_embeddings.mean(dim=0)
        
        # Set backtrack token embedding
        with torch.no_grad():
            embeddings.weight[backtrack_id] = mean_embedding
    
    # Option 2: Use mean of all embeddings (safer fallback)
    else:
        embeddings = model.get_input_embeddings()
        mean_embedding = embeddings.weight.mean(dim=0)
        with torch.no_grad():
            embeddings.weight[backtrack_id] = mean_embedding
    
    # Also update output embeddings if tied
    if hasattr(model, 'lm_head'):
        # For models with separate lm_head
        pass  # Usually tied, so this updates automatically
    
    print(f"Initialized {backtrack_token} with semantic embedding")
```

### When to Apply
Call this function **right after** adding the special token and resizing embeddings:

```python
# In model setup
tokenizer.add_special_tokens({"additional_special_tokens": ["<|BACKTRACK|>"]})
model.resize_token_embeddings(len(tokenizer))
initialize_backtrack_token(model, tokenizer)  # <-- ADD THIS
```

### Expected Outcome
- More stable initial training
- Faster convergence
- Reduced risk of catastrophic forgetting
- **Estimated improvement**: 2-5% stability gain

---

## Technique 3: Curriculum Learning 🥈

### Concept
Start with simple backtracking (1-2 tokens) and progressively increase difficulty.

### Implementation

```python
class CurriculumBacktrackDataset:
    def __init__(self, base_dataset, max_epochs=10):
        self.base_dataset = base_dataset
        self.max_epochs = max_epochs
        self.current_epoch = 0
    
    def set_epoch(self, epoch):
        self.current_epoch = epoch
    
    def get_max_backtrack_tokens(self):
        """Increase max backtrack tokens as training progresses."""
        # Epoch 0-2: max 2 tokens
        # Epoch 3-5: max 5 tokens
        # Epoch 6+: max 10 tokens
        if self.current_epoch < 3:
            return 2
        elif self.current_epoch < 6:
            return 5
        else:
            return 10
    
    def __getitem__(self, idx):
        example = self.base_dataset[idx]
        max_tokens = self.get_max_backtrack_tokens()
        
        # Limit backtrack complexity for current epoch
        if example.get("error_token_count", 0) > max_tokens:
            # Re-generate with fewer error tokens
            # Or filter out this example
            pass
        
        return example
```

### Training Loop Integration

```python
for epoch in range(num_epochs):
    train_dataset.set_epoch(epoch)
    trainer.train()
```

### Expected Outcome
- More stable training progression
- Better generalization to complex cases
- **Estimated improvement**: 3-8% accuracy gain

---

## Technique 4: Hard Backtrack Attention Mask 🥈

### Concept
Create custom attention masks where correction tokens **cannot attend** to error tokens.

### Implementation

```python
def create_backtrack_attention_mask(
    seq_length: int,
    error_start: int,
    error_end: int,
    backtrack_start: int,
    backtrack_end: int,
    correction_start: int
) -> torch.Tensor:
    """
    Create attention mask for backtrack training.
    
    Correction tokens can see:
    - Prompt tokens
    - Other correction tokens
    
    Correction tokens CANNOT see:
    - Error tokens
    - Backtrack tokens (optional)
    
    Returns:
        attention_mask: Shape (seq_length, seq_length)
        1 = attend, 0 = don't attend
    """
    # Start with causal mask
    mask = torch.tril(torch.ones(seq_length, seq_length))
    
    # Block correction tokens from seeing error tokens
    for i in range(correction_start, seq_length):
        for j in range(error_start, error_end):
            mask[i, j] = 0
    
    # Optional: Block correction from seeing backtrack tokens
    for i in range(correction_start, seq_length):
        for j in range(backtrack_start, backtrack_end):
            mask[i, j] = 0
    
    return mask

# Visualization:
# Tokens:       [P P P] [E E] [B B] [C C]
# P attends to: [1 1 1   0 0   0 0   0 0]  <- causal
# E attends to: [1 1 1   1 1   0 0   0 0]  <- causal
# B attends to: [1 1 1   1 1   1 1   0 0]  <- causal
# C attends to: [1 1 1   0 0   0 0   1 1]  <- ERROR BLOCKED!
```

### Integration with HuggingFace

```python
class BacktrackAttentionDataCollator:
    def __call__(self, features):
        # ... standard collation ...
        
        # Create custom attention masks
        attention_masks = []
        for feature in features:
            mask = create_backtrack_attention_mask(
                seq_length=len(feature["input_ids"]),
                error_start=feature["error_start"],
                error_end=feature["error_end"],
                backtrack_start=feature["backtrack_start"],
                backtrack_end=feature["backtrack_end"],
                correction_start=feature["correction_start"]
            )
            attention_masks.append(mask)
        
        batch["attention_mask"] = torch.stack(attention_masks)
        return batch
```

### Caveat
This requires modifying how attention is passed to the model. Standard HuggingFace expects 1D attention masks, not 2D. You may need to use the `attention_mask` + `position_ids` pattern or modify the model forward pass.

### Expected Outcome
- Eliminates attention pollution
- Correction quality independent of error content
- **Estimated improvement**: 10-20% accuracy gain

---

## Technique 5: Self-Generated Error Data 🥉

### Concept
Use the model's **own errors** for training, not random tokens.

### Implementation

```python
def generate_self_error_data(model, tokenizer, prompts, num_samples=1000):
    """
    Generate training data from model's actual errors.
    
    1. Generate responses without backtracking
    2. Identify incorrect portions (via verification)
    3. Create backtrack training examples
    """
    training_examples = []
    
    for prompt in prompts:
        # Generate multiple responses
        responses = model.generate(
            prompt,
            num_return_sequences=5,
            do_sample=True,
            temperature=0.7
        )
        
        # Verify each response
        for response in responses:
            verification = verify_response(prompt, response)  # External verifier
            
            if not verification.is_correct:
                # Create backtrack example from actual error
                error_start = verification.error_start_position
                error_tokens = response[error_start:verification.error_end_position]
                correct_continuation = verification.correct_continuation
                
                backtrack_example = {
                    "prompt": prompt,
                    "error_prefix": response[:error_start],
                    "error_tokens": error_tokens,
                    "backtrack_tokens": ["<|BACKTRACK|>"] * len(error_tokens),
                    "correction": correct_continuation
                }
                training_examples.append(backtrack_example)
    
    return training_examples
```

### Iterative Self-Play Loop

```python
for iteration in range(num_iterations):
    # 1. Generate errors using current model
    error_data = generate_self_error_data(model, tokenizer, prompts)
    
    # 2. Create training dataset
    train_dataset = BacktrackDataset(error_data)
    
    # 3. Fine-tune model (with masked-error loss!)
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        data_collator=BacktrackDataCollator()  # With loss masking
    )
    trainer.train()
    
    # 4. Evaluate and repeat
    evaluate(model)
```

### Expected Outcome
- Training data matches model's actual error distribution
- Better generalization to inference-time errors
- **Estimated improvement**: 10-25% accuracy gain

---

## Technique 6: Token-Level DPO 🥉

### Concept
Use Direct Preference Optimization at the token level, not sequence level.

### Implementation

```python
from trl import DPOTrainer

def create_dpo_pairs(example):
    """
    Create preference pairs for DPO training.
    
    Chosen (winner): Error → Backtrack → Correction
    Rejected (loser): Error → Continuation of Error
    """
    prompt = example["prompt"]
    error_part = example["error_tokens"]
    
    # Chosen: Backtrack and correct
    chosen = error_part + example["backtrack_tokens"] + example["correction"]
    
    # Rejected: Continue with error (hallucinate more)
    rejected = error_part + generate_error_continuation(error_part)
    
    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected
    }

# Create DPO dataset
dpo_dataset = base_dataset.map(create_dpo_pairs)

# Train with DPO
dpo_trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,  # Frozen copy
    train_dataset=dpo_dataset,
    tokenizer=tokenizer,
    args=dpo_args
)
dpo_trainer.train()
```

### Token-Level Extension

For true token-level DPO, use recent research like **Token-Level Direct Preference Optimization** which provides per-token rewards:

```python
# Pseudo-code for token-level DPO
def token_level_dpo_loss(chosen_logprobs, rejected_logprobs, token_rewards):
    """
    Apply DPO loss at each token position.
    
    token_rewards: Per-token reward signal (e.g., 1 for backtrack, -1 for error)
    """
    losses = []
    for t in range(len(chosen_logprobs)):
        # Higher reward tokens get larger preference margins
        margin = token_rewards[t]
        loss_t = -log_sigmoid(margin * (chosen_logprobs[t] - rejected_logprobs[t]))
        losses.append(loss_t)
    
    return sum(losses)
```

### Expected Outcome
- Explicit preference for backtracking over error continuation
- Token-level credit assignment
- **Estimated improvement**: 15-25% accuracy gain

---

## Technique 7: Process Reward Model (PRM)

### Concept
Train a separate model to provide step-level rewards for each generation step.

### Architecture

```
Main Model (Generator)          PRM (Verifier)
        ↓                              ↓
   Generate step              Evaluate step correctness
        ↓                              ↓
   [token_1, ...]                  [score_1]
   [token_2, ...]                  [score_2]
        ...                           ...
```

### Implementation

```python
class ProcessRewardModel(nn.Module):
    def __init__(self, base_model_name, num_labels=1):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(base_model_name)
        self.reward_head = nn.Linear(self.backbone.config.hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask):
        # Get hidden states
        outputs = self.backbone(input_ids, attention_mask)
        hidden_states = outputs.last_hidden_state
        
        # Get reward for each position
        rewards = self.reward_head(hidden_states)
        return rewards  # Shape: (batch, seq_len, 1)

# Training PRM
def train_prm(prm, dataset):
    """
    Train PRM on labeled step-correctness data.
    
    Dataset format:
    - input_ids: Partial generation
    - step_labels: 1 if step is correct, 0 otherwise
    """
    for batch in dataset:
        rewards = prm(batch["input_ids"], batch["attention_mask"])
        loss = F.binary_cross_entropy_with_logits(rewards, batch["step_labels"])
        loss.backward()
        optimizer.step()
```

### Using PRM for Backtrack Training

```python
def generate_with_prm(model, prm, prompt, threshold=0.5):
    """Generate with PRM-guided backtracking."""
    generated = []
    
    while not done:
        # Generate next token
        next_token = model.generate_next_token(prompt + generated)
        generated.append(next_token)
        
        # Get PRM score
        score = prm(tokenize(prompt + generated))[-1]
        
        if score < threshold:
            # PRM flags this as likely wrong → backtrack
            generated.append("<|BACKTRACK|>")
            # Can also remove the low-score token from context
```

### Expected Outcome
- Dynamic, learned triggering of backtrack
- Model learns *when* to backtrack, not just *how*
- **Estimated improvement**: 20-35% accuracy gain

---

## Technique 8: MCTS-Based Data Generation

### Concept
Use Monte Carlo Tree Search to find optimal backtracking strategies, then distill into the model.

### Implementation

```python
class MCTSBacktrackNode:
    def __init__(self, tokens, parent=None):
        self.tokens = tokens
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0
    
    def expand(self, model, tokenizer):
        """Expand node with possible next actions."""
        # Action 1: Generate next token
        next_tokens = model.generate_candidates(self.tokens, k=5)
        for tok in next_tokens:
            child = MCTSBacktrackNode(self.tokens + [tok], parent=self)
            self.children.append(child)
        
        # Action 2: Backtrack (if tokens exist)
        if len(self.tokens) > 0:
            backtrack_child = MCTSBacktrackNode(
                self.tokens[:-1] + ["<|BACKTRACK|>"], 
                parent=self
            )
            self.children.append(backtrack_child)
    
    def evaluate(self, verifier):
        """Evaluate node value using verifier."""
        return verifier(self.tokens)


def mcts_generate(model, verifier, prompt, num_simulations=100):
    """Generate using MCTS with backtracking actions."""
    root = MCTSBacktrackNode(tokenize(prompt))
    
    for _ in range(num_simulations):
        node = select(root)       # UCB selection
        node = expand(node)        # Add children
        value = evaluate(node)     # Rollout + evaluate
        backpropagate(node, value) # Update ancestors
    
    # Extract best path
    best_path = get_best_path(root)
    return best_path


def distill_mcts_to_model(model, prompts, num_samples=10000):
    """Use MCTS to generate optimal traces, then train model."""
    traces = []
    for prompt in prompts:
        trace = mcts_generate(model, verifier, prompt)
        traces.append(trace)
    
    # Train model on MCTS-generated traces
    # Using MASKED loss (don't learn errors, only backtracks and corrections)
    trainer = SFTTrainer(
        model=model,
        train_dataset=Dataset.from_list(traces),
        data_collator=BacktrackDataCollator()  # With error masking
    )
    trainer.train()
```

### Expected Outcome
- Optimal backtracking strategies discovered via search
- Model learns from near-optimal demonstrations
- **Estimated improvement**: 25-40% accuracy gain

---

## Technique 9: KV-Cache Rewinding Inference

### Concept
At inference time, when backtrack token is generated, **physically remove** the previous token from the KV cache.

### Implementation

```python
def generate_with_backtrack(model, tokenizer, prompt, max_length=512):
    """
    Custom generation loop with physical backtracking.
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    past_key_values = None
    generated_tokens = []
    
    for _ in range(max_length):
        # Forward pass
        outputs = model(
            input_ids=input_ids[:, -1:] if past_key_values else input_ids,
            past_key_values=past_key_values,
            use_cache=True
        )
        
        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1)
        
        # Check if backtrack token
        if next_token.item() == tokenizer.convert_tokens_to_ids("<|BACKTRACK|>"):
            # PHYSICAL BACKTRACK
            if len(generated_tokens) > 0:
                # Remove last generated token
                generated_tokens.pop()
                
                # CRITICAL: Rewind KV cache
                past_key_values = rewind_kv_cache(past_key_values, num_tokens=2)
                # Remove both: the token being deleted AND the backtrack token
                
                continue  # Don't add backtrack to output
        
        generated_tokens.append(next_token.item())
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)
        
        if next_token.item() == tokenizer.eos_token_id:
            break
    
    return tokenizer.decode(generated_tokens)


def rewind_kv_cache(past_key_values, num_tokens=1):
    """
    Remove the last N tokens from the KV cache.
    
    past_key_values: Tuple of (key, value) for each layer
    """
    new_past = []
    for layer_past in past_key_values:
        # Each layer_past is (key, value), each of shape (batch, heads, seq, dim)
        key, value = layer_past
        # Remove last num_tokens positions
        new_key = key[:, :, :-num_tokens, :]
        new_value = value[:, :, :-num_tokens, :]
        new_past.append((new_key, new_value))
    
    return tuple(new_past)
```

### Expected Outcome
- Eliminates context pollution at inference
- Corrections generated from clean state
- **Estimated improvement**: 5-15% at inference (no training change)

---

## Technique 10: Verifier Head Architecture

### Concept
Add a separate "verifier head" to the model that predicts probability of needing backtrack.

### Architecture

```python
class LLMWithVerifier(nn.Module):
    def __init__(self, base_model_name):
        super().__init__()
        self.base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
        hidden_size = self.base_model.config.hidden_size
        
        # Verifier head: predicts P(should_backtrack | context)
        self.verifier_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, input_ids, attention_mask=None, labels=None, verifier_labels=None):
        # Get base model outputs
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True
        )
        
        # Get last hidden state for verifier
        hidden_states = outputs.hidden_states[-1]  # (batch, seq, hidden)
        
        # Verifier prediction
        verifier_logits = self.verifier_head(hidden_states)  # (batch, seq, 1)
        
        # Compute verifier loss if labels provided
        verifier_loss = None
        if verifier_labels is not None:
            verifier_loss = F.binary_cross_entropy(
                verifier_logits.squeeze(-1),
                verifier_labels.float()
            )
        
        return {
            "loss": outputs.loss + verifier_loss if verifier_loss else outputs.loss,
            "lm_loss": outputs.loss,
            "verifier_loss": verifier_loss,
            "verifier_logits": verifier_logits,
            "logits": outputs.logits
        }
```

### Training

```python
# Verifier labels: 1 if the next token should be BACKTRACK, 0 otherwise
# These can be derived from the training data:
# - Position right after error tokens: label = 1
# - All other positions: label = 0

def create_verifier_labels(input_ids, error_end_positions, backtrack_token_id):
    """Create labels for verifier head."""
    labels = torch.zeros_like(input_ids, dtype=torch.float)
    
    for error_end in error_end_positions:
        # The position after error should predict backtrack
        labels[:, error_end] = 1.0
    
    return labels
```

### Inference with Verifier

```python
def generate_with_verifier(model, tokenizer, prompt, threshold=0.5):
    """Use verifier head to decide when to backtrack."""
    # ... standard generation loop ...
    
    for step in range(max_length):
        outputs = model(input_ids)
        
        # Check verifier
        verifier_prob = outputs["verifier_logits"][:, -1, 0].item()
        
        if verifier_prob > threshold:
            # Verifier says: "You should backtrack!"
            # Force backtrack token
            next_token = backtrack_token_id
        else:
            # Normal generation
            next_token = sample(outputs["logits"][:, -1, :])
        
        # ... continue ...
```

### Expected Outcome
- Learned triggering of backtrack based on internal state
- Model "knows when it's wrong"
- **Estimated improvement**: 20-35% accuracy gain

---

## Recommended Implementation Order

### Phase 1: Quick Wins (1-2 days)

1. **Technique 2**: Semantic Token Initialization
   - Very easy, immediate stability improvement
   
2. **Technique 1**: Masked-Error SFT
   - Fixes the fundamental negative learning problem

### Phase 2: Core Improvements (1 week)

3. **Technique 3**: Curriculum Learning
   - Easy to implement, stabilizes training

4. **Technique 4**: Hard Backtrack Attention Mask
   - Requires more work but high impact

5. **Technique 9**: KV-Cache Rewinding Inference
   - Improves inference quality immediately

### Phase 3: Advanced Methods (2-4 weeks)

6. **Technique 5**: Self-Generated Error Data
   - Requires iterative training loop

7. **Technique 6**: Token-Level DPO
   - Builds on DPO library

### Phase 4: Research Directions (1-3 months)

8. **Technique 7**: Process Reward Model
9. **Technique 8**: MCTS-Based Generation
10. **Technique 10**: Verifier Head Architecture

---

## Summary

The key insight is that **standard SFT/RL fails because it teaches the model to generate errors**. The solutions fall into three categories:

1. **Don't learn errors**: Mask error tokens in loss computation
2. **Don't attend to errors**: Use custom attention masks
3. **Learn when to backtrack**: Use verifiers, PRMs, or RL

Implementing Techniques 1, 2, 4, and 9 should provide immediate significant improvements. For cutting-edge performance, Techniques 7-10 represent the research frontier.
