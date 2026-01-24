# Special Token Embedding Initialization Strategies

This document summarizes the embedding initialization strategies implemented for special tokens (e.g., `<UNDO>`, `<|BACKTRACK|>`) in the N-MAR framework.

## Overview

When introducing new special tokens to a pretrained language model, the embedding vectors must be initialized thoughtfully. Random initialization places the token in an arbitrary region of the embedding space, often requiring significant training to move it toward semantically meaningful positions.

Our implementation (`llmhalluc/models/embedding.py`) provides four initialization strategies that leverage existing vocabulary embeddings to provide better starting points.

---

## Initialization Strategies

### 1. Mean-Pool Initialization (Baseline)

**Current Name:** `mean`  
**Proposed Names:** `centroid`, `vocabulary-mean`, `global-average`

**Method:**
Computes the centroid (mean) of all existing token embeddings in the vocabulary.

```python
base_embedding = model.get_input_embeddings().weight.mean(dim=0)
```

**Properties:**
- Places the new token at the "center" of the embedding space
- Model-agnostic — requires no semantic information
- Serves as fallback when other strategies fail

**When to Use:**
- As a baseline for comparison
- When no semantic information is available
- As fallback when description/semantic tokens are invalid

---

### 2. Description-Anchored Initialization

**Current Name:** `description`  
**Proposed Names:** `contextual-anchor`, `sentence-prototype`, `descriptive-prior`

**Method:**
Tokenizes a natural language description of the token's intended function, then averages the embeddings of all resulting tokens.

```python
# Example description for <UNDO> token:
description = "This token is used to delete the previous token in the response."

# Process:
tokens = tokenizer(description, add_special_tokens=False)
token_embeds = model.get_input_embeddings()(token_ids)
base_embedding = token_embeds.mean(dim=0)
```

**Properties:**
- Captures functional/behavioral semantics through natural language
- Leverages the model's existing understanding of the description words
- More expressive — can encode complex behaviors in a sentence

**Configuration:**
```python
{
    "<|BACKTRACK|>": {
        "description": "This token is used to delete the previous token in the response.",
        "strategy": "description"
    }
}
```

**When to Use:**
- When the token's behavior is best described in natural language
- When functional semantics are more important than lexical similarity

---

### 3. Lexical-Prototype Initialization

**Current Name:** `semantic`  
**Proposed Names:** `lexical-prototype`, `synonym-anchor`, `word-cluster`

**Method:**
Averages embeddings of semantically related words that represent the token's core concept.

```python
# Example semantic words for <UNDO> token:
semantic_words = ["delete", "remove", "undo", "erase", "back", 
                  "cancel", "retry", "revert", "reset", "clear", "backspace"]

# Process:
for word in semantic_words:
    token_id = tokenizer.convert_tokens_to_ids(word)
    if token_id != unk_token_id:
        valid_ids.append(token_id)
    else:
        # Handle subword tokenization
        subword_ids = tokenizer.encode(word, add_special_tokens=False)
        valid_ids.extend(subword_ids)

token_embeds = model.get_input_embeddings()(valid_ids)
base_embedding = token_embeds.mean(dim=0)
```

**Properties:**
- Places token in a region surrounded by related concepts
- Robust to tokenization differences across models
- Intuitive — directly specifies "this token means something like these words"

**Default Semantic Words (for backtrack-like tokens):**
```python
BACKTRACK_SEMANTIC_WORDS = [
    "delete", "remove", "undo", "erase", "back",
    "cancel", "retry", "revert", "reset", "clear", "backspace"
]
```

**When to Use:**
- When the token's meaning can be approximated by existing vocabulary
- When you want the token to inherit properties of similar words

---

### 4. Hybrid-Anchor Initialization

**Current Name:** `combined`  
**Proposed Names:** `hybrid-anchor`, `dual-prototype`, `blended-prior`

**Method:**
Computes a weighted average of the description-based and lexical-prototype embeddings.

```python
desc_embedding = _get_description_embedding(description, ...)
semantic_embedding = _get_semantic_embedding(semantic_words, ...)

# Weighted combination (default: 50/50)
base_embedding = description_weight * desc_embedding + (1 - description_weight) * semantic_embedding
```

**Properties:**
- Balances functional description with lexical similarity
- Configurable weight allows tuning based on task
- Most robust — benefits from both approaches

**Configuration:**
```python
{
    "<|BACKTRACK|>": {
        "description": "This token is used to delete the previous token in the response.",
        "semantic_words": ["delete", "remove", "undo", "erase", ...],
        "strategy": "combined",
        "description_weight": 0.5  # 50% description, 50% semantic
    }
}
```

**When to Use:**
- Default choice for most special tokens
- When both functional behavior and lexical semantics matter

---

## Noise Injection

All strategies support optional Gaussian noise injection to prevent multiple special tokens from having identical embeddings and to aid optimization:

```python
if add_noise:
    noise = torch.randn_like(base_embedding) * (1.0 / math.sqrt(embedding_dim))
    embed_weight[token_id] = base_embedding + noise
```

The noise scale `1/sqrt(d)` is chosen to be small relative to typical embedding norms while providing sufficient perturbation.

---

## Summary Table

| Strategy | Current Name | Proposed Name | Input | Best For |
|----------|--------------|---------------|-------|----------|
| Vocabulary centroid | `mean` | `centroid` | None | Fallback, baseline |
| Description averaging | `description` | `contextual-anchor` | Sentence | Behavioral semantics |
| Word prototype | `semantic` | `lexical-prototype` | Word list | Lexical similarity |
| Weighted combination | `combined` | `hybrid-anchor` | Both | Default choice |

---

## Implementation Details

### Key Files
- `llmhalluc/models/embedding.py` — Core initialization logic
- `llmhalluc/extras/constant.py` — Token mappings and default semantic words
- `llmhalluc/models/patcher.py` — Integration with model loading

### Configuration Example (from `constant.py`)
```python
SPECIAL_TOKEN_MAPPING = {
    "llama3": {
        "<|reserved_special_token_0|>": {
            "description": "This token is used to delete the previous token in the response.",
            "semantic_words": BACKTRACK_SEMANTIC_WORDS,
            "strategy": "combined",
            "description_weight": 0.5,
        }
    },
    "qwen3": {
        "<|BACKTRACK|>": {
            "description": "This token is used to delete the previous token in the response.",
            "semantic_words": BACKTRACK_SEMANTIC_WORDS,
            "strategy": "combined",
            "description_weight": 0.5,
        }
    },
}
```

### Enum Definition
```python
class InitStrategy(str, Enum):
    DESCRIPTION = "description"
    SEMANTIC = "semantic"
    COMBINED = "combined"
    MEAN = "mean"
```

---

## Recommendations for Paper

**Naming Suggestions for Publication:**

| Implementation | Paper-Ready Name | Rationale |
|----------------|------------------|-----------|
| `mean` | **Centroid Initialization** | Standard ML terminology |
| `description` | **Contextual Anchor** | Emphasizes use of natural language context |
| `semantic` | **Lexical Prototype** | Aligns with prototype theory in semantics |
| `combined` | **Hybrid Anchor** | Clear indication of dual approach |

These names are more descriptive for an academic audience and avoid potential confusion with other uses of "semantic" in NLP literature.
