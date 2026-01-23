# SFT Training Discrepancy Analysis: llmhalluc vs ReVISE

**Date**: 2026-01-22  
**Model**: Llama 3.2 1B  
**Dataset**: GSM8K  
**Training Method**: SFT with LoRA

## Executive Summary

This document analyzes the performance discrepancy between our `llmhalluc` SFT implementation and the reference `ReVISE` codebase when training Llama 3.2 1B on GSM8K. Several key differences were identified across hyperparameters, dataset processing, prompt formatting, and loss computation.

---

## 1. Hyperparameter Comparison

| Parameter | llmhalluc | ReVISE | Impact |
|-----------|-----------|--------|--------|
| **Batch Size** | 16 (per_device) × 8 (grad_accum) = 128 | 8 (per_device) × dynamic = 32 | **HIGH** - Different effective batch sizes affect learning dynamics |
| **Learning Rate** | 1e-4 | 1e-4 | Same |
| **Warmup Ratio** | 0.1 | 0.1 | Same |
| **Epochs** | 3 | 3 | Same |
| **Eval Strategy** | steps (500) | steps (0.1 = 10% of epoch) | Different eval frequency |
| **Save Strategy** | steps (500) | steps (0.1 = 10% of epoch) | Different checkpoint frequency |
| **Precision** | bf16 | bf16 | Same |
| **Flash Attention** | fa2 | flash_attention_2 | Same |
| **DeepSpeed** | ZeRO Stage 0 | Not used (accelerate only) | Potential overhead difference |

### Key Findings:
1. **Effective Batch Size Mismatch**: Our config uses `per_device_train_batch_size=16` with `gradient_accumulation_steps=8`, yielding an effective batch of 128 (single GPU). ReVISE uses 8 per device with dynamic gradient accumulation targeting batch 32.

---

## 2. Dataset Processing Pipeline

### 2.1 Data Source

| Aspect | llmhalluc | ReVISE |
|--------|-----------|--------|
| **Source** | `mtybilly/GSM8K` (HuggingFace) | `{HUB_USER_ID}/gsm8k` (preprocessed) |
| **Preprocessing** | None (pre-cleaned dataset) | Removes CoT tags `<<...>>`, replaces `####` with `The answer is:` |
| **Train/Eval Split** | Separate datasets (`gsm8k_train`, `gsm8k_eval`) | 90/10 split from train |

### 2.2 Prompt Formatting

**llmhalluc** (`llmhalluc/prompts/MathPrompt.py`):
```
{question}
Let's think step by step. Put your final answer at the end, starts with ####.
```

**ReVISE** (`revise/prompts.py`):
```
{question}
Let's think step by step. Put your final answer at the end with 'The answer is: .'
```

### **CRITICAL DIFFERENCE**: Answer Format Mismatch
- **llmhalluc**: Expects `####` as the answer marker
- **ReVISE**: Uses `The answer is:` as the answer marker (after preprocessing)

This mismatch means:
1. The model learns different answer formats
2. Evaluation metrics may not align if using different parsing logic

---

## 3. Dataset Converter Comparison

### llmhalluc (`llmhalluc/data/sft.py` → `SFTDatasetConverter`)
```python
def __call__(self, example):
    prompt_content = example.get("prompt", "")
    query_content = example.get("query", "")
    response_content = example.get("response", "")
    return {
        "prompt": [{"role": "user", "content": prompt_content + "\n" + query_content}],
        "completion": [{"role": "assistant", "content": response_content}],
    }
```

### ReVISE (`revise/sft.py` → `load_dataset`)
```python
dataset = dataset.map(lambda x: {
    "prompt": prepare_chat_messages_fn(x["question"]),  # Returns list of dicts
    "completion": [{"role": "assistant", "content": x["answer"]}],
})
```

### Key Differences:
1. **Message Structure**: Both use `prompt`/`completion` format compatible with TRL's SFTTrainer
2. **System Prompt Handling**: llmhalluc concatenates prompt+query into user message; ReVISE uses dedicated function
3. **Field Names**: llmhalluc uses `prompt`, `query`, `response` keys; ReVISE uses `question`, `answer`

---

## 4. Loss Computation

### llmhalluc
- Uses default TRL `SFTTrainer` behavior
- `completion_only_loss` is NOT explicitly configured in our YAML
- TRL's default: trains on full sequence (both prompt and completion)

### ReVISE
- Script passes `--completion_only_loss false`
- **Inconsistency**: The flag is passed but `revise/args/sft.py` doesn't define it
- Effectively uses default behavior (train on full sequence)

### **FINDING**: Both codebases train on full sequence, but this should be verified.

---

## 5. Chat Template

### llmhalluc
- Uses model's default chat template (Llama 3 instruct format)
- Template applied via `tokenizer.apply_chat_template()`

### ReVISE
- Custom Jinja2 template if tokenizer lacks one:
```jinja
{% for message in messages %}
{% if message['role'] == 'user' %}{{ '<|user|>\n' + message['content'] + '\n' }}
{% elif message['role'] == 'assistant' %}{{ '<|assistant|>\n' + message['content'] + eos_token }}
{% endif %}{% endfor %}
```

### **POTENTIAL ISSUE**: Different chat templates produce different tokenized sequences, affecting:
- Position of special tokens
- Where the model learns to generate vs attend
- Token counts and padding behavior

---

## 6. LoRA Configuration

| Parameter | llmhalluc | ReVISE |
|-----------|-----------|--------|
| **Rank** | 8 | Not specified (full fine-tuning) |
| **Alpha** | 16 | N/A |
| **Dropout** | 0.05 | N/A |
| **Target Modules** | all-linear | N/A |

### **MAJOR DIFFERENCE**: llmhalluc uses LoRA; ReVISE appears to do full fine-tuning
- ReVISE's `SFTConfig` extends `trl.SFTConfig` without PEFT configuration
- The script doesn't pass LoRA-related arguments

---

## 7. Root Causes of Discrepancy (Ranked by Likelihood)

### HIGH IMPACT

1. **LoRA vs Full Fine-tuning**
   - llmhalluc: LoRA (rank=8, alpha=16)
   - ReVISE: Full fine-tuning
   - **Impact**: Full fine-tuning typically achieves better performance but requires more compute/memory

2. **Answer Format Mismatch**
   - llmhalluc: `####` marker
   - ReVISE: `The answer is:` marker
   - **Impact**: Model learns different output patterns; evaluation may parse answers differently

3. **Effective Batch Size**
   - llmhalluc: 128
   - ReVISE: 32
   - **Impact**: Larger batches can lead to different convergence behavior

### MEDIUM IMPACT

4. **Chat Template Differences**
   - Different special token placement affects learning signal distribution

5. **Dataset Source**
   - Using different preprocessed versions of GSM8K

### LOW IMPACT

6. **DeepSpeed Overhead**
   - ZeRO-0 has minimal overhead but different memory management

---

## 8. Recommendations

### Immediate Actions

1. **Verify LoRA vs Full Fine-tuning**
   ```bash
   # Check ReVISE script for PEFT config
   grep -r "lora" repos/ReVISE/
   ```
   If ReVISE uses full fine-tuning, consider:
   - Running llmhalluc with `finetuning_type: full`
   - Or accepting LoRA's trade-off (efficiency vs performance)

2. **Align Answer Format**
   - Update `llmhalluc/prompts/MathPrompt.py`:
   ```python
   MATH_INSTRUCTION = """
   Let's think step by step. Put your final answer at the end with 'The answer is: .'
   """
   ```
   - Ensure dataset preprocessing matches

3. **Match Effective Batch Size**
   - Update `configs/llmhalluc/gsm8k/sft.yaml`:
   ```yaml
   per_device_train_batch_size: 8
   gradient_accumulation_steps: 4  # For 32 effective batch
   ```

4. **Use Same Dataset**
   - Either use ReVISE's preprocessed dataset
   - Or ensure our `mtybilly/GSM8K` has identical preprocessing

### Verification Steps

1. **Compare tokenized outputs**:
   ```python
   # Tokenize same example with both pipelines, compare token IDs
   ```

2. **Log training loss curves**:
   - Compare loss progression at same step counts

3. **Eval with identical metrics**:
   - Use same answer extraction regex
   - Use same evaluation harness version

---

## 9. Files Reference

### llmhalluc
- Config: `configs/llmhalluc/gsm8k/sft.yaml`
- SFT Executor: `llmhalluc/train/sft.py`
- Arguments: `llmhalluc/hparams/ft_args.py`
- Data Converter: `llmhalluc/data/sft.py`
- Prompt: `llmhalluc/prompts/MathPrompt.py`

### ReVISE
- Script: `repos/ReVISE/scripts/step-0-sft.sh`
- SFT Training: `repos/ReVISE/revise/sft.py`
- Arguments: `repos/ReVISE/revise/args/sft.py`
- Prompts: `repos/ReVISE/revise/prompts.py`
- Preprocessing: `repos/ReVISE/revise/preprocess.py`
