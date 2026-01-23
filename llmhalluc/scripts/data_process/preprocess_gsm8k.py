#!/usr/bin/env python3
"""Preprocess GSM8K dataset following ReVISE logic.

This script processes openai/gsm8k to mtybilly/GSM8K by:
1. Removing calculator annotations <<...>>
2. Replacing #### with "The answer is:"
3. Splitting train into train/eval (90/10)

Reference: repos/ReVISE/revise/preprocess.py
"""

import re
from datasets import load_dataset


def preprocess_gsm8k():
    """Preprocess GSM8K following ReVISE logic."""
    print("Loading openai/gsm8k...")
    dataset = load_dataset("openai/gsm8k", name="main")
    dataset.cleanup_cache_files()

    print("Original dataset:")
    print(f"  train: {len(dataset['train'])} examples")
    print(f"  test: {len(dataset['test'])} examples")

    # Step 1: Remove calculator annotations <<...>>
    print("\nStep 1: Removing calculator annotations <<...>>")
    dataset = dataset.map(
        lambda x: {
            "question": x["question"],
            "answer": re.sub(r"<<.*?>>", "", x["answer"]),
        }
    )

    # Step 2: Replace #### with "The answer is:"
    print("Step 2: Replacing '####' with 'The answer is:'")
    dataset = dataset.map(
        lambda x: {"answer": x["answer"].replace("####", "The answer is:")}
    )

    # Step 3: Split train into train/eval (90/10)
    print("Step 3: Splitting train into train/eval (90/10)")
    train_eval_split = dataset["train"].train_test_split(test_size=0.1, seed=42)
    dataset["train"] = train_eval_split["train"]
    dataset["eval"] = train_eval_split["test"]

    print("\nProcessed dataset:")
    print(f"  train: {len(dataset['train'])} examples")
    print(f"  eval: {len(dataset['eval'])} examples")
    print(f"  test: {len(dataset['test'])} examples")

    # Show sample
    print("\n=== Sample (train[0]) ===")
    sample = dataset["train"][0]
    print(f"question: {sample['question'][:150]}...")
    print(f"answer: {sample['answer']}")

    # Verify no <<...>> remaining
    print("\n=== Verification ===")
    for split_name in dataset.keys():
        has_calc = sum(1 for ex in dataset[split_name] if "<<" in ex["answer"])
        has_hash = sum(1 for ex in dataset[split_name] if "####" in ex["answer"])
        print(
            f"{split_name}: <<...>> remaining: {has_calc}, #### remaining: {has_hash}"
        )

    # Rename columns to match our convention
    print("\nStep 4: Renaming columns (question->query, answer->response)")
    dataset = dataset.rename_columns({"question": "query", "answer": "response"})

    # Push to hub
    print("\nStep 5: Pushing to mtybilly/GSM8K...")
    dataset.push_to_hub("mtybilly/GSM8K")
    print("Done!")

    return dataset


if __name__ == "__main__":
    preprocess_gsm8k()
