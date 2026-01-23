#!/usr/bin/env python3
"""Generate GSM8K-Random-All dataset with random backtrack injections.

This script creates the mtybilly/GSM8K-Random-All dataset by:
1. Loading the original GSM8K dataset
2. Applying RandomBacktrackConverter with various configurations
3. Uploading specified subsets to HuggingFace Hub

Subsets to upload:
- p1_n1: 1 position, 1 error token per position
- p1_n3: 1 position, 3 error tokens per position
- p0.1_n10: 10% of positions, 10 error tokens per position
"""

import logging
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import our converter
import sys

sys.path.insert(0, "/Users/tianyima/Documents/GitHub/Tianyi-Billy-Ma/Halluc")
from llmhalluc.data.backtrack import RandomBacktrackConverter, BACKTRACK_TOKEN


def create_subset(
    dataset,
    tokenizer,
    backtrack_ratio: int | float,
    backtrack_num_errors: int,
    seed: int = 42,
) -> DatasetDict:
    """Create a dataset subset with specific backtrack configuration.

    Args:
        dataset: Original GSM8K dataset.
        tokenizer: Tokenizer for encoding/decoding.
        backtrack_ratio: Number or ratio of positions to sample.
        backtrack_num_errors: Number of error tokens per position.
        seed: Random seed for reproducibility.

    Returns:
        Processed DatasetDict with backtrack columns.
    """
    converter = RandomBacktrackConverter(
        tokenizer=tokenizer,
        backtrack_ratio=backtrack_ratio,
        backtrack_num_errors=backtrack_num_errors,
        query_key="query",
        response_key="response",
        seed=seed,
    )

    processed = {}
    for split_name, split_data in dataset.items():
        logger.info(
            f"Processing {split_name} split with ratio={backtrack_ratio}, num_errors={backtrack_num_errors}"
        )
        processed[split_name] = split_data.map(
            converter,
            remove_columns=split_data.column_names,
            desc=f"Converting {split_name}",
        )

    return DatasetDict(processed)


def format_ratio_name(ratio: int | float) -> str:
    """Format ratio for subset naming (e.g., 0.1 -> '0.1', 1 -> '1')."""
    if isinstance(ratio, float):
        return str(ratio)
    return str(int(ratio))


def main():
    # Configuration
    HF_REPO = "mtybilly/GSM8K-Random-All"
    MODEL_ID = "meta-llama/Llama-3.2-1B"  # Default Llama 3 tokenizer

    # Subsets to upload: (p_ratio, num_errors)
    SUBSETS_TO_UPLOAD = [
        (1, 1),  # p1_n1
        (1, 3),  # p1_n3
        (0.1, 10),  # p0.1_n10
    ]

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # Add backtrack token if not present
    if BACKTRACK_TOKEN not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [BACKTRACK_TOKEN]})
        logger.info(f"Added {BACKTRACK_TOKEN} to tokenizer vocabulary")

    logger.info("Loading preprocessed GSM8K dataset (mtybilly/GSM8K)...")
    original_dataset = load_dataset("mtybilly/GSM8K")
    logger.info(f"Loaded splits: {list(original_dataset.keys())}")

    # Process and upload each subset
    for p_ratio, num_errors in SUBSETS_TO_UPLOAD:
        subset_name = f"p{format_ratio_name(p_ratio)}_n{num_errors}"
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Creating subset: {subset_name}")
        logger.info(f"  backtrack_ratio={p_ratio}, backtrack_num_errors={num_errors}")
        logger.info(f"{'=' * 60}")

        # Create subset
        subset_dataset = create_subset(
            original_dataset,
            tokenizer,
            backtrack_ratio=p_ratio,
            backtrack_num_errors=num_errors,
            seed=42,  # Fixed seed for reproducibility
        )

        # Show sample
        sample = subset_dataset["train"][0]
        logger.info(f"\nSample from {subset_name}:")
        logger.info(f"  query: {sample['query'][:100]}...")
        logger.info(f"  response: {sample['response'][:100]}...")
        logger.info(f"  backtrack_response: {sample['backtrack_response'][:100]}...")
        logger.info(f"  backtrack_prefix: {sample['backtrack_prefix'][:80]}...")
        logger.info(f"  backtrack_suffix: {sample['backtrack_suffix'][:80]}...")

        # Verify backtrack correctness
        bt_token_count = sample["backtrack_response"].count(BACKTRACK_TOKEN)
        logger.info(f"  backtrack_token_count: {bt_token_count}")

        # Upload to HuggingFace
        logger.info(f"\nUploading {subset_name} to {HF_REPO}...")
        subset_dataset.push_to_hub(
            HF_REPO,
            config_name=subset_name,
            commit_message=f"Add {subset_name} subset (ratio={p_ratio}, num_errors={num_errors})",
        )
        logger.info(f"✓ Uploaded {subset_name}")

    logger.info("\n" + "=" * 60)
    logger.info("All subsets uploaded successfully!")
    logger.info(f"Dataset available at: https://huggingface.co/datasets/{HF_REPO}")


if __name__ == "__main__":
    main()
