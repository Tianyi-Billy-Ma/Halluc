"""Data processing utilities."""

import json
from collections.abc import Callable
from typing import Any

from datasets import Dataset, DatasetDict


def print_dataset(dataset: Dataset | DatasetDict, n: int = 1) -> None:
    """Print a sample from a dataset (only on rank 0).

    Args:
        dataset: Dataset or DatasetDict to sample from.
        n: Number of samples to print.
    """
    import torch.distributed as dist

    if dist.is_initialized() and dist.get_rank() != 0:
        return

    if isinstance(dataset, DatasetDict):
        for split, ds in dataset.items():
            print(f"=== Split: {split} ===")
            print_dataset(ds, n)
    else:
        print(f"  Num samples: {len(dataset)}")
        print(f"  Columns: {dataset.column_names}")

        if "input_ids" in dataset.column_names:
            try:
                lengths = [len(x) for x in dataset["input_ids"]]
                total_tokens = sum(lengths)
                avg_tokens = total_tokens / len(lengths) if lengths else 0
                max_tokens = max(lengths) if lengths else 0
                print(f"  Total tokens: {total_tokens}")
                print(f"  Avg tokens: {avg_tokens:.2f}")
                print(f"  Max tokens: {max_tokens}")
            except Exception as e:
                print(f"  Could not calculate token stats: {e}")

        print(f"--- Sample 0 ---")
        for i in range(min(n, len(dataset))):
            print(json.dumps(dataset[i], indent=2, default=str))


def process_dataset(
    dataset: Dataset | DatasetDict,
    processor: Callable[[dict[str, Any]], dict[str, Any]],
    split: str | list[str] | None = None,
    repeat: int = 1,
    num_proc: int = 12,
    hf_push_url: str | None = None,
    force_push: bool = False,
    batched: bool = False,
    batch_size: int = 1,
    load_from_cache_file: bool = True,
    **kwargs,
) -> Dataset | DatasetDict:
    """Process dataset using a converter function.

    This function applies a processor function to a dataset and saves the result.
    It handles both single splits and multiple splits recursively.

    Args:
        dataset: Input dataset to process.
        processor: Function to apply to each example.
        dataset_name: Name for the processed dataset.
        split: Dataset split(s) to process.
        repeat: Number of times to repeat the dataset (only for train split).
        num_proc: Number of processes for parallel processing.
        hf_push_url: HuggingFace Hub URL to push the dataset to.
        force_push: If True, force push even if dataset hasn't changed.
        **kwargs: Additional arguments for dataset.map().
    """
    if isinstance(dataset, DatasetDict):
        if split is None:
            split = list(dataset.keys())
        else:
            split = split if isinstance(split, list) else [split]
        dataset_to_return = DatasetDict()
        for s in split:
            dataset_to_return[s] = process_dataset(
                dataset=dataset[s],
                processor=processor,
                split=s,
                repeat=repeat,
                num_proc=num_proc,
                force_push=force_push,
                batched=batched,
                batch_size=batch_size,
                **kwargs,
            )
    else:
        # Repeat dataset if specified (only for train split)
        actual_repeat = repeat if split == "train" else 1
        dataset = dataset.repeat(actual_repeat) if actual_repeat > 1 else dataset

        # Process dataset
        column_names = dataset.column_names

        dataset_to_return = dataset.map(
            processor,
            batched=batched,
            batch_size=batch_size,
            num_proc=num_proc,
            remove_columns=column_names,
            load_from_cache_file=load_from_cache_file,
            **kwargs,
        )
    if hf_push_url is not None:
        from datetime import datetime

        commit_msg = f"Update dataset - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        dataset_to_return.push_to_hub(
            hf_push_url,
            private=False,
            commit_message=commit_msg,
            revision="main",  # Force push to main branch
        )
    return dataset_to_return


def wrap_converter_with_replace(converter, replace_text, batched):
    """Wrap converter to apply text replacements before conversion.

    Args:
        converter: The original converter callable.
        replace_text: Dictionary of text replacements {target: replacement}.
        batched: Whether the converter operates in batched mode.

    Returns:
        Wrapped converter that applies replacements first.
    """
    if not replace_text:
        return converter

    def replace_in_text(text):
        if not isinstance(text, str):
            return text
        for target, replacement in replace_text.items():
            text = text.replace(target, replacement)
        return text

    def text_replacer(example):
        new_example = {}
        if batched:
            for key, values in example.items():
                if isinstance(values, list) and values and isinstance(values[0], str):
                    new_example[key] = [replace_in_text(v) for v in values]
                else:
                    new_example[key] = values
        else:
            for key, value in example.items():
                new_example[key] = replace_in_text(value)
        return new_example

    def wrapped_converter(example, **kwargs):
        example = text_replacer(example)
        return converter(example, **kwargs)

    return wrapped_converter

    # # Save processed dataset
    # save_path = Path(data_dir) / dataset_name / f"{split}.json"
    # if not save_path.parent.exists():
    #     save_path.parent.mkdir(parents=True, exist_ok=True)

    # processed_dataset.to_json(str(save_path), orient="records")
