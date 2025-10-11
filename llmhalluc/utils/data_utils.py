"""Data processing utilities."""

from pathlib import Path
from typing import Any, Callable
from datasets import Dataset, DatasetDict


def process_dataset(
    dataset: Dataset | DatasetDict,
    processor: Callable[[dict[str, Any]], dict[str, Any]],
    split: str | list[str] | None = None,
    repeat: int = 1,
    num_proc: int = 12,
    hf_push_url: str | None = None,
    force_push: bool = False,
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
                **kwargs,
            )
    else:
        # Repeat dataset if specified (only for train split)
        actual_repeat = repeat if split == "train" else 1
        if actual_repeat > 1:
            dataset = dataset.repeat(actual_repeat)

        # Process dataset
        column_names = dataset.column_names
        dataset_to_return = dataset.map(
            processor,
            batched=False,
            num_proc=num_proc,
            remove_columns=column_names,
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

    # # Save processed dataset
    # save_path = Path(data_dir) / dataset_name / f"{split}.json"
    # if not save_path.parent.exists():
    #     save_path.parent.mkdir(parents=True, exist_ok=True)

    # processed_dataset.to_json(str(save_path), orient="records")
