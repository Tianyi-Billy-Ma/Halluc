from trl import SFTTrainer
from datasets import DatasetDict, load_dataset

from .base import BaseExecutor
from llmhalluc.hparams import SFTArguments
from llmhalluc.data import load_data_config, get_dataset_converter
from llmhalluc.utils import process_dataset, print_dataset
from llmhalluc.models.peft import get_peft_config


class SFTExecutor(BaseExecutor):
    def __init__(self, args: SFTArguments):
        super().__init__(args)

    def setup_dataset(self):
        """Load and prepare dataset for SFT training.

        Loads train and eval datasets separately from their configured
        HuggingFace Hub URLs in dataset_info.json and applies SFTDatasetConverter.
        """
        data_config = load_data_config()

        # Load and process train dataset
        train_info = data_config.get(self.args.dataset)
        if train_info is None:
            raise ValueError(
                f"Dataset '{self.args.dataset}' not found in dataset_info.json"
            )

        train_hf_url = train_info.get("hf_hub_url")
        if not train_hf_url:
            raise ValueError(
                f"Dataset '{self.args.dataset}' does not have 'hf_hub_url' in dataset_info.json"
            )

        train_split = train_info.get("split", "train")
        train_dataset = load_dataset(
            train_hf_url,
            name=train_info.get("subset"),
            split=train_info.get("split", "train"),
        )

        converter, converter_args = get_dataset_converter(
            self.args.converter or "sft", **train_info.get("columns", {})
        )

        # Apply converter using existing utility
        train_dataset = process_dataset(
            dataset=train_dataset,
            processor=converter,
            split=train_split,
            load_from_cache_file=self.args.load_from_cache_file,
            **converter_args,
        )

        # Build DatasetDict
        self.dataset = DatasetDict({"train": train_dataset})

        # Load and process eval dataset if specified
        if self.args.eval_dataset:
            eval_info = data_config.get(self.args.eval_dataset)
            if eval_info is None:
                raise ValueError(
                    f"Eval dataset '{self.args.eval_dataset}' not found in dataset_info.json"
                )

            eval_hf_url = eval_info.get("hf_hub_url")
            if not eval_hf_url:
                raise ValueError(
                    f"Eval dataset '{self.args.eval_dataset}' does not have 'hf_hub_url' in dataset_info.json"
                )

            eval_split = eval_info.get("split", "test")
            eval_dataset = load_dataset(
                eval_hf_url,
                name=eval_info.get("subset"),
                split=eval_info.get("split", "test"),
            )

            eval_converter, eval_converter_args = get_dataset_converter(
                self.args.converter or "sft", **eval_info.get("columns", {})
            )

            eval_dataset = process_dataset(
                dataset=eval_dataset,
                processor=eval_converter,
                split=eval_split,
                load_from_cache_file=self.args.load_from_cache_file,
                **eval_converter_args,
            )
            self.dataset["eval"] = eval_dataset

        print_dataset(self.dataset)

    def setup_trainer(self):
        """Setup SFTTrainer with the loaded model, tokenizer, and dataset."""
        train_dataset = self.dataset.get("train")
        eval_dataset = self.dataset.get("eval")

        # Build LoRA config if using PEFT
        peft_config = get_peft_config(self.args)

        self.trainer = SFTTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=self.args,
            peft_config=peft_config,
        )


def run_sft(args: SFTArguments):
    executor = SFTExecutor(args=args)
    executor.on_train_start()
    executor.fit()
    executor.on_train_end()
