import logging
from abc import ABC, abstractmethod

import torch
from datasets import Dataset, DatasetDict, load_dataset
from peft import LoraConfig, TaskType
from transformers import (
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    Trainer,
)

from llmhalluc.data import DatasetConverter, get_dataset_converter, load_data_config
from llmhalluc.hparams import DPOArguments, GRPOArguments, SFTArguments
from llmhalluc.models import get_model, get_tokenizer
from llmhalluc.utils import print_dataset, process_dataset, wrap_converter_with_replace

from .callbacks import get_callbacks

logger = logging.getLogger(__name__)


def run_train(args):
    if isinstance(args, SFTArguments):
        from .sft import run_sft

        run_sft(args)
    elif isinstance(args, DPOArguments):
        from .dpo import run_dpo

        run_dpo(args)
    elif isinstance(args, GRPOArguments):
        from .grpo import run_grpo

        run_grpo(args)
    else:
        raise ValueError(f"Unknown support argument type: {type(args)}")
    logger.info("Training completed successfully")


class BaseExecutor(ABC):
    def __init__(
        self,
        args,
        model: PreTrainedModel = None,
        tokenizer: PreTrainedTokenizer | None = None,
        dataset: DatasetDict | Dataset | None = None,
        converter: DatasetConverter | None = None,
        trainer: Trainer | None = None,
        save_model: bool = True,
    ):
        self.args = args
        self.save_model = save_model
        self.converter = converter  # Store the converter

        # IMPORTANT: Setup tokenizer BEFORE model because model resize
        # depends on tokenizer length when adding special tokens
        if not isinstance(tokenizer, PreTrainedTokenizer):
            self.setup_tokenizer()
        else:
            self.tokenizer = tokenizer

        if not isinstance(model, PreTrainedModel):
            self.setup_model()
        else:
            self.model = model

        if not isinstance(dataset, DatasetDict | Dataset):
            self.setup_dataset()
        else:
            self.dataset = dataset

        if not isinstance(trainer, Trainer):
            self.setup_trainer()
        else:
            self.trainer = trainer

        self.__post_init__()

    def __post_init__(self):
        pass

    @abstractmethod
    def _get_trainer_class(self):
        """Return the trainer class to use (e.g., SFTTrainer, DPOTrainer).

        Subclasses must implement this method to specify which trainer to use.
        """
        pass

    @abstractmethod
    def _get_dataset_converter(self):
        """Return the dataset converter to use (e.g., SFTDatasetConverter, DPODatasetConverter).

        Subclasses must implement this method to specify which dataset converter to use.
        """
        pass

    def _get_peft_config(self):
        """Build LoraConfig if finetuning_type is 'lora' or 'qlora'.

        Returns:
            LoraConfig if using PEFT, None otherwise.
        """
        finetuning_type = getattr(self.args, "finetuning_type", "full")

        if finetuning_type not in ["lora", "qlora"]:
            return None

        # Parse lora_target into list of module names or 'all-linear'
        lora_target = getattr(self.args, "lora_target", "all")
        if lora_target == "all":
            target_modules = "all-linear"
        else:
            target_modules = [m.strip() for m in lora_target.split(",")]

        return LoraConfig(
            r=getattr(self.args, "lora_rank", 8),
            lora_alpha=getattr(self.args, "lora_alpha", 16),
            lora_dropout=getattr(self.args, "lora_dropout", 0.05),
            target_modules=target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

    def _get_quantization_config(self):
        """Build BitsAndBytesConfig for QLoRA quantization.

        Returns:
            BitsAndBytesConfig if using QLoRA/4-bit, None otherwise.
        """
        finetuning_type = getattr(self.args, "finetuning_type", "full")
        load_in_4bit = getattr(self.args, "load_in_4bit", False)

        if finetuning_type != "qlora" and not load_in_4bit:
            return None

        compute_dtype_str = getattr(self.args, "bnb_4bit_compute_dtype", "bfloat16")
        compute_dtype = getattr(torch, compute_dtype_str, torch.bfloat16)

        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=getattr(self.args, "bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=getattr(
                self.args, "bnb_4bit_use_double_quant", True
            ),
        )

    def setup_tokenizer(self):
        """Setup tokenizer with special tokens if configured.

        Must be called before setup_model() since model resize depends on tokenizer.
        """
        self.tokenizer = get_tokenizer(
            self.args.tokenizer_name_or_path,
            args=self.args,
        )

    def setup_model(self):
        """Setup model with optional embedding resize for special tokens.

        Requires tokenizer to be set up first for embedding resize.
        """
        model_kwargs = {
            "low_cpu_mem_usage": True,
            "trust_remote_code": getattr(self.args, "trust_remote_code", True),
        }
        if getattr(self.args, "bf16", False):
            model_kwargs["torch_dtype"] = torch.bfloat16
        elif getattr(self.args, "fp16", False):
            model_kwargs["torch_dtype"] = torch.float16

        # Add quantization config for QLoRA
        quant_config = self._get_quantization_config()
        if quant_config is not None:
            model_kwargs["quantization_config"] = quant_config
            model_kwargs["device_map"] = "auto"

        self.model = get_model(
            self.args.model_name_or_path,
            args=self.args,
            tokenizer=self.tokenizer,
            **model_kwargs,
        )

    def _get_dataset(self, dataset_key: str, split_type: str = "train") -> Dataset:
        """Load and process a single dataset.

        Args:
            dataset_key: Key to look up in dataset_info.json
            split_type: Either "train" or "eval" for default split selection

        Returns:
            Processed dataset ready for training/evaluation
        """
        data_config = load_data_config()

        dataset_info = data_config.get(dataset_key)
        if dataset_info is None:
            raise ValueError(f"Dataset '{dataset_key}' not found in dataset_info.json")

        hf_url = dataset_info.get("hf_hub_url")
        if not hf_url:
            raise ValueError(
                f"Dataset '{dataset_key}' does not have 'hf_hub_url' in dataset_info.json"
            )

        default_split = "train" if split_type == "train" else "test"
        split = dataset_info.get("split", default_split)

        dataset = load_dataset(
            hf_url,
            name=dataset_info.get("subset"),
            split=split,
        )

        preprocess_converter_name = dataset_info.get("converter", None)
        column_mapping = dataset_info.get("columns", {})

        if preprocess_converter_name:
            preprocess_converter, preprocess_converter_args = get_dataset_converter(
                preprocess_converter_name, **column_mapping
            )
            dataset = process_dataset(
                dataset=dataset,
                processor=preprocess_converter,
                load_from_cache_file=self.args.load_from_cache_file,
                **preprocess_converter_args,
            )

        converter, converter_args = get_dataset_converter(
            self._get_dataset_converter(),
            **column_mapping if not preprocess_converter_name else {},
        )

        if getattr(self.args, "replace_text", None):
            converter = wrap_converter_with_replace(
                converter,
                self.args.replace_text,
                converter_args.get("batched", False),
            )

        dataset = process_dataset(
            dataset=dataset,
            processor=converter,
            split=split,
            load_from_cache_file=self.args.load_from_cache_file,
            **converter_args,
        )

        return dataset

    def setup_dataset(self):
        """Load and prepare dataset from dataset_info.json.

        Loads train and eval datasets separately from their configured
        HuggingFace Hub URLs in dataset_info.json and applies the appropriate converter.
        """
        # Load train dataset
        train_dataset = self._get_dataset(self.args.dataset, split_type="train")
        self.dataset = DatasetDict({"train": train_dataset})

        # Load eval dataset if specified
        if self.args.eval_dataset:
            eval_dataset = self._get_dataset(self.args.eval_dataset, split_type="eval")
            self.dataset["eval"] = eval_dataset

        print_dataset(self.dataset)

    def setup_trainer(self):
        """Setup trainer with model, tokenizer, and dataset.

        Uses the trainer class returned by _get_trainer_class().
        """
        trainer_class = self._get_trainer_class()

        # Build LoRA config if using PEFT
        peft_config = self._get_peft_config()

        # Build callbacks
        callbacks = get_callbacks(self.args)

        self.trainer = trainer_class(
            model=self.model,
            args=self.args,
            processing_class=self.tokenizer,
            train_dataset=self.dataset.get("train"),
            eval_dataset=self.dataset.get("eval"),
            peft_config=peft_config,
            callbacks=callbacks if callbacks else None,
        )

    def fit(self):
        self.train_result = self.trainer.train()
        if self.save_model:
            self.trainer.save_model()
        return self.train_result

    def on_train_start(self):
        torch.cuda.empty_cache()

    def on_train_end(self):
        if hasattr(self.train_result, "metrics"):
            metrics = self.train_result.metrics
            metrics["train_samples"] = len(self.dataset["train"])
            self.trainer.log_metrics("train", metrics)
            self.trainer.save_metrics("train", metrics)

        self.trainer.save_state()
