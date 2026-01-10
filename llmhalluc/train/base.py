import torch
import logging
from abc import ABC
from transformers import Trainer
from datasets import DatasetDict, Dataset

from transformers import PreTrainedModel, PreTrainedTokenizer

from llmhalluc.data import DatasetConverter, get_dataset
from llmhalluc.models import get_model, get_tokenizer
from llmhalluc.models.peft import get_quantization_config
from llmhalluc.hparams import SFTArguments

logger = logging.getLogger(__name__)


def run_train(args):
    if isinstance(args, SFTArguments):
        from .sft import run_sft

        run_sft(args)
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

        if not isinstance(model, PreTrainedModel):
            self.setup_model()
        else:
            self.model = model

        if not isinstance(tokenizer, PreTrainedTokenizer):
            self.setup_tokenizer()
        else:
            self.tokenizer = tokenizer

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

    def setup_model(self):
        model_kwargs = {
            "low_cpu_mem_usage": True,
            "trust_remote_code": getattr(self.args, "trust_remote_code", True),
        }
        if getattr(self.args, "bf16", False):
            model_kwargs["torch_dtype"] = torch.bfloat16
        elif getattr(self.args, "fp16", False):
            model_kwargs["torch_dtype"] = torch.float16

        # Add quantization config for QLoRA
        quant_config = get_quantization_config(self.args)
        if quant_config is not None:
            model_kwargs["quantization_config"] = quant_config
            model_kwargs["device_map"] = "auto"

        self.model = get_model(self.args.model_name_or_path, **model_kwargs)

    def setup_tokenizer(self):
        self.tokenizer = get_tokenizer(self.args.tokenizer_name_or_path)

    def setup_dataset(self):
        self.dataset = get_dataset(
            self.args.dataset_path, self.args.dataset_name, converter=self.converter
        )

    def setup_trainer(self):
        raise NotImplementedError("Subclass must implement this method")

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
