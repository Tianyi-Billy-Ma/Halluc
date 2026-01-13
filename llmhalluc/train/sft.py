"""SFT training executor for TRL SFTTrainer."""

from trl import SFTTrainer

from llmhalluc.hparams.ft_args import SFTArguments

from .base import BaseExecutor


class SFTExecutor(BaseExecutor):
    """Executor for Supervised Fine-Tuning (SFT)."""

    def __init__(self, args: SFTArguments):
        super().__init__(args)

    def _get_trainer_class(self):
        """Return SFTTrainer class for supervised fine-tuning."""
        return SFTTrainer

    def _get_dataset_converter(self) -> str:
        """Return SFTDatasetConverter for supervised fine-tuning."""
        return "sft"


def run_sft(args: SFTArguments):
    """Run SFT training with the given arguments.

    Args:
        args: SFTArguments containing all training configuration.
    """
    executor = SFTExecutor(args=args)
    executor.on_train_start()
    executor.fit()
    executor.on_train_end()
