"""DPO training executor for TRL DPOTrainer."""

from llmhalluc.hparams.ft_args import DPOArguments

from .base import BaseExecutor


class DPOExecutor(BaseExecutor):
    """Executor for Direct Preference Optimization (DPO) training.

    DPOTrainer uses the model as its own implicit reference by default,
    so no separate reference model is needed.
    """

    def __init__(self, args: DPOArguments):
        super().__init__(args)

    def _get_trainer_class(self):
        """Return DPOTrainer class for preference optimization."""
        from trl import DPOTrainer

        return DPOTrainer

    def _get_dataset_converter(self):
        """Return DPODatasetConverter for preference optimization."""
        return "dpo"


def run_dpo(args: DPOArguments):
    """Run DPO training with the given arguments.

    Args:
        args: DPOArguments containing all training configuration.
    """
    executor = DPOExecutor(args=args)
    executor.on_train_start()
    executor.fit()
    executor.on_train_end()
