"""DPO training executor for TRL DPOTrainer."""

from trl import DPOTrainer

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
        return DPOTrainer


def run_dpo(args: DPOArguments):
    """Run DPO training with the given arguments.

    Args:
        args: DPOArguments containing all training configuration.
    """
    executor = DPOExecutor(args=args)
    executor.on_train_start()
    executor.fit()
    executor.on_train_end()
