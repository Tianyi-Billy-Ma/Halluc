"""SFT training executor for TRL SFTTrainer."""

import logging

from trl import SFTTrainer

from llmhalluc.data.collator import BacktrackMaskingCollator
from llmhalluc.hparams.ft_args import SFTArguments

from .base import BaseExecutor
from .callbacks import get_callbacks

logger = logging.getLogger(__name__)


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


class BacktrackSFTExecutor(SFTExecutor):
    """SFT executor with error token masking for backtrack training.

    This executor extends SFTExecutor to inject a custom data collator
    that masks error tokens (tokens preceding backtrack tokens) from
    the loss computation. This trains the model to:
    - Learn WHEN to emit backtrack tokens
    - Learn WHAT the correct tokens are
    - NOT learn to generate error tokens
    """

    def setup_trainer(self):
        """Setup trainer with custom BacktrackMaskingCollator."""
        trainer_class = self._get_trainer_class()

        # Build LoRA config if using PEFT
        peft_config = self._get_peft_config()

        # Build callbacks
        callbacks = get_callbacks(self.args)

        # Create custom collator for backtrack masking
        collator = BacktrackMaskingCollator(
            tokenizer=self.tokenizer,
            backtrack_token=self.args.backtrack_token,
            reset_position_ids=self.args.reset_position_ids,
        )

        self.trainer = trainer_class(
            model=self.model,
            args=self.args,
            processing_class=self.tokenizer,
            train_dataset=self.dataset.get("train"),
            eval_dataset=self.dataset.get("eval"),
            peft_config=peft_config,
            callbacks=callbacks if callbacks else None,
            data_collator=collator,
        )


def run_sft(args: SFTArguments):
    """Run SFT training with the given arguments.

    Args:
        args: SFTArguments containing all training configuration.

    If args.train_backtrack is True, uses BacktrackSFTExecutor which
    masks error tokens from the loss computation.
    """
    if getattr(args, "train_backtrack", False):
        executor = BacktrackSFTExecutor(args=args)
        logger.info("Using BacktrackSFTExecutor for backtrack training.")
    else:
        executor = SFTExecutor(args=args)
        logger.info("Using SFTExecutor for standard supervised fine-tuning.")

    executor.on_train_start()
    executor.fit()
    executor.on_train_end()
