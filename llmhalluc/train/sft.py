from trl import SFTTrainer

from .base import BaseExecutor
from llmhalluc.hparams import SFTArguments


class SFTExecutor(BaseExecutor):
    def __init__(self, args):
        super().__init__(args)

    def setup_dataset(self):
        pass

    def setup_trainer(self):
        self.trainer = SFTTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["eval"] if "eval" in self.dataset else None,
            args=self.args,
        )


def run_sft(args: SFTArguments):
    executor = SFTExecutor(args=args)
    executor.on_train_start()
    executor.fit()
    executor.on_train_end()
