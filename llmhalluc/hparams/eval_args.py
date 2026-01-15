from dataclasses import dataclass, field
from pathlib import Path

from llmhalluc.hparams.base_args import BaseArguments


@dataclass(kw_only=True)
class EvaluationArguments(BaseArguments):
    seed: int = 3

    model: str = "hf"
    model_args: str = ""
    tasks: str = ""
    output_path: str = ""
    wandb_args: str | dict = ""

    log_samples: bool = True
    apply_chat_template: bool = False
    confirm_run_unsafe_code: bool = True
    include_path: str = "./configs/lm_eval/tasks"

    # Logging
    wandb_project: str | None = None
    report_to: str = "none"
    disable_wandb: bool = True

    enable_thinking: bool = False
    run_name: str
    exp_path: Path = field()
    config_path: Path = field(init=False)

    model_name_or_path: str = ""
    tokenizer_name_or_path: str = ""
    adapter_name_or_path: str = ""

    @staticmethod
    def parse_wandb_args(wandb_args: str | dict | None) -> dict | None:
        """Parse wandb_args string or dict to dictionary.

        Args:
            wandb_args: Dictionary or comma-separated key=value pairs string.

        Returns:
            Dictionary of wandb init args, or None if input is empty.
        """
        if not wandb_args:
            return None

        if isinstance(wandb_args, dict):
            return wandb_args

        parsed_args = {}
        for part in wandb_args.split(","):
            if "=" in part:
                k, v = part.split("=", 1)
                parsed_args[k.strip()] = v.strip()

        return parsed_args if parsed_args else None

    @property
    def yaml_exclude(self):
        excludes = {
            "wandb_project",
            "enable_thinking",
            "run_name",
            "exp_path",
            "config_path",
            "model_name_or_path",
            "tokenizer_name_or_path",
            "adapter_name_or_path",
            "report_to",
        }
        if self.report_to != "wandb":
            excludes.add("wandb_args")
        return excludes

    def __post_init__(self):
        task_str = "_".join(self.tasks) if isinstance(self.tasks, list) else self.tasks
        self.output_path = str(self.exp_path / "eval" / task_str / "results.json")

        self._update_model_args()
        self._update_wandb_args()
        self.config_path = self.exp_path / "eval_config.yaml"

    def _update_model_args(self):
        self.model_args = f"pretrained={self.model_name_or_path}"
        if self.adapter_name_or_path:
            self.model_args += f",peft={self.adapter_name_or_path}"
        if self.tokenizer_name_or_path:
            self.model_args += f",tokenizer={self.tokenizer_name_or_path}"
        if self.enable_thinking:
            self.model_args += f",enable_thinking={self.enable_thinking}"

    def _update_wandb_args(self):
        if self.report_to == "wandb":
            self.disable_wandb = False
            self.wandb_args = f"name={self.run_name}"
            if self.wandb_project:
                self.wandb_args += f",project={self.wandb_project}"
