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
    wandb_args: str = ""

    log_samples: bool = True
    apply_chat_template: bool = False
    confirm_run_unsafe_code: bool = True
    include_path: str = "./configs/lm_eval/tasks"

    wandb_project: str = "llamafactory"
    disable_wandb: bool = True

    enable_thinking: bool = False
    stage: str
    run_name: str
    model_path: Path
    model_name: str = field(init=False)
    exp_path: Path = field()
    config_path: Path = field(init=False)

    @property
    def yaml_exclude(self):
        excludes = {
            "wandb_project",
            "enable_thinking",
            "stage",
            "run_name",
            "model_path",
            "model_name",
            "exp_path",
            "config_path",
            "disable_wandb",
        }
        if self.disable_wandb:
            excludes.add("wandb_args")
        return excludes

    def __post_init__(self):
        self.model_path = Path(self.model_path)
        self.model_name = self.model_path.name.lower()

        self.run_name = f"{self.model_name}_{self.run_name}_{self.stage}"

        if isinstance(self.tasks, list):
            task_str = "_".join(self.tasks)
        else:
            task_str = self.tasks
        self.output_path = str(self.exp_path / "eval" / task_str / "results.json")

        self.model_args = f"pretrained={str(self.model_path)}"
        if self.enable_thinking:
            self.model_args += f",enable_thinking={self.enable_thinking}"
        if not self.disable_wandb:
            self.wandb_args = f"project={self.wandb_project},name={self.run_name}"
        else:
            self.wandb_args = None
        self.config_path = self.exp_path / "eval_config.yaml"
