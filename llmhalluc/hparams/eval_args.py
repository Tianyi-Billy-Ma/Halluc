from dataclasses import dataclass, field
from pathlib import Path
from llmhalluc.hparams.base_args import BaseArguments
from llmhalluc.extras.constant import OUTPUT_PATH


@dataclass(kw_only=True)
class EvaluationArguments(BaseArguments):
    seed: int = 3

    model: str = "hf"
    model_args: str = ""
    tasks: str = ""
    output_path: str = ""
    wandb_args: str = ""

    log_samples: bool = True
    apply_chat_template: bool = True
    confirm_run_unsafe_code: bool = True
    include_path: str = "./configs/lm_eval/tasks"

    wandb_project: str = "llamafactory"
    enable_thinking: bool = False
    stage: str
    run_name: str
    model_path: Path
    model_name: str = field(init=False)
    exp_path: Path = field()
    config_path: Path = field(init=False)

    @property
    def yaml_exclude(self):
        return {
            "wandb_project",
            "enable_thinking",
            "stage",
            "run_name",
            "model_path",
            "model_name",
            "exp_path",
            "config_path",
        }

    def __post_init__(self):
        self.model_path = Path(self.model_path)
        self.model_name = self.model_path.name.lower()

        self.run_name = f"{self.model_name}_{self.run_name}_{self.stage}"
        self.output_path = str(self.exp_path / "eval" / "results.json")

        self.wandb_args = f"project={self.wandb_project},name={self.run_name}"
        self.model_args = (
            f"pretrained={str(self.model_path)},enable_thinking={self.enable_thinking}"
        )
        self.config_path = self.exp_path / "eval_config.yaml"
