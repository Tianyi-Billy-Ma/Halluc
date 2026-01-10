from dataclasses import dataclass, field
from pathlib import Path

from llmhalluc.extras.constant import MODEL_PATH, OUTPUT_PATH
from llmhalluc.hparams.base_args import BaseArguments


@dataclass(kw_only=True)
class MergeArguments(BaseArguments):
    ### Model
    model_name_or_path: str
    adapter_name_or_path: str = ""
    trust_remote_code: bool = True
    template: str

    ### Export
    export_dir: str = ""
    export_hub_model_id: str | None = None
    export_size: int = 2
    export_device: str = "auto"
    export_legacy_format: bool = False

    ### Special Tokens
    init_special_tokens: bool = False
    new_special_tokens_config: dict[str, str] | None = None
    replace_text: dict[str, str] | None = None

    stage: str
    run_name: str
    model_name: str = field(init=False)
    exp_path: Path = field(init=False)
    config_path: Path = field(init=False)

    @property
    def yaml_exclude(self):
        excludes = {"stage", "run_name", "exp_path", "config_path", "model_name"}
        # Exclude special token fields if not enabled
        if not self.init_special_tokens:
            excludes.add("init_special_tokens")
            excludes.add("new_special_tokens_config")
            excludes.add("replace_text")
        return excludes

    def __post_init__(self):
        self.model_name = Path(self.model_name_or_path).name.lower()

        self.exp_path = Path(OUTPUT_PATH) / self.model_name / self.run_name / self.stage
        self.run_name = f"{self.model_name}_{self.run_name}_{self.stage}"
        self.config_path = self.exp_path / "merge_config.yaml"
        self.export_dir = str(Path(MODEL_PATH) / self.run_name)
