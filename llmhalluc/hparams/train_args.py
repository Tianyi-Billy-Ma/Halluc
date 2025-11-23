from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
from llmhalluc.hparams.base_args import BaseArguments
from llmhalluc.extras.constant import OUTPUT_PATH


@dataclass(kw_only=True)
class TrainArguments(BaseArguments):
    seed: int = 42
    run_name: str
    ### Model
    model_name_or_path: str
    enable_thinking: bool = False
    trust_remote_code: bool = True
    template: str

    ### accelerator
    flash_attn: str = "fa2"
    deepspeed: str = "configs/llamafactory/deepspeed/ds_z0_config.json"

    ### method
    do_train: bool = True
    lora_rank: int = 8
    lora_target: str = "all"

    # wandb
    wandb_project: str = "llamafactory"

    ### dataset
    dataset: str
    dataset_dir: str = "./data"
    cutoff_len: int = 2048
    overwrite_cache: bool = True
    preprocessing_num_workers: int = 8
    dataloader_num_workers: int = 4

    ### output
    output_dir: str | None = None
    logging_steps: int = 10
    save_steps: int = 500
    plot_loss: bool = True
    overwrite_output_dir: bool = True
    save_only_model: bool = False
    report_to: str = "wandb"

    ### train
    stage: str
    finetuning_type: str
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1.0e-4
    num_train_epochs: float = 30.0
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    fp16: bool = True
    ddp_timeout: int = 180000000
    resume_from_checkpoint: Optional[str] = None
    save_total_limit: int = 1
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "loss"
    greater_is_better: bool = False

    # DPO
    pref_loss: str | None = None
    pref_beta: float | None = None

    _pref_loss: str | None = "sigmoid"
    _pref_beta: float | None = 0.1

    ### eval
    eval_dataset: str | None = None
    per_device_eval_batch_size: int = 12
    eval_strategy: str = "steps"
    eval_steps: int = 500
    compute_accuracy: bool = True

    ### Special Tokens
    new_special_tokens_config: str | None = None  # ./configs/llamafactory/token.yaml
    init_special_tokens: str | None = "desc_init"
    force_init_embeddings: bool = False

    # Special Token Initialization
    init_special_tokens: str = ""
    new_special_tokens_config: str = ""

    # Derived fields
    model_name: str = field(init=False)
    exp_path: Path = field(init=False)
    config_path: Path = field(init=False)

    @property
    def yaml_exclude(self):
        return {"model_name", "exp_path", "config_path"}

    def __post_init__(self):
        self.model_name = Path(self.model_name_or_path).name.lower()

        self.exp_path = Path(OUTPUT_PATH) / self.model_name / self.run_name / self.stage

        self.run_name = f"{self.model_name}_{self.run_name}_{self.stage}"

        self.output_dir = str(self.exp_path / "train")
        self.config_path = self.exp_path / "train_config.yaml"
        if self.init_special_tokens:
            self.new_special_tokens_config = str(
                self.exp_path / "special_token_config.yaml"
            )

        self.__dpo_init__()

    def __dpo_init__(self):
        if self.stage.lower() == "dpo":
            self.pref_loss = self.pref_loss or self._pref_loss
            self.pref_beta = self.pref_beta or self._pref_beta
