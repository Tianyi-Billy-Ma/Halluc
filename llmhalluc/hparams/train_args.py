from dataclasses import dataclass, field
from pathlib import Path


import torch

from llmhalluc.extras import OUTPUT_PATH
from llmhalluc.hparams.base_args import BaseArguments


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
    deepspeed: str | None = "./configs/deepspeed/ds_z0_config.json"

    ### method
    do_train: bool = True
    do_eval: bool = False
    do_predict: bool = False

    ### Lora
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target: str = "all"

    # QLoRA (4-bit quantization)
    load_in_4bit: bool = False
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True

    # wandb
    wandb_project: str = ""
    report_to: str = "none"

    ### dataset
    dataset: str
    dataset_dir: str = "./data"
    cutoff_len: int = 2048
    overwrite_cache: bool = True
    load_from_cache_file: bool = False
    converter: str = ""
    preprocessing_num_workers: int = 8
    dataloader_num_workers: int = 4

    ### output
    output_dir: str | None = None
    logging_steps: int = 10
    save_steps: int = 500
    plot_loss: bool = True
    overwrite_output_dir: bool = True
    save_only_model: bool = False

    ### train
    stage: str
    finetuning_type: str
    per_device_train_batch_size: int = 16
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1.0e-4
    num_train_epochs: float = 3
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    fp16: bool = True
    bf16: bool = False
    ddp_timeout: int = 180000000
    resume_from_checkpoint: str | None = None
    save_total_limit: int = 1
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "loss"
    greater_is_better: bool = False

    # DPO
    pref_loss: str = "sigmoid"
    pref_beta: float = 0.1

    ### eval
    eval_dataset: str | None = None
    per_device_eval_batch_size: int | None = 32
    eval_strategy: str | None = "steps"
    eval_steps: int | None = 500
    compute_accuracy: bool = True

    # Early Stopping
    early_stopping: bool = False
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001

    # Special Token Initialization
    init_special_tokens: bool = False
    new_special_tokens_config: dict[str, str] | None = None
    replace_text: dict[str, str] | None = None

    # Reward Function Arguments
    reward_func_args: dict[str, str | int | float | bool] = field(
        default_factory=lambda: {
            "outcome_weight": 1.0,
            "process_weight": 0.7,
            "backtrack_weight": 0.6,
            "format_weight": 0.3,
            "correction_bonus": 0.4,
            "unnecessary_penalty": 0.2,
            "efficiency_weight": 0.25,
            "failed_correction_penalty": 0.3,
            "use_curriculum": False,
            "enable_process_rewards": False,
            "enable_format_rewards": True,
            "max_backtracks": 20,
            "backtrack_token_id": None,
        },
        metadata={"help": "Arguments to pass to reward functions"},
    )

    backtrack_token: str = "<|BACKTRACK|>"
    # Derived fields
    model_name: str = field(init=False)
    exp_path: Path = field(init=False)
    config_path: Path = field(init=False)

    # PPO fields
    reward_model: str = ""
    reward_model_type: str = ""

    # SFT
    compute_only_loss: bool = False

    @property
    def yaml_exclude(self):
        excludes = {
            "model_name",
            "exp_path",
            "config_path",
            "wandb_project",
            "reward_model",
            "reward_model_type",
            "pref_loss",
            "pref_beta",
        }
        if self.stage.lower() == "ppo":
            excludes.remove("reward_model")
            excludes.remove("reward_model_type")
        elif self.stage.lower() == "dpo":
            excludes.remove("pref_loss")
            excludes.remove("pref_beta")
        if not self.init_special_tokens:
            excludes.add("init_special_tokens")
            excludes.add("new_special_tokens_config")
            excludes.add("replace_text")
        if self.finetuning_type not in ["lora", "qlora"]:
            excludes.add("lora_rank")
            excludes.add("lora_alpha")
            excludes.add("lora_dropout")
            excludes.add("lora_target")
        if self.finetuning_type != "qlora" and not self.load_in_4bit:
            excludes.add("load_in_4bit")
            excludes.add("bnb_4bit_compute_dtype")
            excludes.add("bnb_4bit_quant_type")
            excludes.add("bnb_4bit_use_double_quant")
        return excludes

    def __post_init__(self):
        self.model_name = Path(self.model_name_or_path).name.lower()

        self.exp_path = None
        if self.output_dir is None:
            self.exp_path = (
                Path(OUTPUT_PATH) / self.model_name / self.run_name / self.stage
            )
            self.output_dir = str(self.exp_path / "train")
        else:
            self.exp_path = Path(self.output_dir).parent

        model_name, run_name, stage = self.model_name, self.run_name, self.stage

        self.run_name = f"{model_name}_{run_name}_{stage}"
        self.config_path = self.exp_path / "train_config.yaml"

        if self.eval_dataset:
            self.do_eval = True

        if self.stage.lower() == "ppo":
            # PPO does not support load best model at end
            self.load_best_model_at_end = False
            self.eval_dataset = None  # No eval dataset for PPO
            self.do_eval = False
            self.do_predict = False
            self.eval_steps = None
            self.eval_strategy = None
            self.per_device_eval_batch_size = None
            self.compute_accuracy = False

        if torch.cuda.device_count() <= 1:
            self.deepspeed = None

        # self.report_to = "none"
