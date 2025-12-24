from pathlib import Path
import json
from omegaconf import OmegaConf
from easydict import EasyDict
from transformers import HfArgumentParser


from llmhalluc.hparams import patch_configs, SFTArguments, patch_sft_config
from .sys_utils import resolve_path


def load_config(path: str) -> dict[str, any]:
    cfg_path = resolve_path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    if cfg_path.suffix == ".yaml":
        cfg = OmegaConf.load(cfg_path)
        return OmegaConf.to_container(cfg, resolve=True) or {}
    elif cfg_path.suffix == ".json":
        with open(cfg_path, "r") as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported config file type: {cfg_path.suffix}")


def apply_overrides(config: dict[str, any], overrides: list[str]) -> dict[str, any]:
    if not overrides:
        return dict(config)

    base_conf = OmegaConf.create(config)
    override_conf = OmegaConf.from_dotlist(overrides)
    merged = OmegaConf.merge(base_conf, override_conf)
    return OmegaConf.to_container(merged, resolve=True)


def save_config(args: dict[str, any], path: str | Path) -> None:
    cfg_path = resolve_path(path)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(args), cfg_path)


def e2e_cfg_setup(args, save_config: bool = True) -> int:
    config = load_config(args.config)
    config = apply_overrides(config, args.override)

    plan = {"train": args.do_train, "merge": args.do_merge, "eval": args.do_eval}

    if not any(plan.values()):
        raise ValueError("At least one stage must be enabled to generate configs.")

    arg_dict = patch_configs(config, plan)
    train_args = arg_dict.train_args
    merge_args = arg_dict.merge_args
    eval_args = arg_dict.merge_args
    extra_args = arg_dict.extra_args

    if save_config:
        save_config(
            train_args.to_yaml(),
            train_args.config_path,
        )
        save_config(
            merge_args.to_yaml(),
            merge_args.config_path,
        )
        save_config(
            eval_args.to_yaml(),
            eval_args.config_path,
        )

        if extra_args:
            save_config(extra_args, train_args.new_special_tokens_config)

    output = {
        "TRAIN_CONFIG_PATH": str(train_args.config_path),
        "MERGE_CONFIG_PATH": str(merge_args.config_path),
        "EVAL_CONFIG_PATH": str(eval_args.config_path),
        "SPECIAL_TOKEN_CONFIG_PATH": str(train_args.new_special_tokens_config or ""),
    }

    return EasyDict(paths=output, args=args)


def hf_cfg_setup(args) -> int:
    setup_dict = e2e_cfg_setup(args, save_config=False)

    train_args = setup_dict.args.train_args

    hf_args = None
    if args.stage == "sft":
        raw_args: dict[str, any] = patch_sft_config(train_args)
        hf_args, *_ = HfArgumentParser(SFTArguments).parse_dict(
            raw_args, allow_extra_keys=True
        )
    return hf_args
