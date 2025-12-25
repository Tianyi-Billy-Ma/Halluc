import os
import subprocess
import logging
from copy import deepcopy
from llmhalluc.utils import load_config


logger = logging.getLogger(__name__)


def run_eval(config_path: str, ddp=False):
    eval_config = load_config(config_path)

    args = []

    for key, val in eval_config.items():
        if isinstance(val, bool) and val:
            args.append(f"--{key}")
        elif isinstance(val, str):
            args.append(f"--{key}")
            args.append(val)
        elif isinstance(val, int):
            args.append(f"--{key}")
            args.append(str(val))
        elif isinstance(val, dict):
            sub_args = ",".join([f"{k}={v}" for k, v in val.items()])
            args.append(f"--{key}")
            args.append(sub_args)
        else:
            raise ValueError(f"Invalid value type: {type(val)}")

    if ddp:
        cmd = ["accelerate", "launch", "-m", "lm_eval"] + args
    else:
        cmd = ["lm_eval"] + args
    logger.info(f"Running evaluation with command: \n{cmd}")
    subprocess.run(cmd, env=deepcopy(os.environ), check=True)


__all__ = ["run_eval"]
