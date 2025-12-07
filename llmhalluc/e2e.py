import sys
import lm_eval
from lm_eval.config.evaluate_config import EvaluatorConfig
from llamafactory.cli import main as llamafactory_main

from llmhalluc.scripts.e2e_setup import e2e_setup


def run_eval(mode, config_path):
    config = EvaluatorConfig.from_config(config_path)

    task_manager = config.process_tasks()

    results = lm_eval.simple_evaluate(
        model=config.model,
        model_args=config.model_args,
        tasks=config.tasks,
        num_fewshot=config.num_fewshot,
        batch_size=config.batch_size,
        device=config.device,
        task_manager=task_manager,
        log_samples=config.log_samples,
        gen_kwargs=config.gen_kwargs,
        apply_chat_template=config.apply_chat_template,
        system_instruction=config.system_instruction,
    )


def run_llamafactory(mode, config_path, additional: list[str] | None = None):
    assert mode in ["train", "export"]
    sys.argv = [
        "llamafactory-cli",
        mode,
        config_path,
    ] + (additional if additional else [])
    llamafactory_main()


def main():
    argv = ["--format", "else"]
    setup_dict = e2e_setup(argv)
    print(setup_dict)
    run_llamafactory("train", setup_dict["TRAIN_CONFIG_PATH"])
    run_llamafactory("export", setup_dict["MERGE_CONFIG_PATH"])
    run_eval(setup_dict["EVAL_MODE"], setup_dict["EVAL_CONFIG_PATH"])


if __name__ == "__main__":
    main()
