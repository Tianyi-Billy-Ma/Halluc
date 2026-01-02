import logging
import lm_eval
from lm_eval.tasks import TaskManager
from llmhalluc.utils import load_config
from llmhalluc.utils.sys_utils import is_dir

# Import metrics to register them before evaluation
import llmhalluc.eval.metrics  # noqa: F401

logger = logging.getLogger(__name__)


def run_eval(config_path: str, ddp=False):
    eval_config = load_config(config_path)

    model = eval_config.get("model", "hf")
    model_args = eval_config.get("model_args", {})
    if isinstance(model_args, dict):
        model_args = ",".join(f"{k}={v}" for k, v in model_args.items())

    tasks = eval_config.get("tasks", "")
    if isinstance(tasks, str):
        tasks = [tasks]

    task_manager = None
    include_path = eval_config.get("include_path")
    if include_path:
        task_manager = TaskManager(include_path=include_path)

    logger.info(f"Running evaluation with model={model}, tasks={tasks}")
    results = lm_eval.simple_evaluate(
        model=model,
        model_args=model_args,
        tasks=tasks,
        task_manager=task_manager,
        num_fewshot=eval_config.get("num_fewshot"),
        batch_size=eval_config.get("batch_size", "auto"),
        limit=eval_config.get("limit"),
        log_samples=eval_config.get("log_samples", True),
        random_seed=eval_config.get("seed"),
        numpy_random_seed=eval_config.get("seed"),
        torch_random_seed=eval_config.get("seed"),
        fewshot_random_seed=eval_config.get("seed"),
    )

    output_path = eval_config.get("output_path")
    if output_path:
        import json
        from pathlib import Path

        path = Path(output_path)
        if is_dir(output_path):
            path.mkdir(parents=True, exist_ok=True)
            results_path = path / "results.json"
            samples_path = path / "samples.jsonl"
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            results_path = path
            samples_path = path.parent / f"{path.stem}_samples{path.suffix}"

        # Save main results without samples
        results_to_save = {k: v for k, v in results.items() if k != "samples"}
        with open(results_path, "w") as f:
            json.dump(results_to_save, f, indent=2, default=str)
        logger.info(f"Results saved to {results_path}")

        # Save samples separately if log_samples is enabled
        if eval_config.get("log_samples") and "samples" in results:
            with open(samples_path, "w") as f:
                for task_samples in results["samples"].values():
                    for sample in task_samples:
                        f.write(json.dumps(sample, default=str) + "\n")
            logger.info(f"Samples saved to {samples_path}")

    return results


__all__ = ["run_eval"]
