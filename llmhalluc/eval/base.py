import logging

from lm_eval import simple_evaluate
from lm_eval.loggers import EvaluationTracker
from lm_eval.tasks import TaskManager

# Import metrics to register them before evaluation
import llmhalluc.eval.metrics  # noqa: F401
from llmhalluc.hparams import load_config

logger = logging.getLogger(__name__)


def run_eval(config_path: str, ddp=False):
    eval_config = load_config(config_path)

    model = eval_config.get("model", "hf")
    model_args = eval_config.get("model_args", {})
    if isinstance(model_args, dict):
        model_args = ",".join(f"{k}={v}" for k, v in model_args.items())

    output_path = eval_config.get("output_path")
    tasks = eval_config.get("tasks", "")

    if isinstance(tasks, str):
        tasks = [tasks]

    task_manager = None
    include_path = eval_config.get("include_path")
    if include_path:
        task_manager = TaskManager(include_path=include_path)

    evaluation_tracker = EvaluationTracker(output_path=output_path)

    logger.info(f"Running evaluation with model={model}, tasks={tasks}")
    results = simple_evaluate(
        model=model,
        model_args=model_args,
        tasks=tasks,
        task_manager=task_manager,
        wandb_args=eval_config.get("wandb_args"),
        # num_fewshot=eval_config.get("num_fewshot"),
        # batch_size=eval_config.get("batch_size", "auto"),
        # limit=eval_config.get("limit"),
        log_samples=eval_config.get("log_samples", True),
        # random_seed=eval_config.get("seed"),
        # numpy_random_seed=eval_config.get("seed"),
        # torch_random_seed=eval_config.get("seed"),
        # fewshot_random_seed=eval_config.get("seed"),
        evaluation_tracker=evaluation_tracker,
    )

    evaluation_tracker.save_results_aggregated(
        results=results, samples=results.get("samples")
    )
    return results


__all__ = ["run_eval"]
