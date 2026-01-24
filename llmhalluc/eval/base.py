import logging

logger = logging.getLogger(__name__)


def run_eval(config_path: str):
    from lm_eval import simple_evaluate
    from lm_eval.loggers import EvaluationTracker, WandbLogger
    from lm_eval.tasks import TaskManager

    import llmhalluc.eval.metrics  # noqa: F401
    import llmhalluc.models  # noqa: F401
    from llmhalluc.hparams import EvaluationArguments, load_config
    from llmhalluc.utils import is_rank_zero

    eval_config = load_config(config_path)

    model = eval_config.get("model", "hf")
    model_args = eval_config.get("model_args", {})
    if isinstance(model_args, dict):
        model_args = ",".join(f"{k}={v}" for k, v in model_args.items())

    include_path = eval_config.get("include_path")
    tasks = eval_config.get("tasks", "")
    log_samples = eval_config.get("log_samples", True)

    if isinstance(tasks, str):
        tasks = [tasks]

    task_manager = TaskManager(include_path=include_path) if include_path else None

    evaluation_tracker = EvaluationTracker(output_path=eval_config.get("output_path"))

    # Parse wandb configuration
    wandb_args = EvaluationArguments.parse_wandb_args(eval_config.get("wandb_args"))

    # Initialize WandbLogger if wandb_args provided
    wandb_logger = None
    if wandb_args and is_rank_zero():
        try:
            wandb_logger = WandbLogger(wandb_args)
        except Exception as e:
            logger.warning(f"Failed to initialize WandbLogger: {e}")

    logger.info(f"Running evaluation with model={model}, tasks={tasks}")
    results = simple_evaluate(
        model=model,
        model_args=model_args,
        tasks=tasks,
        task_manager=task_manager,
        log_samples=log_samples,
        evaluation_tracker=evaluation_tracker,
    )

    if results is None:
        # In distributed mode, only rank 0 gets results; other ranks get None
        if not is_rank_zero():
            logger.debug("Non-main process: skipping result processing.")
            return None
        logger.error("Evaluation returned no results. Check model/task configuration.")
        return None

    samples = results.get("samples") if log_samples else None
    evaluation_tracker.save_results_aggregated(results=results, samples=samples)

    # Save samples to separate files (one per task)
    if log_samples and samples:
        for task_name in results.get("configs", {}).keys():
            if task_name in samples:
                evaluation_tracker.save_results_samples(
                    task_name=task_name, samples=samples[task_name]
                )
    # Log to wandb after evaluation
    if wandb_logger and results:
        try:
            wandb_logger.post_init(results)
            wandb_logger.log_eval_result()
            if samples:
                wandb_logger.log_eval_samples(samples)
        except Exception as e:
            logger.warning(f"Logging to W&B failed: {e}")

    return results


__all__ = ["run_eval"]
