"""Training callbacks for Hugging Face Trainer."""

from transformers import EarlyStoppingCallback, TrainerCallback


def get_early_stopping_callback(
    early_stopping_patience: int = 3,
    early_stopping_threshold: float = 0.0,
) -> TrainerCallback:
    """Create an EarlyStoppingCallback.

    Args:
        early_stopping_patience: Number of evaluations with no improvement
            before training is stopped.
        early_stopping_threshold: Minimum change in the monitored metric
            to qualify as an improvement.

    Returns:
        EarlyStoppingCallback configured with the given parameters.
    """
    return EarlyStoppingCallback(
        early_stopping_patience=early_stopping_patience,
        early_stopping_threshold=early_stopping_threshold,
    )


def get_callbacks(args) -> list[TrainerCallback]:
    """Build list of trainer callbacks based on arguments.

    Args:
        args: Training arguments containing callback configuration.

    Returns:
        List of TrainerCallback instances to use during training.
    """
    callbacks = []

    if getattr(args, "early_stopping", False):
        callbacks.append(
            get_early_stopping_callback(
                early_stopping_patience=getattr(args, "early_stopping_patience", 3),
                early_stopping_threshold=getattr(args, "early_stopping_threshold", 0.0),
            )
        )

    return callbacks
