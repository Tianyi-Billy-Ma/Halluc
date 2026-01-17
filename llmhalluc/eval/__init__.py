"""Evaluation module for llmhalluc."""

from . import models  # noqa: F401 - Register custom lm_eval models (hf-bt)
from .base import run_eval
from .metrics import rouge1_fn, rouge2_fn, rougeL_fn

__all__ = ["run_eval", "rouge1_fn", "rouge2_fn", "rougeL_fn"]
