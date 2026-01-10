"""Squad dataset converter."""

from dataclasses import dataclass
from typing import Any

from ..prompts.QAPrompt import QA_INSTRUCTION
from .base import DatasetConverter


@dataclass
class SquadDatasetConverter(DatasetConverter):
    """Converter for Squad v2 dataset.

    Converts Squad v2 examples to the standard format with prompt, query, and response.

    This converter only handles example transformation, not dataset loading.
    """

    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        """Convert a Squad v2 example to standard format.

        Args:
            example: Squad v2 example with 'context', 'question', and 'answers' keys.

        Returns:
            Converted example with 'prompt', 'query', and 'response' keys.
        """
        return {
            "prompt": QA_INSTRUCTION,
            "query": f"Context: {example['context']}\nQuestion: {example['question']}",
            "response": (
                example["answers"]["text"][0]
                if len(example["answers"]["text"]) > 0
                else "IDK"
            ),
        }
