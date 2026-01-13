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

    title_key: str = "title"
    context_key: str = "context"
    question_key: str = "question"
    answer_key: str = "answers"

    title: str | None = None
    context: str | None = None
    question: str | None = None
    answer: str | None = None

    def __post_init__(self):
        self.title_key = self.title or self.title_key
        self.context_key = self.context or self.context_key
        self.question_key = self.question or self.question_key
        self.answer_key = self.answer or self.answer_key

    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        """Convert a Squad v2 example to standard format.

        Args:
            example: Squad v2 example with 'context', 'question', and 'answers' keys.

        Returns:
            Converted example with 'prompt', 'query', and 'response' keys.
        """
        title = example[self.title_key]
        context = example[self.context_key]
        question = example[self.question_key]
        answer = example[self.answer_key]

        return {
            "prompt": QA_INSTRUCTION,
            "query": f"Title: {title}\nContext: {context}\nQuestion: {question}",
            "response": (
                answer["text"][0] if len(answer["text"]) > 0 else "unanswerable"
            ),
        }
