"""DPO dataset converter for TRL DPOTrainer."""

from dataclasses import dataclass
from typing import Any

from .base import DatasetConverter


@dataclass
class DPODatasetConverter(DatasetConverter):
    """Converter that transforms datasets to TRL DPO format.

    DPO requires datasets with prompt, chosen, and rejected columns.
    This converter maps configurable input keys to the standard DPO format.

    Supports both:
    - Standard format: plain text prompt, chosen, rejected
    - Conversational format: messages-style prompt with chosen/rejected responses

    Args:
        prompt_key: Key for the prompt/input in the source dataset.
        chosen_key: Key for the preferred/chosen response.
        rejected_key: Key for the less preferred/rejected response.
    """

    prompt_key: str = "prompt"
    query_key: str = "query"
    chosen_key: str = "chosen"
    rejected_key: str = "rejected"

    prompt: str | None = None
    query: str | None = None
    chosen: str | None = None
    rejected: str | None = None

    def __post_init__(self):
        self.prompt_key = self.prompt or self.prompt_key
        self.query_key = self.query or self.query_key
        self.chosen_key = self.chosen or self.chosen_key
        self.rejected_key = self.rejected or self.rejected_key

    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        """Convert an example to TRL DPO format.

        Args:
            example: Input example with prompt, chosen, and rejected fields.

        Returns:
            Example with 'prompt', 'chosen', 'rejected' fields for DPOTrainer.
        """
        prompt = example.get(self.prompt_key, "")
        query = example.get(self.query_key, "")
        chosen = example.get(self.chosen_key, "")
        rejected = example.get(self.rejected_key, "")

        prompt = query + "\n" + prompt if query else prompt

        return {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        }
