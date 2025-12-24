"""SFT dataset converter for TRL SFTTrainer."""

from dataclasses import dataclass
from typing import Any

from .base import DatasetConverter


@dataclass
class SFTDatasetConverter(DatasetConverter):
    """Converter that transforms {prompt, query, response} to TRL messages format.

    This converter takes the standard format produced by other converters
    and converts it to the chat messages format expected by TRL's SFTTrainer.

    Args:
        prompt_key: Key for the system prompt in input examples.
        query_key: Key for the user query in input examples.
        response_key: Key for the assistant response in input examples.
    """

    prompt_key: str = "prompt"
    query_key: str = "query"
    response_key: str = "response"

    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        """Convert an example to TRL messages format.

        Args:
            example: Input example with prompt, query, and response fields.

        Returns:
            Example with 'messages' field containing list of role/content dicts.
        """
        messages = []

        # Add system message if prompt exists
        prompt_content = example.get(self.prompt_key, "")
        if prompt_content:
            messages.append({
                "role": "system",
                "content": prompt_content
            })

        # Add user message
        query_content = example.get(self.query_key, "")
        if query_content:
            messages.append({
                "role": "user",
                "content": query_content
            })

        # Add assistant response
        response_content = example.get(self.response_key, "")
        messages.append({
            "role": "assistant",
            "content": response_content
        })

        return {"messages": messages}
