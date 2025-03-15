from typing import Any, Dict, List, Union, Optional, Generator, Callable

from openai import OpenAI

from agents_manager.models.OpenAi import OpenAi


class DeepSeek(OpenAi):
    def __init__(self, name: str, **kwargs: Any) -> None:
        """
        Initialize the DeepSeek model with a name and optional keyword arguments.

        Args:
            name (str): The name of the DeepSeek model (e.g., "deepseek-chat").
            **kwargs (Any): Additional arguments, including optional "api_key".
        """
        super().__init__(name, **kwargs)

        if name is None:
            raise ValueError("A valid DeepSeek model name is required")

        self.client = OpenAI(
            api_key=kwargs.get("api_key"), base_url="https://api.deepseek.com"
        )
