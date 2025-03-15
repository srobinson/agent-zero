from typing import Any, Dict, List, Union, Optional, Generator, Callable

from openai import OpenAI

from agents_manager.models.OpenAi import OpenAi
from agents_manager.Container import Container


class Grok(OpenAi):
    """
    Implementation of the Model interface for X.AI's Grok API.

    This class inherits from OpenAi since Grok uses an OpenAI-compatible API,
    but configures the client to use X.AI's API endpoint.
    """

    def __init__(self, name: str, **kwargs: Any) -> None:
        """
        Initialize the Grok model with a name and optional keyword arguments.

        Args:
            name (str): The name of the Grok model (e.g., "grok-2-latest").
            **kwargs (Any): Additional arguments, including:
                - api_key (str, optional): The API key for X.AI's Grok.
                - base_url (str, optional): The base URL for Grok's API.
                  Defaults to "https://api.x.ai/v1".

        Raises:
            ValueError: If name is None.
        """
        # Initialize with the parent class
        super().__init__(name, **kwargs)

        if name is None:
            raise ValueError("A valid Grok model name is required")

        # Override the client with Grok's API endpoint
        self.client = OpenAI(
            api_key=kwargs.get("api_key"),
            base_url=kwargs.get("base_url", "https://api.x.ai/v1"),
        )

    def generate_response(self) -> Dict[str, Any]:
        """
        Generate a non-streaming response from the Grok model.

        This method uses the OpenAI-compatible API provided by X.AI.

        Returns:
            Dict[str, Any]: A dictionary containing the response content and tool calls.
        """
        return super().generate_response()

    def generate_stream_response(self) -> Generator[Dict[str, Any], None, None]:
        """
        Generate a streaming response from the Grok model.

        This method uses the OpenAI-compatible API provided by X.AI.

        Yields:
            Dict[str, Any]: Chunks of the response containing content and tool calls.
        """
        yield from super().generate_stream_response()

    def set_tools(self, tools: List[Union[Callable, Container]]) -> None:
        """
        Set the tools available to the model.

        Args:
            tools (List[Union[Callable, Container]]): A list of functions or Container instances.

        Note:
            Check X.AI's documentation for any limitations on tool usage
            compared to OpenAI's implementation.
        """
        super().set_tools(tools)
