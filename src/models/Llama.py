from typing import Any, Dict, List, Union, Optional, Generator, Callable

from openai import OpenAI

from agents_manager.models.OpenAi import OpenAi
from agents_manager.Container import Container


class Llama(OpenAi):
    """
    Implementation of the Model interface for Llama API.

    This class inherits from OpenAi since Llama uses an OpenAI-compatible API,
    but configures the client to use Llama's API endpoint.
    """

    def __init__(self, name: str, **kwargs: Any) -> None:
        """
        Initialize the Llama model with a name and optional keyword arguments.

        Args:
            name (str): The name of the Llama model (e.g., "llama3.1-70b").
            **kwargs (Any): Additional arguments, including:
                - api_key (str, optional): The API key for Llama API.
                - base_url (str, optional): The base URL for Llama's API.
                  Defaults to "https://api.llama-api.com".

        Raises:
            ValueError: If name is None.
        """
        # Initialize with the parent class
        super().__init__(name, **kwargs)

        if name is None:
            raise ValueError("A valid Llama model name is required")

        # Override the client with Llama's API endpoint
        self.client = OpenAI(
            api_key=kwargs.get("api_key"),
            base_url=kwargs.get("base_url", "https://api.llama-api.com"),
        )

    def generate_response(self) -> Dict[str, Any]:
        """
        Generate a non-streaming response from the Llama model.

        This method uses the OpenAI-compatible API provided by Llama.

        Returns:
            Dict[str, Any]: A dictionary containing the response content and tool calls.
        """
        return super().generate_response()

    def generate_stream_response(self) -> Generator[Dict[str, Any], None, None]:
        """
        Generate a streaming response from the Llama model.

        This method uses the OpenAI-compatible API provided by Llama.

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
            Check Llama's documentation for any limitations on tool usage
            compared to OpenAI's implementation.
        """
        super().set_tools(tools)
