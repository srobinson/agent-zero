import json
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Generator, Callable, Union, TypeVar

T = TypeVar("T")  # Generic type for responses


class Model(ABC):
    """
    Abstract base class for language model implementations.

    This class defines the interface that all model implementations must follow.
    It provides methods for managing messages, generating responses, and handling tools.
    Concrete subclasses must implement the abstract methods to provide specific
    functionality for different language model providers.
    """

    def __init__(self, name: str, **kwargs: Any) -> None:
        """
        Initialize the Model with a name and optional keyword arguments.

        Args:
            name (str): The name of the model.
            **kwargs (Any): Additional keyword arguments specific to the model implementation.
        """
        self.messages: Optional[str] = None
        self.name: str = name
        self.kwargs: Dict[str, Any] = kwargs

    def set_messages(self, messages: List[Dict[str, str]]) -> None:
        """
        Set the messages for the model.

        Args:
            messages (List[Dict[str, str]]): A list of message dictionaries with "role" and "content".
        """
        if not isinstance(messages, list):
            raise TypeError("Messages must be a list of dictionaries")

        for message in messages:
            if (
                not isinstance(message, dict)
                or "role" not in message
                or "content" not in message
            ):
                raise ValueError(
                    "Each message must be a dictionary with 'role' and 'content' keys"
                )

        self.messages = json.dumps(messages)

    def get_messages(self) -> Optional[List[Dict[str, str]]]:
        """
        Get the messages for the model.

        Returns:
            Optional[List[Dict[str, str]]]: The list of message dictionaries if set, else None.
        """
        if self.messages is None or self.messages == "":
            return None
        return json.loads(self.messages)

    def clear_messages(self) -> None:
        """
        Clear the messages for the model.
        """
        self.messages = None

    def set_kwargs(self, kwargs: Dict[str, Any]) -> None:
        """
        Update the model's keyword arguments by merging with existing ones.

        Args:
            kwargs (Dict[str, Any]): New keyword arguments to merge with existing ones.
        """
        if not isinstance(kwargs, dict):
            raise TypeError("kwargs must be a dictionary")

        self.kwargs = {**self.kwargs, **kwargs}

    def get_kwargs(self) -> Dict[str, Any]:
        """
        Get the model's keyword arguments.

        Returns:
            Dict[str, Any]: The model's keyword arguments.
        """
        return self.kwargs

    @abstractmethod
    def generate_response(self) -> Dict[str, Any]:
        """
        Generate a non-streaming response based on the model's implementation.

        Returns:
            Dict[str, Any]: The response, containing at least 'content' and 'tool_calls' keys.

        Raises:
            ValueError: If messages are not set or are invalid.
        """
        pass

    @abstractmethod
    def generate_stream_response(self) -> Generator[Dict[str, Any], None, None]:
        """
        Generate a streaming response based on the model's implementation.

        Yields:
            Dict[str, Any]: Chunks of the response, each containing at least 'content' key.

        Raises:
            ValueError: If messages are not set or are invalid.
        """
        pass

    @abstractmethod
    def get_tool_format(self) -> Dict[str, Any]:
        """
        Get the format for the tool call.

        Returns:
            Dict[str, Any]: The tool call format specific to the model implementation.
        """
        pass

    @abstractmethod
    def get_keys_in_tool_output(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract key information from a tool call.

        Args:
            tool_call (Dict[str, Any]): The tool call data.

        Returns:
            Dict[str, Any]: The parsed tool call data, containing at least 'id', 'name', and 'arguments' keys.
        """
        pass

    @abstractmethod
    def get_assistant_message(
        self, response: Any
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Get the assistant message for prepending to the response.

        Args:
            response (Any): The response from the model.

        Returns:
            Union[Dict[str, Any], List[Dict[str, Any]]]: The assistant message(s).
        """
        pass

    @abstractmethod
    def get_tool_message(
        self, tool_responses: List[Dict[str, Any]]
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Get the tool message for appending to the response.

        Args:
            tool_responses (List[Dict[str, Any]]): The tool responses.

        Returns:
            Union[Dict[str, Any], List[Dict[str, Any]]]: The tool message(s).
        """
        pass

    @abstractmethod
    def set_system_message(self, message: str) -> None:
        """
        Set the system message for the model.

        Args:
            message (str): The system message.

        Raises:
            ValueError: If the message is invalid.
        """
        pass

    @abstractmethod
    def set_user_message(self, message: str) -> None:
        """
        Set the user message for the model.

        Args:
            message (str): The user message.

        Raises:
            ValueError: If the message is invalid.
        """
        pass

    @abstractmethod
    def set_tools(self, tools: List[Callable]) -> None:
        """
        Set the tools for the model.

        Args:
            tools (List[Callable]): The tools.

        Raises:
            ValueError: If the tools are invalid.
        """
        pass

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the model to a dictionary representation.

        Returns:
            Dict[str, Any]: Dictionary representation of the model.
        """
        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "has_messages": self.messages is not None and self.messages != "",
            "kwargs_count": len(self.kwargs),
        }

    def __str__(self) -> str:
        """
        Get a string representation of the model.

        Returns:
            str: String representation of the model.
        """
        return f"Model(name='{self.name}', type={self.__class__.__name__})"

    def __repr__(self) -> str:
        """
        Get a detailed string representation of the model.

        Returns:
            str: Detailed string representation of the model.
        """
        return f"Model(name='{self.name}', type={self.__class__.__name__}, kwargs={len(self.kwargs)})"
