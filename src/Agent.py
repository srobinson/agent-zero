from typing import List, Optional, Callable, Dict, Union, Generator, Any, TypeVar, cast

from agents_manager.Container import Container
from agents_manager.Model import Model
from agents_manager.utils import function_to_json

T = TypeVar("T")  # Generic type for tool results


class Agent:
    """
    Represents an AI agent with a specific model, tools, and instructions.

    An Agent encapsulates a language model with specific instructions and tools,
    providing methods to interact with the model and manage its state.
    """

    def __init__(
        self,
        name: str,
        instruction: str = "",
        model: Optional[Model] = None,
        tools: Optional[List[Union[Callable[..., Any], Container]]] = None,
        tool_choice: Optional[Callable[..., Any]] = None,
    ) -> None:
        """
        Initialize the Agent with a name, instruction, model, tools, and tool choice function.

        Args:
            name (str): The unique name of the agent.
            instruction (str): The system instruction for the agent.
            model (Optional[Model]): The language model instance to use.
            tools (Optional[List[Union[Callable, Container]]]): List of tools available to the agent.
            tool_choice (Optional[Callable]): Function that selects a tool from the list of tools.

        Raises:
            ValueError: If no model is provided or if it's not a valid Model instance.
        """
        if not name:
            raise ValueError("Agent name cannot be empty")

        self.name: str = name
        self.instruction: str = instruction
        self.tools: List[Union[Callable[..., Any], Container]] = tools or []

        if model is None or not isinstance(model, Model):
            raise ValueError("A valid instance of a Model subclass is required")
        self.model: Model = model

        self.tool_choice: Optional[Callable[..., Any]] = tool_choice
        if tool_choice:
            self.set_tool_choice(tool_choice)

    def set_instruction(self, instruction: str) -> None:
        """
        Set the system instruction for the agent.

        Args:
            instruction (str): The system instruction for the agent.
        """
        self.instruction = instruction
        # Update the system message if it's already been set
        if hasattr(self.model, "messages") and self.model.messages:
            for message in self.model.messages:
                if message.get("role") == "system":
                    self.set_system_message(instruction)
                    break

    def get_instruction(self) -> str:
        """
        Get the system instruction for the agent.

        Returns:
            str: The system instruction.
        """
        return self.instruction

    def get_messages(self) -> List[Dict[str, str]]:
        """
        Get the messages for the model.

        Returns:
            List[Dict[str, str]]: The list of message dictionaries.

        Raises:
            ValueError: If messages are not set in the model.
        """
        messages = self.model.get_messages()
        if messages is None:
            raise ValueError("Messages not set in the model")
        return messages

    def set_messages(self, messages: List[Dict[str, str]]) -> None:
        """
        Set the messages for the model.

        Args:
            messages (List[Dict[str, str]]): A list of message dictionaries with "role" and "content".
        """
        self.model.set_messages(messages)

    def add_message(self, role: str, content: str) -> None:
        """
        Add a new message to the model's message history.

        Args:
            role (str): The role of the message sender (e.g., "user", "assistant", "system").
            content (str): The content of the message.
        """
        messages = self.get_messages()
        messages.append({"role": role, "content": content})
        self.set_messages(messages)

    def set_tools(self, tools: List[Union[Callable[..., Any], Container]]) -> None:
        """
        Set the tools for the agent and update the model's tools.

        Args:
            tools (List[Union[Callable, Container]]): List of tools to be used by the agent.
        """
        self.tools = tools
        self.model.set_tools(tools)

    def get_tools(self) -> List[Union[Callable[..., Any], Container]]:
        """
        Get the tools for the agent.

        Returns:
            List[Union[Callable, Container]]: The list of tools.
        """
        return self.tools

    def add_tool(self, tool: Union[Callable[..., Any], Container]) -> None:
        """
        Add a new tool to the agent's tools.

        Args:
            tool (Union[Callable, Container]): The tool to add.
        """
        if tool not in self.tools:
            self.tools.append(tool)
            self.model.set_tools(self.tools)

    def remove_tool(self, tool_name: str) -> bool:
        """
        Remove a tool from the agent's tools by name.

        Args:
            tool_name (str): The name of the tool to remove.

        Returns:
            bool: True if the tool was found and removed, False otherwise.
        """
        for i, tool in enumerate(self.tools):
            if (
                isinstance(tool, Callable)
                and hasattr(tool, "__name__")
                and tool.__name__ == tool_name
            ) or (isinstance(tool, Container) and tool.name == tool_name):
                self.tools.pop(i)
                self.model.set_tools(self.tools)
                return True
        return False

    def get_model(self) -> Model:
        """
        Get the model instance for the agent.

        Returns:
            Model: The model instance.
        """
        return self.model

    def set_model(self, model: Model) -> None:
        """
        Set the model instance for the agent.

        Args:
            model (Model): An instance of a concrete Model subclass.

        Raises:
            ValueError: If the provided object is not a valid Model instance.
        """
        if model is None or not isinstance(model, Model):
            raise ValueError("A valid instance of a Model subclass is required")
        self.model = model

    def set_tool_choice(self, tool_choice: Callable[..., Any]) -> None:
        """
        Set the tool choice function for the agent.

        Args:
            tool_choice (Callable): The function that selects a tool from the list of tools.
        """
        self.tool_choice = tool_choice
        self.model.set_kwargs({"tool_choice": function_to_json(tool_choice)})

    def get_response(self) -> Dict[str, Any]:
        """
        Generate a non-streaming response from the model.

        Returns:
            Dict[str, Any]: The response from the model.

        Raises:
            ValueError: If messages are not set before generating a response.
        """
        if not hasattr(self.model, "messages") or self.model.messages is None:
            raise ValueError("Messages must be set before generating a response")
        return self.model.generate_response()

    def get_stream_response(self) -> Generator[Dict[str, Any], None, None]:
        """
        Generate a streaming response from the model.

        Yields:
            Dict[str, Any]: Chunks of the response from the model.

        Raises:
            ValueError: If messages are not set before generating a response.
        """
        if not hasattr(self.model, "messages") or self.model.messages is None:
            raise ValueError("Messages must be set before generating a response")
        yield from self.model.generate_stream_response()

    def stream(self, query: str, callback=None, tool_callback=None):
        """
        Stream a response from the agent with optional callbacks.

        Args:
            query (str): The query to send to the agent
            callback (callable, optional): A function to call for each content chunk
                If not provided, content will be printed to stdout
            tool_callback (callable, optional): A function to call when tool calls are detected

        Returns:
            dict: The complete response with content and tool calls
        """
        import sys

        # Initialize the agent with the query
        self.set_system_message(self.instruction)
        self.set_user_message(query)

        # Default callback prints to stdout
        if callback is None:
            callback = lambda text: sys.stdout.write(text) or sys.stdout.flush()

        # Track the complete response
        full_response = ""
        all_tool_calls = []

        # Stream the response
        for chunk in self.get_stream_response():
            # Handle content
            if chunk.get("content"):
                content = chunk["content"] or ""
                callback(content)
                full_response += content

            # Handle tool calls
            if chunk.get("tool_calls"):
                for tool_call in chunk["tool_calls"]:
                    if tool_call not in all_tool_calls:
                        all_tool_calls.append(tool_call)
                        if tool_callback:
                            tool_callback(tool_call)

        return {"content": full_response, "tool_calls": all_tool_calls}

    def set_system_message(self, message: str) -> None:
        """
        Set the system message for the agent.

        Args:
            message (str): The system message.
        """
        self.model.set_system_message(message)

    def set_user_message(self, message: str) -> None:
        """
        Set the user message for the agent.

        Args:
            message (str): The user message.
        """
        self.model.set_user_message(message)

    def clear_messages(self) -> None:
        """
        Clear all messages in the model's message history.
        """
        self.model.set_messages([])
        # Re-add the system message if instruction is set
        if self.instruction:
            self.set_system_message(self.instruction)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the agent to a dictionary representation.

        Returns:
            Dict[str, Any]: Dictionary representation of the agent.
        """
        return {
            "name": self.name,
            "instruction": self.instruction,
            "tools_count": len(self.tools),
            "model_type": type(self.model).__name__,
        }

    def __str__(self) -> str:
        """
        Get a string representation of the agent.

        Returns:
            str: String representation of the agent.
        """
        return f"Agent(name='{self.name}', model={type(self.model).__name__}, tools={len(self.tools)})"

    def __repr__(self) -> str:
        """
        Get a detailed string representation of the agent.

        Returns:
            str: Detailed string representation of the agent.
        """
        return f"Agent(name='{self.name}', instruction='{self.instruction[:20]}...', model={type(self.model).__name__}, tools={len(self.tools)})"
