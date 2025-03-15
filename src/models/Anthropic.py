from typing import Any, Callable, Dict, Generator, List, Optional, TypeVar, Union

from anthropic import Anthropic as Ap

from agentflow.Container import Container
from agentflow.Model import Model
from agentflow.utils import container_to_json, function_to_json, populate_template

T = TypeVar("T")  # For generic type annotations


class Anthropic(Model):
    def __init__(self, name: str, **kwargs: Any) -> None:
        """
        Initialize the Anthropic model with a name and optional keyword arguments.

        Args:
            name (str): The name of the Anthropic model (e.g., "claude-3-5-sonnet-20241022").
            **kwargs (Any): Additional arguments, including optional "api_key".
        """
        # Set default max_tokens if not provided
        if "max_tokens" not in kwargs:
            kwargs["max_tokens"] = 1024  # Default value

        # Only pass api_key to super().__init__
        api_key = kwargs.get("api_key")
        super().__init__(name, **kwargs)

        if name is None:
            raise ValueError("A valid Anthropic model name is required")

        self.instruction = ""
        self.client = Ap(api_key=api_key)

    def generate_response(self) -> Dict[str, Any]:
        """
        Generate a non-streaming response from the Anthropic model.

        Returns:
            Dict[str, Any]: A dictionary containing the response content and tool calls.
        """
        # Remove api_key from kwargs
        kwargs = self.kwargs.copy()
        if "api_key" in kwargs:
            kwargs.pop("api_key")

        kwargs["stream"] = False
        message = self.client.messages.create(
            model=self.name,
            system=self.instruction,
            messages=self.get_messages(),
            **kwargs,
        )
        return {
            "tool_calls": self.extract_content(message, "tool_use"),
            "content": (
                self.extract_content(message, "text")[0].text
                if self.extract_content(message, "text")
                else ""
            ),
        }

    def generate_stream_response(self) -> Generator[Dict[str, Any], None, None]:
        """
        Generate a streaming response from the Anthropic model with tool_calls and content.

        Yields:
            Dict[str, Any]: Chunks of the response containing content and tool calls.
        """
        # Remove api_key from kwargs
        kwargs = self.kwargs.copy()
        if "api_key" in kwargs:
            kwargs.pop("api_key")

        if "stream" in kwargs:
            kwargs.pop("stream")

        with self.client.messages.stream(
            model=self.name,
            system=self.instruction,
            messages=self.get_messages(),
            **kwargs,
        ) as stream:
            current_content_blocks = {}
            accumulated_json = {}

            current_tool = None  # Track tool call metadata, but don't accumulate input

            for event in stream:
                result = {
                    "content": None,
                    "tool_calls": None,
                }  # Fresh result dict each iteration

                # Handle text tokens as they arrive
                if (
                    event.type == "content_block_delta"
                    and event.delta.type == "text_delta"
                ):
                    result["content"] = (
                        event.delta.text
                    )  # Yield only the current text token

                # Handle tool call start
                elif (
                    event.type == "content_block_start"
                    and event.content_block.type == "tool_use"
                ):
                    current_tool = {
                        "id": event.content_block.id,
                        "name": event.content_block.name,
                        "input": None,
                    }
                    result["tool_calls"] = (
                        current_tool  # Yield tool metadata without input yet
                    )

                # Handle tool call input tokens
                elif (
                    event.type == "content_block_delta"
                    and event.delta.type == "input_json_delta"
                    and current_tool
                ):
                    # Yield the raw partial_json token as it arrives
                    result["tool_calls"] = {
                        "id": current_tool["id"],
                        "name": current_tool["name"],
                        "input": event.delta.partial_json,
                    }

                # Handle block completion
                elif event.type == "content_block_stop":
                    if current_tool:
                        # No input to finalize since we're not appending; just clear the tool
                        current_tool = None
                    # No content to yield here since we're not accumulating

                # Yield the result with the current token (if any)
                if result["content"] or result["tool_calls"]:
                    yield result.copy()  # Return a copy to avoid reference issues

    @staticmethod
    def parse_stream(stream: Any) -> Any:
        """
        Parse a streaming response from Anthropic.

        Args:
            stream: The stream from Anthropic's API.

        Returns:
            Any: The parsed final message.
        """
        current_content_blocks = {}
        accumulated_json = {}

        for event in stream:
            # Handle different event types
            if event.type == "message_start":
                pass

            elif event.type == "content_block_start":
                # Initialize a new content block
                index = event.index
                content_block = event.content_block
                current_content_blocks[index] = content_block

                if content_block.type == "tool_use":
                    accumulated_json[index] = ""

            elif event.type == "content_block_delta":
                index = event.index
                delta = event.delta

                # Handle text deltas
                if delta.type == "text_delta":
                    if (
                        index in current_content_blocks
                        and current_content_blocks[index].type == "text"
                    ):
                        if not hasattr(current_content_blocks[index], "text"):
                            current_content_blocks[index].text = ""
                        current_content_blocks[index].text += delta.text

                # Handle tool use input deltas
                elif delta.type == "input_json_delta":
                    if index in accumulated_json:
                        accumulated_json[index] += delta.partial_json
                        if accumulated_json[index].endswith("}"):
                            try:
                                import json

                                json.loads(accumulated_json[index])
                            except json.JSONDecodeError:
                                pass

            elif event.type == "content_block_stop":
                index = event.index
                if index in current_content_blocks:
                    block_type = current_content_blocks[index].type
                    if block_type == "tool_use" and index in accumulated_json:
                        # Final parse of the complete JSON
                        try:
                            import json

                            json.loads(accumulated_json[index])
                        except json.JSONDecodeError:
                            pass

            elif event.type == "message_delta":
                # Handle updates to the message metadata
                pass

            elif event.type == "message_stop":
                pass

        # Get the final message after streaming completes
        return stream.get_final_message()

    @staticmethod
    def extract_content(response: Any, type_filter: str = "tool_use") -> List[Any]:
        """
        Extract items of a specific type from a Claude API response object.

        Args:
            response: The response object from Claude API
            type_filter (str): The type of items to extract (default: "tool_use")

        Returns:
            List[Any]: A list of filtered items
        """
        items = []
        if hasattr(response, "content") and isinstance(response.content, list):
            for item in response.content:
                if hasattr(item, "type") and item.type == type_filter:
                    items.append(item)
        return items

    def get_tool_format(self) -> Dict[str, Any]:
        """
        Get the format for tool definitions in Anthropic's expected structure.

        Returns:
            Dict[str, Any]: The tool format template.
        """
        return {
            "name": "{name}",
            "description": "{description}",
            "input_schema": {
                "type": "object",
                "properties": "{parameters}",
                "required": "{required}",
            },
        }

    def get_keys_in_tool_output(self, tool_call: Any) -> Dict[str, Any]:
        """
        Extract key information from a tool call.

        Args:
            tool_call (Any): The tool call object from Anthropic.

        Returns:
            Dict[str, Any]: A dictionary containing the tool call's id, name, and arguments.
        """
        return {
            "id": tool_call.id,
            "name": tool_call.name,
            "arguments": tool_call.input,
        }

    @staticmethod
    def _get_tool_call_format() -> Dict[str, Any]:
        """
        Get the format for tool calls in Anthropic's expected structure.

        Returns:
            Dict[str, Any]: The tool call format template.
        """
        return {
            "type": "tool_use",
            "id": "{id}",
            "name": "{name}",
            "input": "{arguments}",
        }

    def get_assistant_message(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format an assistant message with tool calls for inclusion in the message history.

        Args:
            response (Dict[str, Any]): The response containing content and tool calls.

        Returns:
            Dict[str, Any]: A formatted assistant message.
        """
        tool_calls = response["tool_calls"]
        output_tool_calls = []
        for tool_call in tool_calls:
            output = self.get_keys_in_tool_output(tool_call)
            populated_data = populate_template(self._get_tool_call_format(), output)
            output_tool_calls.append(populated_data)

        return {
            "role": "assistant",
            "content": output_tool_calls,
        }

    def get_tool_message(self, tool_responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Format tool responses for inclusion in the message history.

        Args:
            tool_responses (List[Dict[str, Any]]): The responses from tools.

        Returns:
            Dict[str, Any]: A formatted tool message.
        """
        tool_results = []
        for tool_response in tool_responses:
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tool_response["id"],
                    "content": tool_response["tool_result"],
                }
            )

        return {"role": "user", "content": tool_results}

    def set_system_message(self, message: str) -> None:
        """
        Set the system message for the conversation.

        Args:
            message (str): The system message.
        """
        self.instruction = message

    def set_user_message(
        self, message: Union[str, Dict[str, Any], List[Dict[str, Any]]]
    ) -> None:
        """
        Add a user message to the conversation history.

        Args:
            message (Union[str, Dict[str, Any], List[Dict[str, Any]]]):
                The user message as a string, a message dict, or a list of message dicts.
        """
        current_messages = self.get_messages() or []
        if isinstance(message, str):
            user_input = {"role": "user", "content": message}
            current_messages.append(user_input)
        elif isinstance(message, dict):
            current_messages.append(message)
        elif isinstance(message, list):
            current_messages.extend(message)
        self.set_messages(current_messages)

    def set_tools(self, tools: List[Union[Callable, Container]]) -> None:
        """
        Set the tools available to the model.

        Args:
            tools (List[Union[Callable, Container]]): A list of functions or Container instances.
        """
        json_tools: List[Dict[str, Any]] = []
        for tool in tools:
            if callable(tool) and not isinstance(tool, Container):
                json_tools.append(function_to_json(tool, self.get_tool_format()))
            elif isinstance(tool, Container):
                json_tools.append(container_to_json(tool, self.get_tool_format()))

        self.kwargs.update({"tools": json_tools})
