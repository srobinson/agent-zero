from typing import List, Dict, Any, Union, Optional, Generator, Callable

from openai import OpenAI
from openai.types.chat import ChatCompletion

from agents_manager.Container import Container
from agents_manager.Model import Model
from agents_manager.utils import populate_template, function_to_json, container_to_json


class OpenAi(Model):
    def __init__(self, name: str, **kwargs: Any) -> None:
        """
        Initialize the OpenAi model with a name and optional keyword arguments.

        Args:
            name (str): The name of the OpenAI model (e.g., "gpt-3.5-turbo").
            **kwargs (Any): Additional arguments, including optional "api_key".
        """
        super().__init__(name, **kwargs)

        if name is None:
            raise ValueError("A valid OpenAI model name is required")

        self.client = OpenAI(
            api_key=kwargs.get("api_key"),  # type: Optional[str]
        )
        self.has_tools_execute = False

    def generate_response(self) -> Dict[str, Any]:
        """
        Generate a non-streaming response from the OpenAI model.

        Returns:
            Dict[str, Any]: A dictionary containing the response content and tool calls.
        """
        # Remove api_key from kwargs
        kwargs = self.kwargs.copy()
        if "api_key" in kwargs:
            kwargs.pop("api_key")

        kwargs["stream"] = False
        response = self.client.chat.completions.create(
            model=self.name,
            messages=self.get_messages(),
            **kwargs,
        )
        message = response.choices[0].message
        return {
            "tool_calls": message.tool_calls,
            "content": message.content,
        }

    def generate_stream_response(self) -> Generator[Dict[str, Any], None, None]:
        """
        Generate a streaming response from the OpenAI model.

        Yields:
            Dict[str, Any]: Chunks of the response containing content and tool calls.
        """
        # Remove api_key from kwargs
        kwargs = self.kwargs.copy()
        if "api_key" in kwargs:
            kwargs.pop("api_key")

        kwargs["stream"] = True
        response = self.client.chat.completions.create(
            model=self.name,
            messages=self.get_messages(),
            **kwargs,
        )

        final_tool_calls = {}
        accumulated_content = ""  # Track accumulated content

        for chunk in response:
            result = {
                "tool_calls": [],
                "content": "",
            }

            # Process tool calls
            for tool_call in chunk.choices[0].delta.tool_calls or []:
                index = tool_call.index
                if index not in final_tool_calls:
                    final_tool_calls[index] = tool_call
                final_tool_calls[
                    index
                ].function.arguments += tool_call.function.arguments
                result["tool_calls"] = list(final_tool_calls.values())

            # Process content - only yield the new content, not accumulated
            if chunk.choices[0].delta.content is not None:
                # Store the current chunk's content
                current_content = chunk.choices[0].delta.content
                # Update accumulated content (for tool calls that need the full content)
                accumulated_content += current_content
                # But only yield the current chunk's content
                result["content"] = current_content

            yield result

    def get_tool_format(self) -> Dict[str, Any]:
        """
        Get the format for tool definitions in OpenAI's expected structure.

        Returns:
            Dict[str, Any]: The tool format template.
        """
        return {
            "type": "function",
            "function": {
                "name": "{name}",
                "description": "{description}",
                "parameters": {
                    "type": "object",
                    "properties": "{parameters}",
                    "required": "{required}",
                    "additionalProperties": False,
                },
                "strict": True,
            },
        }

    @staticmethod
    def _get_tool_call_format() -> Dict[str, Any]:
        """
        Get the format for tool calls in OpenAI's expected structure.

        Returns:
            Dict[str, Any]: The tool call format template.
        """
        return {
            "id": "{id}",
            "type": "function",
            "function": {"name": "{name}", "arguments": "{arguments}"},
        }

    def get_keys_in_tool_output(self, tool_call: Any) -> Dict[str, Any]:
        """
        Extract key information from a tool call.

        Args:
            tool_call (Any): The tool call object from OpenAI.

        Returns:
            Dict[str, Any]: A dictionary containing the tool call's id, name, and arguments.
        """
        return {
            "id": tool_call.id,
            "name": tool_call.function.name,
            "arguments": tool_call.function.arguments,
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
            "content": response["content"] or "",
            "tool_calls": output_tool_calls,
        }

    def get_tool_message(
        self, tool_responses: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Format tool responses for inclusion in the message history.

        Args:
            tool_responses (List[Dict[str, Any]]): The responses from tools.

        Returns:
            List[Dict[str, Any]]: A list of formatted tool messages.
        """
        tool_results = []
        for tool_response in tool_responses:
            tool_results.append(
                {
                    "role": "tool",
                    "content": tool_response["tool_result"],
                    "tool_call_id": tool_response["id"],
                }
            )

        return tool_results

    def set_system_message(self, message: str) -> None:
        """
        Set the system message for the conversation.

        Args:
            message (str): The system message.
        """
        self.set_messages(
            [
                {
                    "role": "system",
                    "content": message,
                }
            ]
        )

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
            user_input = message
            current_messages.append(user_input)
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
