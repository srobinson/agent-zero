from typing import Any, Callable, Dict, Generator, List, Optional, Union

from google import genai
from google.genai import types

from agentflow.Container import Container
from agentflow.Model import Model
from agentflow.utils import container_to_json, function_to_json, populate_template


class Genai(Model):
    """
    Implementation of the Model interface for Google's Generative AI (Gemini) API.

    This class provides methods to interact with Google's Gemini models,
    supporting both streaming and non-streaming responses, as well as tool usage.
    """

    def __init__(self, name: str, **kwargs: Any) -> None:
        """
        Initialize the Genai model with a name and optional keyword arguments.

        Args:
            name (str): The name of the Genai model (e.g., "gemini-2.0-flash").
            **kwargs (Any): Additional arguments, including:
                - api_key (str, optional): The API key for Google Genai.
                - api_version (str, optional): The API version to use.
                - project (str, optional): The Google Cloud project ID.
                - location (str, optional): The Google Cloud location.
                - vertexai (bool, optional): Whether to use VertexAI.

        Raises:
            ValueError: If name is None.
        """
        super().__init__(name, **kwargs)

        if name is None:
            raise ValueError("A valid Genai model name is required")

        # Extract Google API specific arguments
        args = {}
        if "api_key" in kwargs:
            args["api_key"] = kwargs["api_key"]
        if "api_version" in kwargs:
            args["api_version"] = types.HttpOptions(api_version=kwargs["api_version"])
        if "project" in kwargs:
            args["project"] = kwargs["project"]
        if "location" in kwargs:
            args["location"] = kwargs["location"]
        if "vertexai" in kwargs:
            args["vertexai"] = kwargs["vertexai"]

        self.instructions = ""
        self.client = genai.Client(**args)

    def generate_response(self) -> Dict[str, Any]:
        """
        Generate a non-streaming response from the Genai model.

        Returns:
            Dict[str, Any]: A dictionary containing the response content, tool calls, and candidates.
        """
        # Create a copy of kwargs to avoid modifying the original
        kwargs = self.kwargs.copy()

        # Remove Google API specific arguments
        for key in ["api_key", "api_version", "project", "location", "vertexai"]:
            if key in kwargs:
                kwargs.pop(key)

        # Configure tools if provided
        config = {}
        if kwargs.get("tools"):
            functions = self.convert_to_function_declarations(kwargs.get("tools"))
            tool = types.Tool(function_declarations=functions)
            config = types.GenerateContentConfig(
                system_instruction=self.instructions,
                tools=[tool],
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=True
                ),
            )

        # Generate content
        response = self.client.models.generate_content(
            model=self.name,
            contents=self._convert_to_contents(self.get_messages()),
            config=config,
        )

        return {
            "tool_calls": response.function_calls,
            "content": response.text if not response.function_calls else "",
            "candidates": response.candidates or [],
        }

    def generate_stream_response(self) -> Generator[Dict[str, Any], None, None]:
        """
        Generate a streaming response from the Genai model.

        Yields:
            Dict[str, Any]: Chunks of the response containing content and tool calls.
        """
        # Create a copy of kwargs to avoid modifying the original
        kwargs = self.kwargs.copy()

        # Remove Google API specific arguments
        for key in ["api_key", "api_version", "project", "location", "vertexai"]:
            if key in kwargs:
                kwargs.pop(key)

        # Configure tools if provided
        config = {}
        if kwargs.get("tools"):
            functions = self.convert_to_function_declarations(kwargs.get("tools"))
            tool = types.Tool(function_declarations=functions)
            config = types.GenerateContentConfig(
                system_instruction=self.instructions,
                tools=[tool],
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=True
                ),
            )

        # Generate streaming content
        response = self.client.models.generate_content_stream(
            model=self.name,
            contents=self._convert_to_contents(self.get_messages()),
            config=config,
        )

        # Process the stream
        result = {
            "tool_calls": [],
            "content": "",
        }
        for chunk in response:
            if chunk.function_calls:
                result["tool_calls"] = chunk.function_calls
            if chunk.text is not None:
                result["content"] = chunk.text
            yield result.copy()  # Return a copy to avoid reference issues

    @staticmethod
    def convert_to_function_declarations(
        json_input: List[Dict[str, Any]],
    ) -> List[types.FunctionDeclaration]:
        """
        Convert JSON tool definitions to Google's FunctionDeclaration format.

        Args:
            json_input (List[Dict[str, Any]]): A list of tool definitions.

        Returns:
            List[types.FunctionDeclaration]: A list of function declarations.

        Raises:
            ValueError: If the input is not a list or if required fields are missing.
        """
        if not isinstance(json_input, list):
            raise ValueError("Input should be a list of dictionaries")

        function_declarations = []

        for data in json_input:
            # Extract name, description, and parameters
            name = data.get("name")
            description = data.get("description")
            params = data.get("parameters", {})

            # Validate required fields
            if not name or not description:
                raise ValueError("Each function must have name and description")

            # Convert parameters to Schema format
            schema_properties = {}
            for prop_name, prop_details in params.get("properties", {}).items():
                schema_properties[prop_name] = types.Schema(
                    type=prop_details.get("type", "STRING").upper(),
                    description=prop_details.get("description", ""),
                )

            parameters = types.Schema(
                type=params.get("type", "OBJECT").upper(),
                properties=schema_properties,
                required=params.get("required", []),
            )

            # Create FunctionDeclaration and add to list
            function_declarations.append(
                types.FunctionDeclaration(
                    name=name, description=description, parameters=parameters
                )
            )

        return function_declarations

    @staticmethod
    def _convert_to_contents(messages: List[Dict[str, Any]]) -> List[types.Content]:
        """
        Convert a list of message dictionaries to Google's Content objects.

        Args:
            messages (List[Dict[str, Any]]): List of message dictionaries.

        Returns:
            List[types.Content]: List of Content objects.
        """
        if not messages:
            return []

        contents = []
        for message in messages:
            parts = []
            content = message.get("content", "")

            # Handle string content
            if isinstance(content, str):
                parts = [types.Part.from_text(text=content)]
            # Handle list of parts
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, str):
                        parts.append(types.Part.from_text(text=part))
                    elif isinstance(part, dict):
                        if "text" in part:
                            parts.append(types.Part.from_text(text=part["text"]))
                        elif "file_data" in part:
                            file_data = part["file_data"]
                            parts.append(
                                types.Part.from_uri(
                                    uri=file_data["file_uri"],
                                    mime_type=file_data["mime_type"],
                                )
                            )
                        elif "inline_data" in part:
                            inline_data = part["inline_data"]
                            parts.append(
                                types.Part.from_data(
                                    data=inline_data["data"],
                                    mime_type=inline_data["mime_type"],
                                )
                            )
                        elif "function_response" in part:
                            function_response = part["function_response"]
                            parts.append(
                                types.Part.from_function_response(
                                    name=function_response["name"],
                                    response=function_response["response"],
                                )
                            )
                        elif "function_call" in part:
                            function_call = part["function_call"]
                            parts.append(
                                types.Part.from_function_call(
                                    name=function_call["name"],
                                    args=function_call["args"],
                                )
                            )

            contents.append(types.Content(parts=parts, role=message["role"]))
        return contents

    def get_tool_format(self) -> Dict[str, Any]:
        """
        Get the format for tool definitions in Google's expected structure.

        Returns:
            Dict[str, Any]: The tool format template.
        """
        return {
            "name": "{name}",
            "description": "{description}",
            "parameters": {
                "type": "object",
                "properties": "{parameters}",
                "required": "{required}",
            },
        }

    @staticmethod
    def _get_tool_call_format() -> Dict[str, Any]:
        """
        Get the format for tool calls in Google's expected structure.

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
            tool_call (Any): The tool call object from Google.

        Returns:
            Dict[str, Any]: A dictionary containing the tool call's id, name, and arguments.
        """
        return {"id": tool_call.id, "name": tool_call.name, "arguments": tool_call.args}

    @staticmethod
    def _content_to_json(content: Any) -> List[Dict[str, Any]]:
        """
        Convert a Content object to a JSON-serializable dictionary.

        Args:
            content (Any): The Content object.

        Returns:
            List[Dict[str, Any]]: A list of message dictionaries.
        """
        parts_list = []
        for part in content.parts:
            part_dict = {}
            if part.function_call:
                function_call_dict = {
                    "name": part.function_call.name,
                    "args": part.function_call.args,
                }
                part_dict["function_call"] = function_call_dict
            if part_dict:
                parts_list.append(part_dict)

        contents = [{"role": content.role, "content": parts_list}]

        return contents

    def get_assistant_message(self, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Format an assistant message with tool calls for inclusion in the message history.

        Args:
            response (Dict[str, Any]): The response containing content and candidates.

        Returns:
            List[Dict[str, Any]]: A list of formatted assistant messages.
        """
        if not response.get("candidates"):
            return [{"role": "assistant", "content": response.get("content", "")}]

        return self._content_to_json(response["candidates"][0].content)

    def get_tool_message(self, tool_responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Format tool responses for inclusion in the message history.

        Args:
            tool_responses (List[Dict[str, Any]]): The responses from tools.

        Returns:
            Dict[str, Any]: A formatted tool message.
        """
        content = []
        for tool_response in tool_responses:
            content.append(
                {
                    "function_response": {
                        "name": tool_response["name"],
                        "response": {
                            "result": tool_response["tool_result"],
                        },
                    }
                }
            )

        return {"role": "tool", "content": content}

    def set_system_message(self, message: str) -> None:
        """
        Set the system message for the conversation.

        Args:
            message (str): The system message.
        """
        self.instructions = message

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
