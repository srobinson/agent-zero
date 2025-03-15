import json
from unittest.mock import MagicMock, Mock, patch

import pytest

from agentflow.Container import Container
from models.OpenAi import OpenAi


class TestOpenAiModel:
    """Test suite for the OpenAi model implementation."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.model_name = "gpt-3.5-turbo"
        self.api_key = "test-api-key"
        # Use a context manager to patch OpenAI during initialization
        with patch("models.OpenAi.OpenAI"):
            self.model = OpenAi(name=self.model_name, api_key=self.api_key)

    def test_init(self):
        """Test initialization of OpenAi model."""
        # Patch the OpenAI class in the correct module
        with patch("models.OpenAi.OpenAI") as mock_openai:
            model = OpenAi(name=self.model_name, api_key=self.api_key)

            assert model.name == self.model_name
            assert model.kwargs == {"api_key": self.api_key}
            mock_openai.assert_called_once_with(api_key=self.api_key)

    def test_init_with_none_name(self):
        """Test initialization with None name raises ValueError."""
        with (
            patch("models.OpenAi.OpenAI"),
            pytest.raises(ValueError, match="A valid OpenAI model name is required"),
        ):
            OpenAi(name=None, api_key=self.api_key)

    def test_generate_response(self):
        """Test generating a non-streaming response."""
        # Mock the OpenAI client and its response
        mock_message = Mock()
        mock_message.content = "Test response"
        mock_message.tool_calls = []

        mock_choice = Mock()
        mock_choice.message = mock_message

        mock_response = Mock()
        mock_response.choices = [mock_choice]

        self.model.client = Mock()
        self.model.client.chat.completions.create.return_value = mock_response

        # Set up test messages
        test_messages = [{"role": "user", "content": "Hello"}]
        self.model.set_messages(test_messages)

        # Call the method
        response = self.model.generate_response()

        # Verify the response
        assert response == {"tool_calls": [], "content": "Test response"}

        # Verify the client was called correctly
        self.model.client.chat.completions.create.assert_called_once_with(
            model=self.model_name, messages=test_messages, stream=False
        )

    def test_generate_response_with_tools(self):
        """Test generating a response with tool calls."""
        # Mock the OpenAI client and its response with tool calls
        mock_tool_call = Mock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function.name = "test_function"
        mock_tool_call.function.arguments = '{"param": "value"}'

        mock_message = Mock()
        mock_message.content = "Using a tool"
        mock_message.tool_calls = [mock_tool_call]

        mock_choice = Mock()
        mock_choice.message = mock_message

        mock_response = Mock()
        mock_response.choices = [mock_choice]

        self.model.client = Mock()
        self.model.client.chat.completions.create.return_value = mock_response

        # Set up test messages
        test_messages = [{"role": "user", "content": "Use a tool"}]
        self.model.set_messages(test_messages)

        # Call the method
        response = self.model.generate_response()

        # Verify the response
        assert response["content"] == "Using a tool"
        assert len(response["tool_calls"]) == 1
        assert response["tool_calls"][0].id == "call_123"
        assert response["tool_calls"][0].function.name == "test_function"
        assert response["tool_calls"][0].function.arguments == '{"param": "value"}'

    def test_generate_stream_response(self):
        """Test generating a streaming response."""
        # Create mock chunks for the streaming response
        chunk1 = Mock()
        chunk1.choices = [Mock()]
        chunk1.choices[0].delta = Mock()
        chunk1.choices[0].delta.content = "Hello"
        chunk1.choices[0].delta.tool_calls = None

        chunk2 = Mock()
        chunk2.choices = [Mock()]
        chunk2.choices[0].delta = Mock()
        chunk2.choices[0].delta.content = " world"
        chunk2.choices[0].delta.tool_calls = None

        # Set up the mock client directly on the model instance
        self.model.client = Mock()
        self.model.client.chat.completions.create.return_value = [chunk1, chunk2]

        # Set up test messages
        test_messages = [{"role": "user", "content": "Hello"}]
        self.model.set_messages(test_messages)

        # Call the method and collect results
        results = list(self.model.generate_stream_response())

        # Verify the results
        assert len(results) == 2

        # The content in each result should be the accumulated content up to that point
        # First chunk should have just "Hello"
        assert results[0]["content"] == "Hello"

        # Second chunk should have "Hello world" (accumulated)
        assert results[1]["content"] == " world"

        # Verify the client was called correctly
        self.model.client.chat.completions.create.assert_called_once_with(
            model=self.model_name, messages=test_messages, stream=True
        )

    def test_get_tool_format(self):
        """Test getting the tool format."""
        tool_format = self.model.get_tool_format()

        assert tool_format["type"] == "function"
        assert tool_format["function"]["name"] == "{name}"
        assert tool_format["function"]["description"] == "{description}"
        assert tool_format["function"]["parameters"]["type"] == "object"
        assert tool_format["function"]["parameters"]["properties"] == "{parameters}"
        assert tool_format["function"]["parameters"]["required"] == "{required}"

    def test_get_keys_in_tool_output(self):
        """Test extracting keys from a tool call."""
        # Create a mock tool call
        mock_tool_call = Mock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function.name = "test_function"
        mock_tool_call.function.arguments = '{"param": "value"}'

        # Call the method
        result = self.model.get_keys_in_tool_output(mock_tool_call)

        # Verify the result
        assert result == {
            "id": "call_123",
            "name": "test_function",
            "arguments": '{"param": "value"}',
        }

    def test_get_assistant_message(self):
        """Test formatting an assistant message with tool calls."""
        # Create a mock response with tool calls
        mock_tool_call = Mock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function.name = "test_function"
        mock_tool_call.function.arguments = '{"param": "value"}'

        response = {"content": "Using a tool", "tool_calls": [mock_tool_call]}

        # Call the method
        result = self.model.get_assistant_message(response)

        # Verify the result
        assert result["role"] == "assistant"
        assert result["content"] == "Using a tool"
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["id"] == "call_123"
        assert result["tool_calls"][0]["type"] == "function"
        assert result["tool_calls"][0]["function"]["name"] == "test_function"
        assert result["tool_calls"][0]["function"]["arguments"] == '{"param": "value"}'

    def test_get_tool_message(self):
        """Test formatting tool responses."""
        # Create mock tool responses
        tool_responses = [
            {
                "id": "call_123",
                "tool_result": "Result from tool",
                "name": "test_function",
            }
        ]

        # Call the method
        result = self.model.get_tool_message(tool_responses)

        # Verify the result
        assert len(result) == 1
        assert result[0]["role"] == "tool"
        assert result[0]["content"] == "Result from tool"
        assert result[0]["tool_call_id"] == "call_123"

    def test_set_system_message(self):
        """Test setting a system message."""
        # Call the method
        self.model.set_system_message("System instruction")

        # Verify the messages were set correctly
        messages = self.model.get_messages()
        assert len(messages) == 1
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "System instruction"

    def test_set_user_message_string(self):
        """Test setting a user message as a string."""
        # Set up initial messages
        self.model.set_system_message("System instruction")

        # Call the method
        self.model.set_user_message("User message")

        # Verify the messages were set correctly
        messages = self.model.get_messages()
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "User message"

    def test_set_user_message_dict(self):
        """Test setting a user message as a dictionary."""
        # Set up initial messages
        self.model.set_system_message("System instruction")

        # Call the method
        self.model.set_user_message({"role": "user", "content": "User message"})

        # Verify the messages were set correctly
        messages = self.model.get_messages()
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "User message"

    def test_set_user_message_list(self):
        """Test setting user messages as a list."""
        # Set up initial messages
        self.model.set_system_message("System instruction")

        # Call the method
        self.model.set_user_message(
            [
                {"role": "user", "content": "First message"},
                {"role": "assistant", "content": "Response"},
                {"role": "user", "content": "Second message"},
            ]
        )

        # Verify the messages were set correctly
        messages = self.model.get_messages()
        assert len(messages) == 4
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "First message"
        assert messages[2]["role"] == "assistant"
        assert messages[3]["role"] == "user"
        assert messages[3]["content"] == "Second message"

    def test_set_tools(self):
        """Test setting tools for the model."""

        # Define a test function
        def test_function(param: str) -> str:
            """Test function description."""
            return f"Result: {param}"

        # Create a test container
        mock_container = Mock(spec=Container)
        mock_container.name = "test_container"
        mock_container.description = "Test container description"

        # Call the method
        with (
            patch("models.OpenAi.function_to_json") as mock_func_to_json,
            patch("models.OpenAi.container_to_json") as mock_container_to_json,
        ):

            mock_func_to_json.return_value = {"function": "json"}
            mock_container_to_json.return_value = {"container": "json"}

            self.model.set_tools([test_function, mock_container])

            # Verify the conversion functions were called correctly
            mock_func_to_json.assert_called_once()
            mock_container_to_json.assert_called_once()

            # Verify the tools were added to kwargs
            assert "tools" in self.model.kwargs
            assert len(self.model.kwargs["tools"]) == 2
            assert self.model.kwargs["tools"][0] == {"function": "json"}
            assert self.model.kwargs["tools"][1] == {"container": "json"}
