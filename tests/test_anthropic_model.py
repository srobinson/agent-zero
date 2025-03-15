from unittest.mock import MagicMock, Mock, patch

import pytest

from agentflow.Container import Container
from models.Anthropic import Anthropic


class TestAnthropicModel:
    """Test suite for the Anthropic model implementation."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.model_name = "claude-3-5-sonnet-20241022"
        self.api_key = "test-api-key"
        with patch("models.Anthropic.Ap"):
            self.model = Anthropic(
                name=self.model_name, api_key=self.api_key, max_tokens=1024
            )

    def test_init(self):
        """Test initialization of Anthropic model."""
        with patch("models.Anthropic.Ap") as mock_anthropic:
            model = Anthropic(
                name=self.model_name, api_key=self.api_key, max_tokens=1024
            )

            assert model.name == self.model_name
            # Only check for api_key in kwargs, as that's what the current implementation stores
            assert model.kwargs.get("api_key") == self.api_key

    def test_init_with_none_name(self):
        """Test initialization with None name raises ValueError."""
        with (
            patch("models.Anthropic.Ap"),
            pytest.raises(ValueError, match="A valid Anthropic model name is required"),
        ):
            Anthropic(name=None, api_key=self.api_key)

    def test_generate_response(self):
        """Test generating a non-streaming response."""
        # Mock the Anthropic client and its response
        mock_text_content = Mock()
        mock_text_content.type = "text"
        mock_text_content.text = "Test response"

        mock_response = Mock()
        mock_response.content = [mock_text_content]

        self.model.client = Mock()
        self.model.client.messages.create.return_value = mock_response

        # Set up test messages
        test_messages = [{"role": "user", "content": "Hello"}]
        self.model.set_messages(test_messages)

        # Call the method
        response = self.model.generate_response()

        # Verify the response
        assert response == {"tool_calls": [], "content": "Test response"}

        # Verify the client was called correctly - without max_tokens
        self.model.client.messages.create.assert_called_once_with(
            model=self.model_name,
            system=self.model.instruction,
            messages=test_messages,
            stream=False,
            max_tokens=1024,
        )

    def test_generate_response_with_tools(self):
        """Test generating a response with tool calls."""
        # Mock the Anthropic client and its response with tool calls
        mock_text_content = Mock()
        mock_text_content.type = "text"
        mock_text_content.text = "Using a tool"

        mock_tool_call = Mock()
        mock_tool_call.type = "tool_use"
        mock_tool_call.id = "call_123"
        mock_tool_call.name = "test_function"
        mock_tool_call.input = '{"param": "value"}'

        mock_response = Mock()
        mock_response.content = [mock_text_content, mock_tool_call]

        self.model.client = Mock()
        self.model.client.messages.create.return_value = mock_response

        # Set up test messages
        test_messages = [{"role": "user", "content": "Use a tool"}]
        self.model.set_messages(test_messages)

        # Call the method
        response = self.model.generate_response()

        # Verify the response
        assert response["content"] == "Using a tool"
        assert len(response["tool_calls"]) == 1
        assert response["tool_calls"][0].id == "call_123"
        assert response["tool_calls"][0].name == "test_function"
        assert response["tool_calls"][0].input == '{"param": "value"}'

    def test_generate_stream_response(self):
        """Test generating a streaming response."""
        # Create mock events for the streaming response
        text_event = Mock()
        text_event.type = "content_block_delta"
        text_event.delta = Mock()
        text_event.delta.type = "text_delta"
        text_event.delta.text = "Hello world"

        tool_start_event = Mock()
        tool_start_event.type = "content_block_start"
        tool_start_event.content_block = Mock()
        tool_start_event.content_block.type = "tool_use"
        tool_start_event.content_block.id = "call_123"
        tool_start_event.content_block.name = "test_function"

        tool_input_event = Mock()
        tool_input_event.type = "content_block_delta"
        tool_input_event.delta = Mock()
        tool_input_event.delta.type = "input_json_delta"
        tool_input_event.delta.partial_json = '{"param": "value"}'

        # Mock the stream context manager
        mock_stream = MagicMock()
        mock_stream.__enter__.return_value = [
            text_event,
            tool_start_event,
            tool_input_event,
        ]

        # Set up the mock client
        self.model.client = Mock()
        self.model.client.messages.stream.return_value = mock_stream

        # Set up test messages
        test_messages = [{"role": "user", "content": "Hello"}]
        self.model.set_messages(test_messages)

        # Call the method and collect results
        results = list(self.model.generate_stream_response())

        # Verify the results
        assert len(results) == 3

        # First event is text content
        assert results[0]["content"] == "Hello world"
        assert results[0]["tool_calls"] is None

        # Second event is tool start
        assert results[1]["content"] is None
        assert results[1]["tool_calls"]["id"] == "call_123"
        assert results[1]["tool_calls"]["name"] == "test_function"

        # Third event is tool input
        assert results[2]["content"] is None
        assert results[2]["tool_calls"]["id"] == "call_123"
        assert results[2]["tool_calls"]["name"] == "test_function"
        assert results[2]["tool_calls"]["input"] == '{"param": "value"}'

        # Verify the client was called correctly
        self.model.client.messages.stream.assert_called_once()

    def test_extract_content(self):
        """Test extracting content from a response."""
        # Create mock content items
        mock_text = Mock()
        mock_text.type = "text"
        mock_text.text = "Hello"

        mock_tool = Mock()
        mock_tool.type = "tool_use"
        mock_tool.id = "call_123"

        # Create a mock response
        mock_response = Mock()
        mock_response.content = [mock_text, mock_tool]

        # Test extracting text content
        text_items = Anthropic.extract_content(mock_response, "text")
        assert len(text_items) == 1
        assert text_items[0].type == "text"
        assert text_items[0].text == "Hello"

        # Test extracting tool content
        tool_items = Anthropic.extract_content(mock_response, "tool_use")
        assert len(tool_items) == 1
        assert tool_items[0].type == "tool_use"
        assert tool_items[0].id == "call_123"

    def test_get_tool_format(self):
        """Test getting the tool format."""
        tool_format = self.model.get_tool_format()

        assert tool_format["name"] == "{name}"
        assert tool_format["description"] == "{description}"
        assert tool_format["input_schema"]["type"] == "object"
        assert tool_format["input_schema"]["properties"] == "{parameters}"
        assert tool_format["input_schema"]["required"] == "{required}"

    def test_get_keys_in_tool_output(self):
        """Test extracting keys from a tool call."""
        # Create a mock tool call
        mock_tool_call = Mock()
        mock_tool_call.id = "call_123"
        mock_tool_call.name = "test_function"
        mock_tool_call.input = '{"param": "value"}'

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
        mock_tool_call.name = "test_function"
        mock_tool_call.input = '{"param": "value"}'

        response = {"content": "Using a tool", "tool_calls": [mock_tool_call]}

        # Call the method
        result = self.model.get_assistant_message(response)

        # Verify the result
        assert result["role"] == "assistant"
        assert isinstance(result["content"], list)
        assert len(result["content"]) == 1
        assert result["content"][0]["type"] == "tool_use"
        assert result["content"][0]["id"] == "call_123"
        assert result["content"][0]["name"] == "test_function"
        assert result["content"][0]["input"] == '{"param": "value"}'

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
        assert result["role"] == "user"
        assert isinstance(result["content"], list)
        assert len(result["content"]) == 1
        assert result["content"][0]["type"] == "tool_result"
        assert result["content"][0]["tool_use_id"] == "call_123"
        assert result["content"][0]["content"] == "Result from tool"

    def test_set_system_message(self):
        """Test setting a system message."""
        # Call the method
        self.model.set_system_message("System instruction")

        # Verify the instruction was set correctly
        assert self.model.instruction == "System instruction"

    def test_set_user_message_string(self):
        """Test setting a user message as a string."""
        # Call the method
        self.model.set_user_message("User message")

        # Verify the messages were set correctly
        messages = self.model.get_messages()
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "User message"

    def test_set_user_message_dict(self):
        """Test setting a user message as a dictionary."""
        # Call the method
        self.model.set_user_message({"role": "user", "content": "User message"})

        # Verify the messages were set correctly
        messages = self.model.get_messages()
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "User message"

    def test_set_user_message_list(self):
        """Test setting user messages as a list."""
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
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "First message"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"
        assert messages[2]["content"] == "Second message"

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
            patch("models.Anthropic.function_to_json") as mock_func_to_json,
            patch("models.Anthropic.container_to_json") as mock_container_to_json,
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
