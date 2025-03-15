import pytest
from unittest.mock import Mock, patch, MagicMock

from agents_manager.models.Genai import Genai
from agents_manager.Container import Container


class TestGenaiModel:
    """Test suite for the Genai model implementation."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.model_name = "gemini-2.0-flash"
        self.api_key = "test-api-key"
        with patch("agents_manager.models.Genai.genai"):
            self.model = Genai(name=self.model_name, api_key=self.api_key)

    def test_init(self):
        """Test initialization of Genai model."""
        with patch("agents_manager.models.Genai.genai") as mock_genai:
            model = Genai(name=self.model_name, api_key=self.api_key)

            assert model.name == self.model_name
            assert model.kwargs == {"api_key": self.api_key}
            assert model.instructions == ""

            # Verify that genai.Client was initialized with the correct parameters
            mock_genai.Client.assert_called_once_with(api_key=self.api_key)

    def test_init_with_none_name(self):
        """Test initialization with None name raises ValueError."""
        with (
            patch("agents_manager.models.Genai.genai"),
            pytest.raises(ValueError, match="A valid Genai model name is required"),
        ):
            Genai(name=None, api_key=self.api_key)

    def test_init_with_additional_args(self):
        """Test initialization with additional arguments."""
        with patch("agents_manager.models.Genai.genai") as mock_genai:
            with patch(
                "agents_manager.models.Genai.types.HttpOptions"
            ) as mock_http_options:
                mock_http_options.return_value = "mock_http_options"

                model = Genai(
                    name=self.model_name,
                    api_key=self.api_key,
                    api_version="v1beta",
                    project="test-project",
                    location="us-central1",
                    vertexai=True,
                )

                # Verify that genai.Client was initialized with all parameters
                mock_genai.Client.assert_called_once_with(
                    api_key=self.api_key,
                    api_version="mock_http_options",
                    project="test-project",
                    location="us-central1",
                    vertexai=True,
                )

    def test_generate_response(self):
        """Test generating a non-streaming response."""
        # Mock the Genai client and its response
        mock_response = Mock()
        mock_response.text = "Test response"
        mock_response.function_calls = None
        mock_response.candidates = []

        self.model.client = Mock()
        self.model.client.models.generate_content.return_value = mock_response

        # Set up test messages
        test_messages = [{"role": "user", "content": "Hello"}]
        self.model.set_messages(test_messages)

        # Call the method
        response = self.model.generate_response()

        # Verify the response
        assert response == {
            "tool_calls": None,
            "content": "Test response",
            "candidates": [],
        }

        # Verify the client was called correctly
        self.model.client.models.generate_content.assert_called_once()

    @pytest.mark.filterwarnings(
        "ignore:.*is not a valid DynamicRetrievalConfigMode.*:UserWarning"
    )
    def test_generate_response_with_tools(self):
        """Test generating a response with tool calls."""
        # Mock the Genai client and its response with tool calls
        mock_function_calls = Mock()

        mock_response = Mock()
        mock_response.text = ""
        mock_response.function_calls = mock_function_calls
        mock_response.candidates = []

        self.model.client = Mock()
        self.model.client.models.generate_content.return_value = mock_response

        # Set up test messages and tools
        test_messages = [{"role": "user", "content": "Use a tool"}]
        self.model.set_messages(test_messages)
        self.model.kwargs["tools"] = [
            {"name": "test_tool", "description": "A test tool"}
        ]

        # Mock the convert_to_function_declarations method
        with (
            patch(
                "agents_manager.models.Genai.Genai.convert_to_function_declarations"
            ) as mock_convert,
            patch("agents_manager.models.Genai.types.Tool") as mock_tool,
        ):
            # Return a proper dictionary structure instead of a string
            mock_convert.return_value = [
                {"name": "test_function", "description": "Test function"}
            ]

            # Mock the Tool constructor to avoid validation errors
            mock_tool_instance = Mock()
            mock_tool.return_value = mock_tool_instance

            # Call the method
            response = self.model.generate_response()

            # Verify the response
            assert response["tool_calls"] == mock_function_calls
            assert response["content"] == ""

            # Verify the client was called correctly with tools
            self.model.client.models.generate_content.assert_called_once()
            mock_convert.assert_called_once_with(self.model.kwargs["tools"])

            # Verify Tool was constructed with the correct parameters
            mock_tool.assert_called_once_with(
                function_declarations=mock_convert.return_value
            )

    def test_generate_stream_response(self):
        """Test generating a streaming response."""
        # Create mock chunks for the streaming response
        chunk1 = Mock()
        chunk1.text = "Hello"
        chunk1.function_calls = None

        chunk2 = Mock()
        chunk2.text = "Hello world"
        chunk2.function_calls = None

        # Set up the mock client
        self.model.client = Mock()
        self.model.client.models.generate_content_stream.return_value = [chunk1, chunk2]

        # Set up test messages
        test_messages = [{"role": "user", "content": "Hello"}]
        self.model.set_messages(test_messages)

        # Call the method and collect results
        results = list(self.model.generate_stream_response())

        # Verify the results
        assert len(results) == 2
        assert results[0]["content"] == "Hello"
        assert results[0]["tool_calls"] == []
        assert results[1]["content"] == "Hello world"
        assert results[1]["tool_calls"] == []

        # Verify the client was called correctly
        self.model.client.models.generate_content_stream.assert_called_once()

    def test_convert_to_function_declarations(self):
        """Test converting JSON tool definitions to function declarations."""
        json_input = [
            {
                "name": "test_function",
                "description": "A test function",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "param1": {
                            "type": "string",
                            "description": "A string parameter",
                        },
                        "param2": {
                            "type": "integer",
                            "description": "An integer parameter",
                        },
                    },
                    "required": ["param1"],
                },
            }
        ]

        with (
            patch("agents_manager.models.Genai.types.Schema") as mock_schema,
            patch(
                "agents_manager.models.Genai.types.FunctionDeclaration"
            ) as mock_func_decl,
        ):

            # Mock Schema instances
            mock_schema.side_effect = lambda **kwargs: kwargs

            # Mock FunctionDeclaration
            mock_func_decl.side_effect = lambda **kwargs: kwargs

            result = Genai.convert_to_function_declarations(json_input)

            # Verify the result
            assert len(result) == 1
            assert result[0]["name"] == "test_function"
            assert result[0]["description"] == "A test function"
            assert "parameters" in result[0]

    def test_convert_to_function_declarations_invalid_input(self):
        """Test converting invalid JSON input."""
        # Test with non-list input
        with pytest.raises(ValueError, match="Input should be a list of dictionaries"):
            Genai.convert_to_function_declarations({"not": "a list"})

        # Test with missing required fields
        with pytest.raises(
            ValueError, match="Each function must have name and description"
        ):
            Genai.convert_to_function_declarations([{"parameters": {}}])

    def test_convert_to_contents(self):
        """Test converting message dictionaries to Content objects."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]

        with (
            patch("agents_manager.models.Genai.types.Part") as mock_part,
            patch("agents_manager.models.Genai.types.Content") as mock_content,
        ):

            # Mock Part.from_text
            mock_part.from_text.side_effect = lambda text: f"text_part:{text}"

            # Mock Content
            mock_content.side_effect = lambda **kwargs: kwargs

            result = Genai._convert_to_contents(messages)

            # Verify the result
            assert len(result) == 2
            assert result[0]["role"] == "user"
            assert result[0]["parts"] == ["text_part:Hello"]
            assert result[1]["role"] == "assistant"
            assert result[1]["parts"] == ["text_part:Hi there"]

            # Verify Part.from_text was called correctly
            assert mock_part.from_text.call_count == 2

    def test_get_tool_format(self):
        """Test getting the tool format."""
        tool_format = self.model.get_tool_format()

        assert tool_format["name"] == "{name}"
        assert tool_format["description"] == "{description}"
        assert tool_format["parameters"]["type"] == "object"
        assert tool_format["parameters"]["properties"] == "{parameters}"
        assert tool_format["parameters"]["required"] == "{required}"

    def test_get_keys_in_tool_output(self):
        """Test extracting keys from a tool call."""
        # Create a mock tool call
        mock_tool_call = Mock()
        mock_tool_call.id = "call_123"
        mock_tool_call.name = "test_function"
        mock_tool_call.args = {"param": "value"}

        # Call the method
        result = self.model.get_keys_in_tool_output(mock_tool_call)

        # Verify the result
        assert result == {
            "id": "call_123",
            "name": "test_function",
            "arguments": {"param": "value"},
        }

    def test_content_to_json(self):
        """Test converting a Content object to JSON."""
        # Create a mock content with function calls
        mock_function_call = Mock()
        mock_function_call.name = "test_function"
        mock_function_call.args = {"param": "value"}

        mock_part = Mock()
        mock_part.function_call = mock_function_call

        mock_content = Mock()
        mock_content.parts = [mock_part]
        mock_content.role = "assistant"

        # Call the method
        result = Genai._content_to_json(mock_content)

        # Verify the result
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert len(result[0]["content"]) == 1
        assert result[0]["content"][0]["function_call"]["name"] == "test_function"
        assert result[0]["content"][0]["function_call"]["args"] == {"param": "value"}

    def test_get_assistant_message(self):
        """Test formatting an assistant message."""
        # Create a mock response with candidates
        mock_content = Mock()
        mock_candidate = Mock()
        mock_candidate.content = mock_content

        response = {"content": "Using a tool", "candidates": [mock_candidate]}

        # Mock the _content_to_json method
        with patch(
            "agents_manager.models.Genai.Genai._content_to_json"
        ) as mock_to_json:
            mock_to_json.return_value = [
                {"role": "assistant", "content": "Converted content"}
            ]

            # Call the method
            result = self.model.get_assistant_message(response)

            # Verify the result
            assert result == [{"role": "assistant", "content": "Converted content"}]
            mock_to_json.assert_called_once_with(mock_content)

    def test_get_assistant_message_no_candidates(self):
        """Test formatting an assistant message with no candidates."""
        response = {"content": "Simple response", "candidates": []}

        # Call the method
        result = self.model.get_assistant_message(response)

        # Verify the result
        assert result == [{"role": "assistant", "content": "Simple response"}]

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
        assert result["role"] == "tool"
        assert isinstance(result["content"], list)
        assert len(result["content"]) == 1
        assert "function_response" in result["content"][0]
        assert result["content"][0]["function_response"]["name"] == "test_function"
        assert (
            result["content"][0]["function_response"]["response"]["result"]
            == "Result from tool"
        )

    def test_set_system_message(self):
        """Test setting a system message."""
        # Call the method
        self.model.set_system_message("System instruction")

        # Verify the instruction was set correctly
        assert self.model.instructions == "System instruction"

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
        assert messages[1]["content"] == "Response"
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

        # Mock the conversion functions
        with (
            patch("agents_manager.models.Genai.function_to_json") as mock_func_to_json,
            patch(
                "agents_manager.models.Genai.container_to_json"
            ) as mock_container_to_json,
        ):

            mock_func_to_json.return_value = {"function": "json"}
            mock_container_to_json.return_value = {"container": "json"}

            # Call the method
            self.model.set_tools([test_function, mock_container])

            # Verify the conversion functions were called correctly
            mock_func_to_json.assert_called_once()
            mock_container_to_json.assert_called_once()

            # Verify the tools were added to kwargs
            assert "tools" in self.model.kwargs
            assert len(self.model.kwargs["tools"]) == 2
            assert self.model.kwargs["tools"][0] == {"function": "json"}
            assert self.model.kwargs["tools"][1] == {"container": "json"}
