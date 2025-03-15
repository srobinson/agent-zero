from unittest.mock import MagicMock, Mock, patch

import pytest

from models.DeepSeek import DeepSeek


class TestDeepSeekModel:
    """Test suite for the DeepSeek model implementation."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.model_name = "deepseek-chat"
        self.api_key = "test-api-key"
        # Patch the OpenAI import within the DeepSeek module
        with patch("models.DeepSeek.OpenAI"):
            self.model = DeepSeek(name=self.model_name, api_key=self.api_key)

    def test_init(self):
        """Test initialization of DeepSeek model."""
        # Patch the OpenAI import within the DeepSeek module
        with patch("models.DeepSeek.OpenAI") as mock_openai:
            model = DeepSeek(name=self.model_name, api_key=self.api_key)

            assert model.name == self.model_name
            assert model.kwargs == {"api_key": self.api_key}

            # Verify that OpenAI client was initialized with the correct base URL
            mock_openai.assert_called_once_with(
                api_key=self.api_key, base_url="https://api.deepseek.com"
            )

    def test_init_with_none_name(self):
        """Test initialization with None name raises ValueError."""
        # The error message comes from OpenAi, not DeepSeek
        with (
            patch("models.DeepSeek.OpenAI"),
            pytest.raises(ValueError, match="A valid OpenAI model name is required"),
        ):
            DeepSeek(name=None, api_key=self.api_key)

    def test_inheritance_from_openai(self):
        """Test that DeepSeek inherits methods from OpenAi."""
        # Since DeepSeek inherits from OpenAi, it should have all the same methods
        assert hasattr(self.model, "generate_response")
        assert hasattr(self.model, "generate_stream_response")
        assert hasattr(self.model, "get_tool_format")
        assert hasattr(self.model, "get_keys_in_tool_output")
        assert hasattr(self.model, "get_assistant_message")
        assert hasattr(self.model, "get_tool_message")
        assert hasattr(self.model, "set_system_message")
        assert hasattr(self.model, "set_user_message")
        assert hasattr(self.model, "set_tools")

    def test_client_configuration(self):
        """Test that the client is configured correctly."""
        # Create a new mock for the OpenAI class
        with patch("models.DeepSeek.OpenAI") as mock_openai_class:
            # Create a mock instance that will be returned by the OpenAI constructor
            mock_client = Mock()
            mock_openai_class.return_value = mock_client

            # Initialize the model
            model = DeepSeek(name=self.model_name, api_key=self.api_key)

            # Verify that the client was initialized with the correct parameters
            mock_openai_class.assert_called_once_with(
                api_key=self.api_key, base_url="https://api.deepseek.com"
            )

            # Verify that the client was assigned to the model
            assert model.client == mock_client

    @patch("models.OpenAi.OpenAi.generate_response")
    def test_generate_response_inheritance(self, mock_generate_response):
        """Test that generate_response is inherited from OpenAi."""
        # Set up the mock to return a specific value
        mock_generate_response.return_value = {
            "content": "Test response",
            "tool_calls": [],
        }

        # Call the method
        response = self.model.generate_response()

        # Verify that the OpenAi implementation was called
        mock_generate_response.assert_called_once()

        # Verify the response
        assert response == {"content": "Test response", "tool_calls": []}

    @patch("models.OpenAi.OpenAi.generate_stream_response")
    def test_generate_stream_response_inheritance(self, mock_generate_stream_response):
        """Test that generate_stream_response is inherited from OpenAi."""
        # Set up the mock to return a specific generator
        mock_generate_stream_response.return_value = iter(
            [
                {"content": "Hello", "tool_calls": []},
                {"content": "Hello world", "tool_calls": []},
            ]
        )

        # Call the method and collect results
        results = list(self.model.generate_stream_response())

        # Verify that the OpenAi implementation was called
        mock_generate_stream_response.assert_called_once()

        # Verify the results
        assert len(results) == 2
        assert results[0]["content"] == "Hello"
        assert results[1]["content"] == "Hello world"
