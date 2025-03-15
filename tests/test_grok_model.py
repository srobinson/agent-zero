import pytest
from unittest.mock import Mock, patch, MagicMock

from agents_manager.models.Grok import Grok
from agents_manager.Container import Container


class TestGrokModel:
    """Test suite for the Grok model implementation."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.model_name = "grok-2-latest"
        self.api_key = "test-api-key"
        with patch("agents_manager.models.Grok.OpenAI"):
            self.model = Grok(name=self.model_name, api_key=self.api_key)

    def test_init(self):
        """Test initialization of Grok model."""
        with patch("agents_manager.models.Grok.OpenAI") as mock_openai:
            model = Grok(name=self.model_name, api_key=self.api_key)

            assert model.name == self.model_name
            assert model.kwargs == {"api_key": self.api_key}

            # Verify that OpenAI client was initialized with the correct base URL
            mock_openai.assert_called_once_with(
                api_key=self.api_key, base_url="https://api.x.ai/v1"
            )

    def test_init_with_custom_base_url(self):
        """Test initialization with a custom base URL."""
        custom_base_url = "https://custom.x.ai/v1"
        with patch("agents_manager.models.Grok.OpenAI") as mock_openai:
            model = Grok(
                name=self.model_name, api_key=self.api_key, base_url=custom_base_url
            )

            assert model.name == self.model_name
            assert model.kwargs == {
                "api_key": self.api_key,
                "base_url": custom_base_url,
            }

            # Verify that OpenAI client was initialized with the custom base URL
            mock_openai.assert_called_once_with(
                api_key=self.api_key, base_url=custom_base_url
            )

    def test_init_with_none_name(self):
        """Test initialization with None name raises ValueError."""
        with (
            patch("agents_manager.models.Grok.OpenAI"),
            pytest.raises(ValueError, match="A valid OpenAI model name is required"),
        ):
            Grok(name=None, api_key=self.api_key)

    def test_inheritance_from_openai(self):
        """Test that Grok inherits methods from OpenAi."""
        # Since Grok inherits from OpenAi, it should have all the same methods
        assert hasattr(self.model, "generate_response")
        assert hasattr(self.model, "generate_stream_response")
        assert hasattr(self.model, "get_tool_format")
        assert hasattr(self.model, "get_keys_in_tool_output")
        assert hasattr(self.model, "get_assistant_message")
        assert hasattr(self.model, "get_tool_message")
        assert hasattr(self.model, "set_system_message")
        assert hasattr(self.model, "set_user_message")
        assert hasattr(self.model, "set_tools")

    @patch("agents_manager.models.OpenAi.OpenAi.generate_response")
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

    @patch("agents_manager.models.OpenAi.OpenAi.generate_stream_response")
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

    @patch("agents_manager.models.OpenAi.OpenAi.set_tools")
    def test_set_tools_inheritance(self, mock_set_tools):
        """Test that set_tools is inherited from OpenAi."""

        # Define a test function
        def test_function(param: str) -> str:
            """Test function description."""
            return f"Result: {param}"

        # Create a test container
        mock_container = Mock(spec=Container)
        mock_container.name = "test_container"

        # Call the method
        self.model.set_tools([test_function, mock_container])

        # Verify that the OpenAi implementation was called with the correct arguments
        mock_set_tools.assert_called_once_with([test_function, mock_container])
