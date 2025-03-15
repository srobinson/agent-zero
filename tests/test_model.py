import json
from typing import Any, Callable, Dict, Generator, List, Union
from unittest.mock import Mock

import pytest

from agentflow.Model import Model


class MockModel(Model):
    """A concrete implementation of Model for testing purposes."""

    def generate_response(self) -> Dict[str, Any]:
        """Mock implementation of generate_response."""
        if self.messages is None or self.messages == "":
            raise ValueError("Messages must be set before generating a response")
        return {"content": "Mock response", "tool_calls": []}

    def generate_stream_response(self) -> Generator[Dict[str, Any], None, None]:
        """Mock implementation of generate_stream_response."""
        if self.messages is None or self.messages == "":
            raise ValueError("Messages must be set before generating a response")
        yield {"content": "Mock stream response"}

    def get_tool_format(self) -> Dict[str, Any]:
        """Mock implementation of get_tool_format."""
        return {"type": "function", "function": {"name": "", "parameters": {}}}

    def get_keys_in_tool_output(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """Mock implementation of get_keys_in_tool_output."""
        return {
            "id": tool_call.get("id", ""),
            "name": tool_call.get("function", {}).get("name", ""),
            "arguments": tool_call.get("function", {}).get("arguments", "{}"),
        }

    def get_assistant_message(self, response: Any) -> Dict[str, Any]:
        """Mock implementation of get_assistant_message."""
        return {"role": "assistant", "content": response.get("content", "")}

    def get_tool_message(self, tool_responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Mock implementation of get_tool_message."""
        return {"role": "tool", "content": json.dumps(tool_responses)}

    def set_system_message(self, message: str) -> None:
        """Mock implementation of set_system_message."""
        messages = self.get_messages() or []
        # Remove existing system message if any
        messages = [m for m in messages if m.get("role") != "system"]
        # Add new system message
        messages.insert(0, {"role": "system", "content": message})
        self.set_messages(messages)

    def set_user_message(self, message: str) -> None:
        """Mock implementation of set_user_message."""
        messages = self.get_messages() or []
        messages.append({"role": "user", "content": message})
        self.set_messages(messages)

    def set_tools(self, tools: List[Callable]) -> None:
        """Mock implementation of set_tools."""
        self.kwargs["tools"] = tools


@pytest.fixture
def mock_model():
    """Fixture that provides a MockModel instance."""
    return MockModel(name="test_model")


class TestModel:
    """Test suite for the Model class."""

    def test_init(self, mock_model):
        """Test initialization of Model."""
        assert mock_model.name == "test_model"
        assert mock_model.messages is None
        assert isinstance(mock_model.kwargs, dict)

    def test_set_messages_valid(self, mock_model):
        """Test setting valid messages."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ]
        mock_model.set_messages(messages)
        assert mock_model.messages == json.dumps(messages)

    def test_set_messages_invalid_type(self, mock_model):
        """Test setting messages with invalid type."""
        with pytest.raises(TypeError, match="Messages must be a list of dictionaries"):
            mock_model.set_messages("not a list")

    def test_set_messages_invalid_format(self, mock_model):
        """Test setting messages with invalid format."""
        with pytest.raises(
            ValueError,
            match="Each message must be a dictionary with 'role' and 'content' keys",
        ):
            mock_model.set_messages([{"invalid": "message"}])

    def test_get_messages_none(self, mock_model):
        """Test getting messages when none are set."""
        assert mock_model.get_messages() is None

    def test_get_messages_empty_string(self, mock_model):
        """Test getting messages when set to empty string."""
        mock_model.messages = ""
        assert mock_model.get_messages() is None

    def test_get_messages_valid(self, mock_model):
        """Test getting valid messages."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ]
        mock_model.set_messages(messages)
        assert mock_model.get_messages() == messages

    def test_clear_messages(self, mock_model):
        """Test clearing messages."""
        messages = [{"role": "user", "content": "Hello!"}]
        mock_model.set_messages(messages)
        mock_model.clear_messages()
        assert mock_model.messages is None

    def test_set_kwargs_valid(self, mock_model):
        """Test setting valid kwargs."""
        mock_model.set_kwargs({"temperature": 0.7})
        assert mock_model.kwargs["temperature"] == 0.7

    def test_set_kwargs_invalid_type(self, mock_model):
        """Test setting kwargs with invalid type."""
        with pytest.raises(TypeError, match="kwargs must be a dictionary"):
            mock_model.set_kwargs("not a dict")

    def test_get_kwargs(self, mock_model):
        """Test getting kwargs."""
        mock_model.kwargs = {"temperature": 0.7}
        assert mock_model.get_kwargs() == {"temperature": 0.7}

    def test_to_dict(self, mock_model):
        """Test converting model to dictionary."""
        mock_model.set_messages([{"role": "user", "content": "Hello!"}])
        mock_model.set_kwargs({"temperature": 0.7})
        result = mock_model.to_dict()
        assert result["name"] == "test_model"
        assert result["type"] == "MockModel"
        assert result["has_messages"] is True
        assert result["kwargs_count"] == 1

    def test_str_representation(self, mock_model):
        """Test string representation of model."""
        assert str(mock_model) == "Model(name='test_model', type=MockModel)"

    def test_repr_representation(self, mock_model):
        """Test detailed string representation of model."""
        mock_model.set_kwargs({"temperature": 0.7})
        assert repr(mock_model) == "Model(name='test_model', type=MockModel, kwargs=1)"


class TestMockModelImplementation:
    """Test suite for the MockModel implementation."""

    def test_generate_response(self, mock_model):
        """Test generate_response method."""
        mock_model.set_messages([{"role": "user", "content": "Hello!"}])
        response = mock_model.generate_response()
        assert response["content"] == "Mock response"
        assert response["tool_calls"] == []

    def test_generate_response_no_messages(self, mock_model):
        """Test generate_response with no messages."""
        with pytest.raises(
            ValueError, match="Messages must be set before generating a response"
        ):
            mock_model.generate_response()

    def test_generate_stream_response(self, mock_model):
        """Test generate_stream_response method."""
        mock_model.set_messages([{"role": "user", "content": "Hello!"}])
        response = list(mock_model.generate_stream_response())
        assert len(response) == 1
        assert response[0]["content"] == "Mock stream response"

    def test_generate_stream_response_no_messages(self, mock_model):
        """Test generate_stream_response with no messages."""
        with pytest.raises(
            ValueError, match="Messages must be set before generating a response"
        ):
            list(mock_model.generate_stream_response())

    def test_get_tool_format(self, mock_model):
        """Test get_tool_format method."""
        format = mock_model.get_tool_format()
        assert format["type"] == "function"
        assert "function" in format

    def test_get_keys_in_tool_output(self, mock_model):
        """Test get_keys_in_tool_output method."""
        tool_call = {
            "id": "call_123",
            "function": {"name": "test_function", "arguments": '{"param": "value"}'},
        }
        keys = mock_model.get_keys_in_tool_output(tool_call)
        assert keys["id"] == "call_123"
        assert keys["name"] == "test_function"
        assert keys["arguments"] == '{"param": "value"}'

    def test_get_assistant_message(self, mock_model):
        """Test get_assistant_message method."""
        response = {"content": "Hello, I'm an assistant"}
        message = mock_model.get_assistant_message(response)
        assert message["role"] == "assistant"
        assert message["content"] == "Hello, I'm an assistant"

    def test_get_tool_message(self, mock_model):
        """Test get_tool_message method."""
        tool_responses = [{"id": "call_123", "tool_result": "result"}]
        message = mock_model.get_tool_message(tool_responses)
        assert message["role"] == "tool"
        assert json.loads(message["content"]) == tool_responses

    def test_set_system_message(self, mock_model):
        """Test set_system_message method."""
        mock_model.set_messages([{"role": "user", "content": "Hello!"}])
        mock_model.set_system_message("You are a helpful assistant.")
        messages = mock_model.get_messages()
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a helpful assistant."

    def test_set_system_message_replace_existing(self, mock_model):
        """Test set_system_message replaces existing system message."""
        mock_model.set_messages(
            [
                {"role": "system", "content": "Old instruction"},
                {"role": "user", "content": "Hello!"},
            ]
        )
        mock_model.set_system_message("New instruction")
        messages = mock_model.get_messages()
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "New instruction"

    def test_set_user_message(self, mock_model):
        """Test set_user_message method."""
        mock_model.set_messages(
            [{"role": "system", "content": "You are a helpful assistant."}]
        )
        mock_model.set_user_message("Hello!")
        messages = mock_model.get_messages()
        assert len(messages) == 2
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Hello!"

    def test_set_tools(self, mock_model):
        """Test set_tools method."""

        def tool1():
            pass

        def tool2():
            pass

        tools = [tool1, tool2]
        mock_model.set_tools(tools)
        assert mock_model.kwargs["tools"] == tools
