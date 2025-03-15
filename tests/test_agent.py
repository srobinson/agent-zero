import pytest
from unittest.mock import Mock, patch

from agents_manager.Agent import Agent
from agents_manager.Model import Model
from agents_manager.Container import Container


@pytest.fixture
def mock_model():
    """Fixture that provides a mock Model."""
    model = Mock(spec=Model)
    model.messages = None
    return model


@pytest.fixture
def agent(mock_model):
    """Fixture that provides a basic Agent instance."""
    return Agent(name="test_agent", instruction="Test instruction", model=mock_model)


def test_agent_init_with_empty_name(mock_model):
    """Test that initializing an agent with an empty name raises ValueError."""
    with pytest.raises(ValueError, match="Agent name cannot be empty"):
        Agent(name="", model=mock_model)


def test_agent_init_with_tool_choice(mock_model):
    """Test initializing an agent with a tool choice function."""

    def tool_choice(tools):
        return tools[0]

    with patch("agents_manager.utils.function_to_json") as mock_function_to_json:
        mock_function_to_json.return_value = {"name": "tool_choice"}
        agent = Agent(name="test_agent", model=mock_model, tool_choice=tool_choice)

        assert agent.tool_choice == tool_choice
        mock_model.set_kwargs.assert_called_once()


def test_set_instruction_updates_system_message(agent):
    """Test that setting instruction updates system message if it exists."""
    # Setup: Add a system message
    agent.model.messages = [{"role": "system", "content": "Old instruction"}]

    # Action: Set new instruction
    agent.set_instruction("New instruction")

    # Assert: System message should be updated
    agent.model.set_system_message.assert_called_once_with("New instruction")
    assert agent.instruction == "New instruction"


def test_get_messages_raises_when_not_set(agent):
    """Test that get_messages raises ValueError when messages are not set."""
    agent.model.get_messages.return_value = None

    with pytest.raises(ValueError, match="Messages not set in the model"):
        agent.get_messages()


def test_add_message(agent):
    """Test adding a message to the agent."""
    # Setup
    agent.model.get_messages.return_value = [
        {"role": "system", "content": "Test instruction"}
    ]

    # Action
    agent.add_message("user", "Hello")

    # Assert
    expected_messages = [
        {"role": "system", "content": "Test instruction"},
        {"role": "user", "content": "Hello"},
    ]
    agent.model.set_messages.assert_called_once_with(expected_messages)


def test_add_tool(agent):
    """Test adding a tool to the agent."""

    # Setup
    def test_tool():
        return "Tool result"

    # Action
    agent.add_tool(test_tool)

    # Assert
    assert test_tool in agent.tools
    agent.model.set_tools.assert_called_once_with([test_tool])


def test_add_tool_duplicate(agent):
    """Test adding a duplicate tool doesn't add it twice."""

    # Setup
    def test_tool():
        return "Tool result"

    agent.tools = [test_tool]

    # Action
    agent.add_tool(test_tool)

    # Assert
    assert len(agent.tools) == 1
    assert test_tool in agent.tools


def test_remove_tool_callable(agent):
    """Test removing a callable tool by name."""

    # Setup
    def test_tool():
        return "Tool result"

    test_tool.__name__ = "test_tool"
    agent.tools = [test_tool]

    # Action
    result = agent.remove_tool("test_tool")

    # Assert
    assert result is True
    assert len(agent.tools) == 0
    agent.model.set_tools.assert_called_once_with([])


def test_remove_tool_container(agent):
    """Test removing a Container tool by name."""
    # Setup
    mock_container = Mock(spec=Container)
    mock_container.name = "container_tool"
    agent.tools = [mock_container]

    # Action
    result = agent.remove_tool("container_tool")

    # Assert
    assert result is True
    assert len(agent.tools) == 0
    agent.model.set_tools.assert_called_once_with([])


def test_remove_tool_not_found(agent):
    """Test removing a tool that doesn't exist."""

    # Setup
    def test_tool():
        return "Tool result"

    test_tool.__name__ = "test_tool"
    agent.tools = [test_tool]

    # Action
    result = agent.remove_tool("nonexistent_tool")

    # Assert
    assert result is False
    assert len(agent.tools) == 1
    agent.model.set_tools.assert_not_called()


def test_clear_messages(agent):
    """Test clearing all messages."""
    # Setup
    agent.instruction = "Test instruction"

    # Action
    agent.clear_messages()

    # Assert
    agent.model.set_messages.assert_called_once_with([])
    agent.model.set_system_message.assert_called_once_with("Test instruction")


def test_clear_messages_no_instruction(agent):
    """Test clearing messages with no instruction set."""
    # Setup
    agent.instruction = ""

    # Action
    agent.clear_messages()

    # Assert
    agent.model.set_messages.assert_called_once_with([])
    agent.model.set_system_message.assert_not_called()


def test_to_dict(agent):
    """Test converting agent to dictionary."""
    # Setup
    agent.name = "test_agent"
    agent.instruction = "Test instruction"
    agent.tools = [lambda: "Tool result"]

    # Mock the model's class name
    type(agent.model).__name__ = "MockModel"

    # Action
    result = agent.to_dict()

    # Assert
    assert result == {
        "name": "test_agent",
        "instruction": "Test instruction",
        "tools_count": 1,
        "model_type": "MockModel",
    }


def test_str_representation(agent):
    """Test string representation of agent."""
    # Setup
    agent.name = "test_agent"
    agent.tools = [lambda: "Tool result"]

    # Mock the model's class name
    type(agent.model).__name__ = "MockModel"

    # Action
    result = str(agent)

    # Assert
    assert result == "Agent(name='test_agent', model=MockModel, tools=1)"


def test_repr_representation(agent):
    """Test detailed string representation of agent."""
    # Setup
    agent.name = "test_agent"
    agent.instruction = "This is a very long instruction that should be truncated"
    agent.tools = [lambda: "Tool result"]

    # Mock the model's class name
    type(agent.model).__name__ = "MockModel"

    # Action
    result = repr(agent)

    # Assert
    # Just check that the key parts are present without being too specific about formatting
    assert "Agent(name='test_agent'" in result
    assert "instruction='" in result
    assert "model=MockModel" in result
    assert "tools=1" in result


def test_get_stream_response(agent):
    """Test streaming response generation."""
    # Setup
    agent.model.messages = [{"role": "user", "content": "Hello"}]
    agent.model.generate_stream_response.return_value = iter(
        [{"content": "Hello"}, {"content": " world"}]
    )

    # Action
    result = list(agent.get_stream_response())

    # Assert
    assert result == [{"content": "Hello"}, {"content": " world"}]
    agent.model.generate_stream_response.assert_called_once()


def test_get_stream_response_no_messages(agent):
    """Test streaming response with no messages raises ValueError."""
    # Setup
    agent.model.messages = None

    # Action & Assert
    with pytest.raises(
        ValueError, match="Messages must be set before generating a response"
    ):
        list(agent.get_stream_response())
