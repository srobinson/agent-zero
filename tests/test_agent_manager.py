import unittest
from unittest.mock import MagicMock, Mock, patch

from agentflow.Agent import Agent
from agentflow.AgentManager import AgentManager
from agentflow.Container import Container


class TestAgentManager(unittest.TestCase):
    """Test suite for the AgentManager class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.agent_manager = AgentManager()

        # Create mock agents
        self.mock_agent = Mock(spec=Agent)
        self.mock_agent.name = "test_agent"
        self.mock_agent.instruction = "Test instruction"
        self.mock_agent.tools = []

        self.mock_agent2 = Mock(spec=Agent)
        self.mock_agent2.name = "test_agent2"
        self.mock_agent2.instruction = "Test instruction 2"
        self.mock_agent2.tools = []

    def test_init(self):
        """Test initialization of AgentManager."""
        self.assertEqual(self.agent_manager.agents, [])

    def test_add_agent(self):
        """Test adding an agent to the manager."""
        self.agent_manager.add_agent(self.mock_agent)
        self.assertEqual(len(self.agent_manager.agents), 1)
        self.assertEqual(self.agent_manager.agents[0], self.mock_agent)

    def test_add_agent_duplicate(self):
        """Test adding a duplicate agent doesn't add it twice."""
        self.agent_manager.add_agent(self.mock_agent)
        self.agent_manager.add_agent(self.mock_agent)
        self.assertEqual(len(self.agent_manager.agents), 1)

    def test_add_agent_invalid_type(self):
        """Test adding an invalid agent type raises ValueError."""
        with self.assertRaises(ValueError):
            self.agent_manager.add_agent("not an agent")

    def test_get_agent(self):
        """Test retrieving an agent by name."""
        self.agent_manager.add_agent(self.mock_agent)
        index, agent = self.agent_manager.get_agent("test_agent")
        self.assertEqual(index, 0)
        self.assertEqual(agent, self.mock_agent)

    def test_get_agent_not_found(self):
        """Test retrieving a non-existent agent returns None."""
        index, agent = self.agent_manager.get_agent("nonexistent_agent")
        self.assertIsNone(index)
        self.assertIsNone(agent)

    def test_initialize_user_input(self):
        """Test initializing an agent with user input."""
        self.agent_manager.add_agent(self.mock_agent)
        index, agent = self.agent_manager.initialize_user_input(
            "test_agent", "user message"
        )

        self.assertEqual(index, 0)
        self.assertEqual(agent, self.mock_agent)
        self.mock_agent.set_system_message.assert_called_once_with(
            self.mock_agent.instruction
        )
        self.mock_agent.set_tools.assert_called_once_with(self.mock_agent.tools)
        self.mock_agent.set_user_message.assert_called_once_with("user message")

    def test_initialize_user_input_no_user_input(self):
        """Test initializing an agent without user input."""
        self.agent_manager.add_agent(self.mock_agent)
        index, agent = self.agent_manager.initialize_user_input("test_agent")

        self.assertEqual(index, 0)
        self.assertEqual(agent, self.mock_agent)
        self.mock_agent.set_system_message.assert_called_once_with(
            self.mock_agent.instruction
        )
        self.mock_agent.set_tools.assert_called_once_with(self.mock_agent.tools)
        self.mock_agent.set_user_message.assert_not_called()

    def test_initialize_user_input_agent_not_found(self):
        """Test initializing a non-existent agent raises ValueError."""
        with self.assertRaises(ValueError):
            self.agent_manager.initialize_user_input("nonexistent_agent")

    def test_prepare_final_messages_dict(self):
        """Test preparing final messages with dict tool response."""
        mock_agent = Mock(spec=Agent)
        mock_model = Mock()
        mock_agent.get_model.return_value = mock_model

        current_messages = [{"role": "user", "content": "Hello"}]
        tool_responses = [{"id": "1", "tool_result": "result", "name": "tool"}]

        # Mock tool message as dict
        mock_model.get_tool_message.return_value = {
            "role": "tool",
            "content": "Tool result",
        }

        AgentManager._prepare_final_messages(
            mock_agent, current_messages, tool_responses
        )

        # Check that the dict was appended
        self.assertEqual(len(current_messages), 2)
        self.assertEqual(
            current_messages[1], {"role": "tool", "content": "Tool result"}
        )
        mock_agent.set_messages.assert_called_once_with(current_messages)

    def test_prepare_final_messages_list(self):
        """Test preparing final messages with list tool response."""
        mock_agent = Mock(spec=Agent)
        mock_model = Mock()
        mock_agent.get_model.return_value = mock_model

        current_messages = [{"role": "user", "content": "Hello"}]
        tool_responses = [{"id": "1", "tool_result": "result", "name": "tool"}]

        # Mock tool message as list
        mock_model.get_tool_message.return_value = [
            {"role": "tool", "content": "Tool result 1"},
            {"role": "tool", "content": "Tool result 2"},
        ]

        AgentManager._prepare_final_messages(
            mock_agent, current_messages, tool_responses
        )

        # Check that the list was extended
        self.assertEqual(len(current_messages), 3)
        self.assertEqual(
            current_messages[1], {"role": "tool", "content": "Tool result 1"}
        )
        self.assertEqual(
            current_messages[2], {"role": "tool", "content": "Tool result 2"}
        )
        mock_agent.set_messages.assert_called_once_with(current_messages)

    @patch("agentflow.AgentManager.AgentManager.initialize_user_input")
    def test_run_agent_no_tool_calls(self, mock_initialize):
        """Test running an agent with no tool calls."""
        mock_agent = Mock(spec=Agent)
        mock_initialize.return_value = (0, mock_agent)

        # Mock response with no tool calls
        mock_agent.get_response.return_value = {
            "content": "Hello, I'm an agent",
            "tool_calls": [],
        }

        result = self.agent_manager.run_agent("test_agent", "user message")

        mock_initialize.assert_called_once_with("test_agent", "user message")
        mock_agent.get_response.assert_called_once()
        self.assertEqual(result, {"content": "Hello, I'm an agent", "tool_calls": []})

    @patch("agentflow.AgentManager.AgentManager.initialize_user_input")
    @patch("agentflow.AgentManager.AgentManager._prepare_final_messages")
    def test_run_agent_with_tool_calls(self, mock_prepare, mock_initialize):
        """Test running an agent with tool calls."""
        mock_agent = Mock(spec=Agent)
        mock_model = Mock()
        mock_agent.get_model.return_value = mock_model
        mock_initialize.return_value = (0, mock_agent)

        # Mock messages
        mock_agent.get_messages.return_value = [{"role": "user", "content": "Hello"}]

        # Mock first response with tool calls
        mock_agent.get_response.side_effect = [
            {
                "content": "I'll help you with that",
                "tool_calls": [
                    {
                        "id": "1",
                        "function": {
                            "name": "test_tool",
                            "arguments": '{"param": "value"}',
                        },
                    }
                ],
            },
            {"content": "Here's your result", "tool_calls": []},
        ]

        # Mock assistant message
        mock_model.get_assistant_message.return_value = {
            "role": "assistant",
            "content": "I'll help you with that",
        }

        # Mock tool keys extraction
        mock_model.get_keys_in_tool_output.return_value = {
            "id": "1",
            "name": "test_tool",
            "arguments": '{"param": "value"}',
        }

        # Create a mock tool function
        mock_tool = Mock(return_value="Tool result")
        mock_tool.__name__ = "test_tool"
        mock_agent.tools = [mock_tool]

        # Mock process_tool_calls to return a simple response
        with patch.object(
            self.agent_manager,
            "_process_tool_calls",
            return_value=[
                {"id": "1", "tool_result": "Tool result", "name": "test_tool"}
            ],
        ):
            result = self.agent_manager.run_agent("test_agent", "user message")

        mock_initialize.assert_called_once_with("test_agent", "user message")
        self.assertEqual(mock_agent.get_response.call_count, 2)
        mock_prepare.assert_called_once()
        self.assertEqual(result, {"content": "Here's your result", "tool_calls": []})

    @patch("agentflow.AgentManager.AgentManager.initialize_user_input")
    def test_run_agent_with_container_tool(self, mock_initialize):
        """Test running an agent with a Container tool."""
        mock_agent = Mock(spec=Agent)
        mock_model = Mock()
        mock_agent.get_model.return_value = mock_model
        mock_initialize.return_value = (0, mock_agent)

        # Mock messages
        mock_agent.get_messages.return_value = [{"role": "user", "content": "Hello"}]

        # Mock response with tool calls
        mock_agent.get_response.side_effect = [
            {
                "content": "I'll help you with that",
                "tool_calls": [
                    {
                        "id": "1",
                        "function": {
                            "name": "container_tool",
                            "arguments": '{"param": "value"}',
                        },
                    }
                ],
            },
            {"content": "Here's your result", "tool_calls": []},
        ]

        # Mock assistant message
        mock_model.get_assistant_message.return_value = {
            "role": "assistant",
            "content": "I'll help you with that",
        }

        # Mock tool keys extraction
        mock_model.get_keys_in_tool_output.return_value = {
            "id": "1",
            "name": "container_tool",
            "arguments": '{"param": "value"}',
        }

        # Create a mock container
        mock_container = Mock(spec=Container)
        mock_container.name = "container_tool"
        mock_container.run.return_value = "Container result"
        mock_agent.tools = [mock_container]

        # Mock process_tool_calls to return a simple response
        with patch.object(
            self.agent_manager,
            "_process_tool_calls",
            return_value=[
                {"id": "1", "tool_result": "Container result", "name": "container_tool"}
            ],
        ):
            result = self.agent_manager.run_agent("test_agent", "user message")

        mock_initialize.assert_called_once_with("test_agent", "user message")
        self.assertEqual(result, {"content": "Here's your result", "tool_calls": []})

    @patch("agentflow.AgentManager.AgentManager.initialize_user_input")
    def test_run_agent_stream_no_tools(self, mock_initialize):
        """Test streaming an agent with no tools."""
        mock_agent = Mock(spec=Agent)
        mock_agent.get_tools.return_value = []
        mock_initialize.return_value = (0, mock_agent)

        # Mock streaming response
        mock_agent.get_stream_response.return_value = iter(
            [{"content": "Hello"}, {"content": " world"}]
        )

        result = list(self.agent_manager.run_agent_stream("test_agent", "user message"))

        mock_initialize.assert_called_once_with("test_agent", "user message")
        mock_agent.get_stream_response.assert_called_once()
        self.assertEqual(result, [{"content": "Hello"}, {"content": " world"}])

    @patch("agentflow.AgentManager.AgentManager.initialize_user_input")
    @patch("agentflow.AgentManager.AgentManager._prepare_final_messages")
    def test_run_agent_stream_with_tool_calls(self, mock_prepare, mock_initialize):
        """Test streaming an agent with tool calls."""
        mock_agent = Mock(spec=Agent)
        mock_model = Mock()
        mock_agent.get_model.return_value = mock_model
        mock_agent.get_tools.return_value = ["tool1"]
        mock_initialize.return_value = (0, mock_agent)

        # Mock messages
        mock_agent.get_messages.return_value = [{"role": "user", "content": "Hello"}]

        # Mock response with tool calls
        mock_agent.get_response.return_value = {
            "content": "I'll help you with that",
            "tool_calls": [
                {
                    "id": "1",
                    "function": {
                        "name": "test_tool",
                        "arguments": '{"param": "value"}',
                    },
                }
            ],
        }

        # Mock streaming response after tool call
        mock_agent.get_stream_response.return_value = iter(
            [{"content": "Tool result: "}, {"content": "success"}]
        )

        # Mock assistant message
        mock_model.get_assistant_message.return_value = {
            "role": "assistant",
            "content": "I'll help you with that",
        }

        # Mock process_tool_calls to return a simple response
        with patch.object(
            self.agent_manager,
            "_process_tool_calls",
            return_value=[
                {"id": "1", "tool_result": "Tool result", "name": "test_tool"}
            ],
        ):
            result = list(
                self.agent_manager.run_agent_stream("test_agent", "user message")
            )

        mock_initialize.assert_called_once_with("test_agent", "user message")
        mock_prepare.assert_called_once()
        self.assertEqual(result, [{"content": "Tool result: "}, {"content": "success"}])
