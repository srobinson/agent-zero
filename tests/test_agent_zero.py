import unittest
from unittest.mock import Mock, patch, MagicMock
import pytest

from agents_manager.AgentZero import AgentZero
from agents_manager.Agent import Agent
from agents_manager.WorkflowManager import Workflow, WorkflowStep


class TestAgentZero(unittest.TestCase):
    """Test suite for the AgentZero class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Patch the AgentManager and WorkflowManager classes
        self.agent_manager_patcher = patch("agents_manager.AgentZero.AgentManager")
        self.workflow_manager_patcher = patch(
            "agents_manager.AgentZero.WorkflowManager"
        )

        # Get the mocks
        self.mock_agent_manager_class = self.agent_manager_patcher.start()
        self.mock_workflow_manager_class = self.workflow_manager_patcher.start()

        # Create mock instances
        self.mock_agent_manager = Mock()
        self.mock_workflow_manager = Mock()

        # Set up the mock classes to return our mock instances
        self.mock_agent_manager_class.return_value = self.mock_agent_manager
        self.mock_workflow_manager_class.return_value = self.mock_workflow_manager

        # Create the AgentZero instance
        self.agent_zero = AgentZero()

        # Create mock agents and workflows for testing
        self.mock_agent = Mock(spec=Agent)
        self.mock_agent.name = "test_agent"

        self.mock_workflow = Mock(spec=Workflow)
        self.mock_workflow.name = "test_workflow"

        self.mock_step = Mock(spec=WorkflowStep)
        self.mock_step.name = "test_step"

    def tearDown(self):
        """Clean up after each test method."""
        self.agent_manager_patcher.stop()
        self.workflow_manager_patcher.stop()

    def test_init(self):
        """Test initialization of AgentZero."""
        # Verify that the managers were created correctly
        self.mock_agent_manager_class.assert_called_once()
        self.mock_workflow_manager_class.assert_called_once_with(
            self.mock_agent_manager
        )

        # Verify that the managers were assigned correctly
        self.assertEqual(self.agent_zero.agent_manager, self.mock_agent_manager)
        self.assertEqual(self.agent_zero.workflow_manager, self.mock_workflow_manager)

    def test_add_agent(self):
        """Test adding an agent."""
        self.agent_zero.add_agent(self.mock_agent)
        self.mock_agent_manager.add_agent.assert_called_once_with(self.mock_agent)

    def test_get_agent(self):
        """Test getting an agent."""
        self.mock_agent_manager.get_agent.return_value = (0, self.mock_agent)

        result = self.agent_zero.get_agent("test_agent")

        self.mock_agent_manager.get_agent.assert_called_once_with("test_agent")
        self.assertEqual(result, (0, self.mock_agent))

    def test_run_agent(self):
        """Test running an agent."""
        expected_result = {"content": "Test result", "tool_calls": []}
        self.mock_agent_manager.run_agent.return_value = expected_result

        result = self.agent_zero.run_agent("test_agent", "Test input")

        self.mock_agent_manager.run_agent.assert_called_once_with(
            "test_agent", "Test input"
        )
        self.assertEqual(result, expected_result)

    def test_run_agent_stream(self):
        """Test streaming an agent."""
        expected_chunks = [{"content": "Chunk 1"}, {"content": "Chunk 2"}]
        self.mock_agent_manager.run_agent_stream.return_value = iter(expected_chunks)

        result = list(self.agent_zero.run_agent_stream("test_agent", "Test input"))

        self.mock_agent_manager.run_agent_stream.assert_called_once_with(
            "test_agent", "Test input"
        )
        self.assertEqual(result, expected_chunks)

    def test_create_workflow(self):
        """Test creating a workflow."""
        self.mock_workflow
