import unittest
from unittest.mock import Mock, patch, MagicMock
import pytest

from agents_manager.WorkflowManager import WorkflowManager, Workflow, WorkflowStep
from agents_manager.AgentManager import AgentManager
from agents_manager.Agent import Agent


class TestWorkflowManager(unittest.TestCase):
    """Test suite for the WorkflowManager class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.agent_manager = Mock(spec=AgentManager)
        self.agent_manager.initialize_user_input = Mock()
        self.workflow_manager = WorkflowManager(self.agent_manager)

        # Create mock agents
        self.mock_agent1 = Mock(spec=Agent)
        self.mock_agent1.name = "agent1"
        self.mock_agent1.instruction = "Agent 1 instruction"
        self.mock_agent1.tools = []

        self.mock_agent2 = Mock(spec=Agent)
        self.mock_agent2.name = "agent2"
        self.mock_agent2.instruction = "Agent 2 instruction"
        self.mock_agent2.tools = []

        # Mock agent_manager.get_agent to return our mock agents
        self.agent_manager.get_agent.side_effect = lambda name: (
            (0, self.mock_agent1)
            if name == "agent1"
            else (1, self.mock_agent2) if name == "agent2" else (None, None)
        )

    def test_init(self):
        """Test initialization of WorkflowManager."""
        self.assertEqual(self.workflow_manager.agent_manager, self.agent_manager)
        self.assertEqual(self.workflow_manager.workflows, {})

    def test_create_workflow(self):
        """Test creating a workflow."""
        workflow = self.workflow_manager.create_workflow(
            "test_workflow", "Test workflow description"
        )

        self.assertEqual(workflow.name, "test_workflow")
        self.assertEqual(workflow.description, "Test workflow description")
        self.assertIsNone(workflow.first_step)
        self.assertEqual(workflow.results, {})
        self.assertEqual(self.workflow_manager.workflows["test_workflow"], workflow)

    def test_create_workflow_default_description(self):
        """Test creating a workflow with default description."""
        workflow = self.workflow_manager.create_workflow("test_workflow")

        self.assertEqual(workflow.name, "test_workflow")
        self.assertEqual(workflow.description, "Workflow test_workflow")
        self.assertIsNone(workflow.first_step)
        self.assertEqual(workflow.results, {})

    def test_create_step_with_agent_object(self):
        """Test creating a step with an Agent object."""
        step = self.workflow_manager.create_step(
            self.mock_agent1, "step1", "Step 1 description"
        )

        self.assertEqual(step.name, "step1")
        self.assertEqual(step.agent, self.mock_agent1)
        self.assertEqual(step.description, "Step 1 description")
        self.assertIsNone(step.next_step)
        self.agent_manager.add_agent.assert_called_once_with(self.mock_agent1)

    def test_create_step_with_agent_name(self):
        """Test creating a step with an agent name."""
        step = self.workflow_manager.create_step(
            "agent1", "step1", "Step 1 description"
        )

        self.assertEqual(step.name, "step1")
        self.assertEqual(step.agent, self.mock_agent1)
        self.assertEqual(step.description, "Step 1 description")
        self.assertIsNone(step.next_step)
        self.agent_manager.get_agent.assert_called_with("agent1")

    def test_create_step_with_nonexistent_agent(self):
        """Test creating a step with a nonexistent agent raises ValueError."""
        with self.assertRaises(ValueError):
            self.workflow_manager.create_step("nonexistent_agent", "step1")

    def test_create_step_default_description(self):
        """Test creating a step with default description."""
        step = self.workflow_manager.create_step(self.mock_agent1, "step1")

        self.assertEqual(step.name, "step1")
        self.assertEqual(step.description, f"Step executed by {self.mock_agent1.name}")

    def test_workflow_starts_with(self):
        """Test setting the first step of a workflow."""
        workflow = self.workflow_manager.create_workflow("test_workflow")
        step = self.workflow_manager.create_step(self.mock_agent1, "step1")

        result = workflow.starts_with(step)

        self.assertEqual(workflow.first_step, step)
        self.assertEqual(result, step)  # Should return the step for chaining

    def test_step_then(self):
        """Test chaining steps."""
        step1 = self.workflow_manager.create_step(self.mock_agent1, "step1")
        step2 = self.workflow_manager.create_step(self.mock_agent2, "step2")

        result = step1.then(step2)

        self.assertEqual(step1.next_step, step2)
        self.assertEqual(result, step2)  # Should return the next step for chaining

    def test_run_workflow(self):
        """Test running a workflow."""
        # Create workflow and steps
        workflow = self.workflow_manager.create_workflow("test_workflow")
        step1 = self.workflow_manager.create_step(self.mock_agent1, "step1")
        step2 = self.workflow_manager.create_step(self.mock_agent2, "step2")
        workflow.starts_with(step1)
        step1.then(step2)

        # Mock agent_manager methods
        self.agent_manager.run_agent.side_effect = [
            {"content": "Step 1 result", "tool_calls": []},
            {"content": "Step 2 result", "tool_calls": []},
        ]

        # Run the workflow
        results = self.workflow_manager.run_workflow("test_workflow", "Initial input")

        # Verify the agent_manager methods were called correctly
        self.agent_manager.initialize_user_input.assert_any_call(
            "agent1", "Initial input"
        )
        self.agent_manager.initialize_user_input.assert_any_call(
            "agent2", "Step 1 result"
        )
        self.agent_manager.run_agent.assert_any_call("agent1")
        self.agent_manager.run_agent.assert_any_call("agent2")

        # Verify the results
        self.assertEqual(
            results,
            {
                "step1": {"content": "Step 1 result", "tool_calls": []},
                "step2": {"content": "Step 2 result", "tool_calls": []},
            },
        )
        self.assertEqual(step1.result, {"content": "Step 1 result", "tool_calls": []})
        self.assertEqual(step2.result, {"content": "Step 2 result", "tool_calls": []})

    def test_run_workflow_nonexistent(self):
        """Test running a nonexistent workflow raises ValueError."""
        with self.assertRaises(ValueError):
            self.workflow_manager.run_workflow("nonexistent_workflow")

    def test_run_workflow_no_first_step(self):
        """Test running a workflow with no first step raises ValueError."""
        workflow = self.workflow_manager.create_workflow("test_workflow")
        with self.assertRaises(ValueError):
            self.workflow_manager.run_workflow("test_workflow")

    def test_stream_workflow(self):
        """Test streaming a workflow."""
        # Create workflow and steps
        workflow = self.workflow_manager.create_workflow("test_workflow")
        step1 = self.workflow_manager.create_step(self.mock_agent1, "step1")
        step2 = self.workflow_manager.create_step(self.mock_agent2, "step2")
        workflow.starts_with(step1)
        step1.then(step2)

        # Mock agent_manager methods
        self.agent_manager.run_agent_stream.side_effect = [
            iter([{"content": "Step 1 chunk 1"}, {"content": "Step 1 chunk 2"}]),
            iter([{"content": "Step 2 chunk 1"}, {"content": "Step 2 chunk 2"}]),
        ]
        self.agent_manager.run_agent.side_effect = [
            {"content": "Step 1 result", "tool_calls": []},
            {"content": "Step 2 result", "tool_calls": []},
        ]

        # Create mock callbacks
        step_callback = Mock()
        chunk_callback = Mock()
        completion_callback = Mock()

        # Run the workflow
        self.workflow_manager.stream_workflow(
            "test_workflow",
            "Initial input",
            step_callback=step_callback,
            chunk_callback=chunk_callback,
            completion_callback=completion_callback,
        )

        # Verify the agent_manager methods were called correctly
        self.agent_manager.initialize_user_input.assert_any_call(
            "agent1", "Initial input"
        )
        self.agent_manager.initialize_user_input.assert_any_call(
            "agent2", "Step 1 result"
        )
        self.agent_manager.run_agent_stream.assert_any_call("agent1")
        self.agent_manager.run_agent_stream.assert_any_call("agent2")
        self.agent_manager.run_agent.assert_any_call("agent1")
        self.agent_manager.run_agent.assert_any_call("agent2")

        # Verify the callbacks were called correctly
        step_callback.assert_any_call(
            "step1",
            {
                "name": "step1",
                "description": "Step executed by agent1",
                "agent": "agent1",
            },
        )
        step_callback.assert_any_call(
            "step2",
            {
                "name": "step2",
                "description": "Step executed by agent2",
                "agent": "agent2",
            },
        )
        chunk_callback.assert_any_call("step1", {"content": "Step 1 chunk 1"})
        chunk_callback.assert_any_call("step1", {"content": "Step 1 chunk 2"})
        chunk_callback.assert_any_call("step2", {"content": "Step 2 chunk 1"})
        chunk_callback.assert_any_call("step2", {"content": "Step 2 chunk 2"})
        completion_callback.assert_called_once_with(
            {
                "step1": {"content": "Step 1 result", "tool_calls": []},
                "step2": {"content": "Step 2 result", "tool_calls": []},
            }
        )
