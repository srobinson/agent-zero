import unittest
from unittest.mock import Mock, patch

from agentflow.Agent import Agent
from agentflow.Workflow import AgentStep, AgentWorkflow


class TestAgentStep(unittest.TestCase):
    """Test suite for the AgentStep class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a mock agent
        self.mock_agent = Mock(spec=Agent)
        self.mock_agent.name = "test_agent"

        # Create a step
        self.step = AgentStep(self.mock_agent, "test_step", "Test step description")

    def test_init(self):
        """Test initialization of AgentStep."""
        # Test with explicit description
        step = AgentStep(self.mock_agent, "test_step", "Test step description")
        self.assertEqual(step.agent, self.mock_agent)
        self.assertEqual(step.name, "test_step")
        self.assertEqual(step.description, "Test step description")
        self.assertEqual(step.next_steps, [])
        self.assertIsNone(step.result)

        # Test with default description
        step = AgentStep(self.mock_agent, "test_step")
        self.assertEqual(
            step.description, f"Step executed by agent {self.mock_agent.name}"
        )

    def test_then_without_condition(self):
        """Test adding a next step without a condition."""
        # Create another step
        next_step = AgentStep(self.mock_agent, "next_step")

        # Add it as a next step
        result = self.step.then(next_step)

        # Check that the next step was added correctly
        self.assertEqual(len(self.step.next_steps), 1)
        self.assertEqual(self.step.next_steps[0]["step"], next_step)
        self.assertIsNone(self.step.next_steps[0]["condition"])

        # Check that the method returns the next step for chaining
        self.assertEqual(result, next_step)

    def test_then_with_condition(self):
        """Test adding a next step with a condition."""
        # Create another step and a condition function
        next_step = AgentStep(self.mock_agent, "next_step")
        condition = lambda result: "success" in result

        # Add it as a next step with the condition
        result = self.step.then(next_step, condition)

        # Check that the next step was added correctly
        self.assertEqual(len(self.step.next_steps), 1)
        self.assertEqual(self.step.next_steps[0]["step"], next_step)
        self.assertEqual(self.step.next_steps[0]["condition"], condition)

        # Check that the method returns the next step for chaining
        self.assertEqual(result, next_step)

    def test_multiple_next_steps(self):
        """Test adding multiple next steps."""
        # Create multiple steps
        step1 = AgentStep(self.mock_agent, "step1")
        step2 = AgentStep(self.mock_agent, "step2")
        step3 = AgentStep(self.mock_agent, "step3")

        # Add them as next steps
        self.step.then(step1)
        self.step.then(step2)
        self.step.then(step3)

        # Check that all steps were added correctly
        self.assertEqual(len(self.step.next_steps), 3)
        self.assertEqual(self.step.next_steps[0]["step"], step1)
        self.assertEqual(self.step.next_steps[1]["step"], step2)
        self.assertEqual(self.step.next_steps[2]["step"], step3)

    def test_str_representation(self):
        """Test string representation of AgentStep."""
        expected = "AgentStep(name='test_step', agent='test_agent')"
        self.assertEqual(str(self.step), expected)

    def test_repr_representation(self):
        """Test detailed string representation of AgentStep."""
        # Initially there are no next steps
        expected = "AgentStep(name='test_step', agent='test_agent', next_steps=[])"
        self.assertEqual(repr(self.step), expected)

        # Add a next step
        next_step = AgentStep(self.mock_agent, "next_step")
        self.step.then(next_step)

        # Check the updated representation
        expected = (
            "AgentStep(name='test_step', agent='test_agent', next_steps=[next_step])"
        )
        self.assertEqual(repr(self.step), expected)


class TestAgentWorkflow(unittest.TestCase):
    """Test suite for the AgentWorkflow class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a mock agent
        self.mock_agent = Mock(spec=Agent)
        self.mock_agent.name = "test_agent"

        # Create a workflow
        self.workflow = AgentWorkflow("test_workflow", "Test workflow description")

        # Create some steps
        self.step1 = AgentStep(self.mock_agent, "step1", "Step 1 description")
        self.step2 = AgentStep(self.mock_agent, "step2", "Step 2 description")
        self.step3 = AgentStep(self.mock_agent, "step3", "Step 3 description")

    def test_init(self):
        """Test initialization of AgentWorkflow."""
        # Test with explicit description
        workflow = AgentWorkflow("test_workflow", "Test workflow description")
        self.assertEqual(workflow.name, "test_workflow")
        self.assertEqual(workflow.description, "Test workflow description")
        self.assertEqual(workflow.steps, {})
        self.assertIsNone(workflow.start_step)
        self.assertEqual(workflow.results, {})

        # Test with default description
        workflow = AgentWorkflow("test_workflow")
        self.assertEqual(workflow.description, "Workflow test_workflow")

    def test_add_step(self):
        """Test adding a step to the workflow."""
        # Add a step
        self.workflow.add_step(self.step1)

        # Check that the step was added correctly
        self.assertEqual(len(self.workflow.steps), 1)
        self.assertEqual(self.workflow.steps["step1"], self.step1)

        # Check that the first step added becomes the start step
        self.assertEqual(self.workflow.start_step, self.step1)

        # Add another step
        self.workflow.add_step(self.step2)

        # Check that the step was added correctly
        self.assertEqual(len(self.workflow.steps), 2)
        self.assertEqual(self.workflow.steps["step2"], self.step2)

        # Check that the start step hasn't changed
        self.assertEqual(self.workflow.start_step, self.step1)

    def test_set_start_step(self):
        """Test setting the starting step for the workflow."""
        # Add steps
        self.workflow.add_step(self.step1)
        self.workflow.add_step(self.step2)

        # Set the start step
        self.workflow.set_start_step(self.step2)

        # Check that the start step was set correctly
        self.assertEqual(self.workflow.start_step, self.step2)

        # Test with a step that's not in the workflow
        with self.assertRaises(ValueError):
            self.workflow.set_start_step(self.step3)

    def test_get_step(self):
        """Test getting a step by name."""
        # Add steps
        self.workflow.add_step(self.step1)
        self.workflow.add_step(self.step2)

        # Get steps
        self.assertEqual(self.workflow.get_step("step1"), self.step1)
        self.assertEqual(self.workflow.get_step("step2"), self.step2)
        self.assertIsNone(self.workflow.get_step("nonexistent_step"))

    def test_get_result(self):
        """Test getting the result of a specific step."""
        # Add a result
        self.workflow.results["step1"] = {"output": "Step 1 result"}

        # Get results
        self.assertEqual(self.workflow.get_result("step1"), {"output": "Step 1 result"})
        self.assertIsNone(self.workflow.get_result("nonexistent_step"))

    def test_starts_with(self):
        """Test setting the starting step with the starts_with method."""
        # Add a step
        self.workflow.add_step(self.step1)

        # Set the start step using starts_with
        result = self.workflow.starts_with(self.step2)

        # Check that the step was added and set as the start step
        self.assertEqual(self.workflow.steps["step2"], self.step2)
        self.assertEqual(self.workflow.start_step, self.step2)

        # Check that the method returns the step for chaining
        self.assertEqual(result, self.step2)

    def test_str_representation(self):
        """Test string representation of AgentWorkflow."""
        # Add steps
        self.workflow.add_step(self.step1)
        self.workflow.add_step(self.step2)

        expected = "AgentWorkflow(name='test_workflow', steps=2, start='step1')"
        self.assertEqual(str(self.workflow), expected)

        # Test with no start step
        workflow = AgentWorkflow("no_start")
        expected = "AgentWorkflow(name='no_start', steps=0, start='None')"
        self.assertEqual(str(workflow), expected)

    def test_repr_representation(self):
        """Test detailed string representation of AgentWorkflow."""
        # Add steps
        self.workflow.add_step(self.step1)
        self.workflow.add_step(self.step2)

        expected = "AgentWorkflow(name='test_workflow', description='Test workflow description', steps=['step1', 'step2'], start='step1')"
        self.assertEqual(repr(self.workflow), expected)

        # Test with no start step
        workflow = AgentWorkflow("no_start")
        expected = "AgentWorkflow(name='no_start', description='Workflow no_start', steps=[], start=None)"
        self.assertEqual(repr(workflow), expected)


class TestWorkflowIntegration(unittest.TestCase):
    """Test suite for integration between AgentStep and AgentWorkflow."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a mock agent
        self.mock_agent = Mock(spec=Agent)
        self.mock_agent.name = "test_agent"

        # Create a workflow
        self.workflow = AgentWorkflow("test_workflow")

    def test_workflow_with_chained_steps(self):
        """Test creating a workflow with chained steps."""
        # Create steps
        step1 = AgentStep(self.mock_agent, "step1")
        step2 = AgentStep(self.mock_agent, "step2")
        step3 = AgentStep(self.mock_agent, "step3")

        # Add steps to workflow and chain them
        self.workflow.add_step(step1)
        self.workflow.add_step(step2)
        self.workflow.add_step(step3)

        step1.then(step2)
        step2.then(step3)

        # Check the workflow structure
        self.assertEqual(self.workflow.start_step, step1)
        self.assertEqual(len(step1.next_steps), 1)
        self.assertEqual(step1.next_steps[0]["step"], step2)
        self.assertEqual(len(step2.next_steps), 1)
        self.assertEqual(step2.next_steps[0]["step"], step3)
        self.assertEqual(len(step3.next_steps), 0)

    def test_workflow_with_conditional_steps(self):
        """Test creating a workflow with conditional steps."""
        # Create steps
        step1 = AgentStep(self.mock_agent, "step1")
        step2a = AgentStep(self.mock_agent, "step2a")
        step2b = AgentStep(self.mock_agent, "step2b")
        step3 = AgentStep(self.mock_agent, "step3")

        # Create conditions
        condition_a = lambda result: "route_a" in result
        condition_b = lambda result: "route_b" in result

        # Add steps to workflow and set up conditional branches
        self.workflow.add_step(step1)
        self.workflow.add_step(step2a)
        self.workflow.add_step(step2b)
        self.workflow.add_step(step3)

        step1.then(step2a, condition_a)
        step1.then(step2b, condition_b)
        step2a.then(step3)
        step2b.then(step3)

        # Check the workflow structure
        self.assertEqual(self.workflow.start_step, step1)
        self.assertEqual(len(step1.next_steps), 2)
        self.assertEqual(step1.next_steps[0]["step"], step2a)
        self.assertEqual(step1.next_steps[0]["condition"], condition_a)
        self.assertEqual(step1.next_steps[1]["step"], step2b)
        self.assertEqual(step1.next_steps[1]["condition"], condition_b)
        self.assertEqual(len(step2a.next_steps), 1)
        self.assertEqual(step2a.next_steps[0]["step"], step3)
        self.assertEqual(len(step2b.next_steps), 1)
        self.assertEqual(step2b.next_steps[0]["step"], step3)

    def test_fluent_workflow_creation(self):
        """Test creating a workflow using the fluent interface."""
        # Create steps
        step1 = AgentStep(self.mock_agent, "step1")
        step2 = AgentStep(self.mock_agent, "step2")
        step3 = AgentStep(self.mock_agent, "step3")

        # Create workflow using fluent interface
        self.workflow.starts_with(step1).then(step2).then(step3)

        # Check the workflow structure
        self.assertEqual(self.workflow.start_step, step1)
        self.assertEqual(len(step1.next_steps), 1)
        self.assertEqual(step1.next_steps[0]["step"], step2)
        self.assertEqual(len(step2.next_steps), 1)
        self.assertEqual(step2.next_steps[0]["step"], step3)
        self.assertEqual(len(step3.next_steps), 0)


if __name__ == "__main__":
    unittest.main()
