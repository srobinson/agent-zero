"""
WorkflowManager module for handling agent workflows.

This module provides functionality for creating and running workflows
of agents, allowing for sequential execution and data passing between agents.
"""

from typing import Dict, List, Any, Optional, Callable, Union

from agents_manager.Agent import Agent
from agents_manager.AgentManager import AgentManager


class WorkflowStep:
    """Represents a step in a workflow."""

    def __init__(self, name: str, agent: Agent, description: Optional[str] = None):
        self.name = name
        self.agent = agent
        self.description = description or f"Step executed by {agent.name}"
        self.next_step = None
        self.result = None

    def then(self, next_step: "WorkflowStep") -> "WorkflowStep":
        """Set the next step and return it for chaining."""
        self.next_step = next_step
        return next_step


class Workflow:
    """Represents a workflow of agent steps."""

    def __init__(self, name: str, description: Optional[str] = None):
        self.name = name
        self.description = description or f"Workflow {name}"
        self.first_step = None
        self.results = {}

    def starts_with(self, step: WorkflowStep) -> WorkflowStep:
        """Set the first step and return it for chaining."""
        self.first_step = step
        return step

    def get_result(self, step_name: str) -> Optional[Dict[str, Any]]:
        """Get the result of a specific step."""
        return self.results.get(step_name)


class WorkflowManager:
    """Manager class for creating and running workflows."""

    def __init__(self, agent_manager: AgentManager):
        """Initialize with an AgentManager instance."""
        self.agent_manager = agent_manager
        self.workflows = {}

    def create_workflow(self, name: str, description: Optional[str] = None) -> Workflow:
        """Create a new workflow."""
        workflow = Workflow(name, description)
        self.workflows[name] = workflow
        return workflow

    def create_step(
        self, agent: Union[str, Agent], name: str, description: Optional[str] = None
    ) -> WorkflowStep:
        """Create a workflow step."""
        # If agent is a string, get the actual agent
        if isinstance(agent, str):
            _, agent_obj = self.agent_manager.get_agent(agent)
            if agent_obj is None:
                raise ValueError(f"No agent found with name: {agent}")
        else:
            agent_obj = agent
            # Make sure the agent is in the manager
            self.agent_manager.add_agent(agent_obj)

        return WorkflowStep(name, agent_obj, description)

    def run_workflow(
        self, workflow_name: str, input_data: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Run a workflow from start to finish."""
        workflow = self.workflows.get(workflow_name)
        if not workflow:
            raise ValueError(f"Workflow '{workflow_name}' not found")

        if not workflow.first_step:
            raise ValueError(f"Workflow '{workflow_name}' has no starting step")

        # Clear previous results
        workflow.results = {}

        # Start with the first step
        current_step = workflow.first_step
        current_input = input_data

        # Execute steps until we reach a step with no next step
        while current_step:
            print(f"\nExecuting step: {current_step.name}")

            # Initialize the agent with the input
            # Changed from _initialize_user_input to initialize_user_input
            self.agent_manager.initialize_user_input(
                current_step.agent.name, current_input
            )

            # Run the agent
            result = self.agent_manager.run_agent(current_step.agent.name)

            # Store the result
            current_step.result = result
            workflow.results[current_step.name] = result

            # Prepare for the next step
            current_step = current_step.next_step
            if current_step:
                current_input = result.get("content", "")

        return workflow.results

    def stream_workflow(
        self,
        workflow_name: str,
        input_data: Optional[Any] = None,
        step_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        chunk_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        completion_callback: Optional[
            Callable[[Dict[str, Dict[str, Any]]], None]
        ] = None,
    ) -> None:
        """Run a workflow with streaming output."""
        workflow = self.workflows.get(workflow_name)
        if not workflow:
            raise ValueError(f"Workflow '{workflow_name}' not found")

        if not workflow.first_step:
            raise ValueError(f"Workflow '{workflow_name}' has no starting step")

        # Clear previous results
        workflow.results = {}

        # Start with the first step
        current_step = workflow.first_step
        current_input = input_data

        # Execute steps until we reach a step with no next step
        while current_step:
            # Call step callback if provided
            if step_callback:
                step_info = {
                    "name": current_step.name,
                    "description": current_step.description,
                    "agent": current_step.agent.name,
                }
                step_callback(current_step.name, step_info)

            # Initialize the agent with the input
            # Changed from _initialize_user_input to initialize_user_input
            self.agent_manager.initialize_user_input(
                current_step.agent.name, current_input
            )

            # Stream the agent's response
            for chunk in self.agent_manager.run_agent_stream(current_step.agent.name):
                if chunk_callback:
                    chunk_callback(current_step.name, chunk)

            # Get the final result
            result = self.agent_manager.run_agent(current_step.agent.name)

            # Store the result
            current_step.result = result
            workflow.results[current_step.name] = result

            # Prepare for the next step
            current_step = current_step.next_step
            if current_step:
                current_input = result.get("content", "")

        # Call completion callback if provided
        if completion_callback:
            completion_callback(workflow.results)
