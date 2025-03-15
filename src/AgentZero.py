"""
AgentZero module - the main entry point for the agents_manager library.

This module provides a unified interface for managing agents and workflows,
simplifying the creation and orchestration of multi-agent systems.
"""

from typing import Dict, List, Any, Optional, Callable, Union, Generator

from agents_manager.Agent import Agent
from agents_manager.AgentManager import AgentManager
from agents_manager.WorkflowManager import WorkflowManager, Workflow, WorkflowStep


class AgentZero:
    """
    Main entry point for the agents_manager library.

    This class delegates to AgentManager and WorkflowManager to provide
    a unified interface for managing agents and workflows.
    """

    def __init__(self):
        """Initialize AgentZero with AgentManager and WorkflowManager."""
        self.agent_manager = AgentManager()
        self.workflow_manager = WorkflowManager(self.agent_manager)

    # Agent management methods - delegate to AgentManager

    def add_agent(self, agent: Agent) -> None:
        """Add an agent to the manager."""
        self.agent_manager.add_agent(agent)

    def get_agent(self, name: str) -> tuple[Optional[int], Optional[Agent]]:
        """Get an agent by name."""
        return self.agent_manager.get_agent(name)

    def run_agent(self, name: str, user_input: Optional[Any] = None) -> Dict[str, Any]:
        """Run an agent and get its response."""
        return self.agent_manager.run_agent(name, user_input)

    def run_agent_stream(
        self, name: str, user_input: Optional[Any] = None
    ) -> Generator[Dict[str, Any], None, None]:
        """Run an agent with streaming output."""
        yield from self.agent_manager.run_agent_stream(name, user_input)

    # Workflow methods - delegate to WorkflowManager

    def create_workflow(self, name: str, description: Optional[str] = None) -> Workflow:
        """Create a new workflow."""
        return self.workflow_manager.create_workflow(name, description)

    def create_step(
        self, agent: Union[str, Agent], name: str, description: Optional[str] = None
    ) -> WorkflowStep:
        """Create a workflow step."""
        return self.workflow_manager.create_step(agent, name, description)

    def run_workflow(
        self, workflow_name: str, input_data: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Run a workflow from start to finish."""
        return self.workflow_manager.run_workflow(workflow_name, input_data)

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
        self.workflow_manager.stream_workflow(
            workflow_name,
            input_data,
            step_callback,
            chunk_callback,
            completion_callback,
        )
