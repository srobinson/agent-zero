"""
Workflow module for orchestrating sequences of agent interactions.

This module provides classes for defining workflows composed of agent steps,
allowing for structured execution of multiple agents in sequence or based on conditions.
"""

from typing import List, Dict, Any, Optional, Callable

from agents_manager.Agent import Agent


class AgentStep:
    """
    Represents a step in an agent workflow.

    A step contains an agent, a name, a description, and information about
    the next steps that can follow this step. Steps can be chained together
    to form a workflow, with optional conditions determining which step to
    execute next.
    """

    def __init__(
        self, agent: Agent, name: str, description: Optional[str] = None
    ) -> None:
        """
        Initialize an AgentStep.

        Args:
            agent (Agent): The agent that will execute this step.
            name (str): The name of the step.
            description (Optional[str]): A description of what this step does.
                If not provided, a default description will be generated.
        """
        self.agent = agent
        self.name = name
        self.description = description or f"Step executed by agent {agent.name}"
        self.next_steps = []  # List of dicts with "condition" and "step" keys
        self.result = None  # Will store the result of this step's execution

    def then(
        self,
        next_step: "AgentStep",
        condition: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ) -> "AgentStep":
        """
        Add a next step to this step with an optional condition.

        This method allows for chaining steps together in a workflow. If a condition
        is provided, it will be evaluated to determine whether to execute the next step.

        Args:
            next_step (AgentStep): The step to execute next.
            condition (Optional[Callable]): A function that takes the result of this step
                and returns True if the next step should be executed, False otherwise.
                If None, the next step will always be executed.

        Returns:
            AgentStep: The next step, for method chaining.
        """
        self.next_steps.append({"step": next_step, "condition": condition})
        return next_step

    def __str__(self) -> str:
        """
        Return a string representation of the step.

        Returns:
            str: A string representation of the step.
        """
        return f"AgentStep(name='{self.name}', agent='{self.agent.name}')"

    def __repr__(self) -> str:
        """
        Return a detailed string representation of the step.

        Returns:
            str: A detailed string representation of the step.
        """
        next_steps_str = ", ".join([s["step"].name for s in self.next_steps])
        return f"AgentStep(name='{self.name}', agent='{self.agent.name}', next_steps=[{next_steps_str}])"


class AgentWorkflow:
    """
    Represents a workflow of agent steps.

    A workflow contains a collection of steps and defines the execution flow
    between them. It has a name, description, and a starting step that serves
    as the entry point for execution.
    """

    def __init__(self, name: str, description: Optional[str] = None) -> None:
        """
        Initialize an AgentWorkflow.

        Args:
            name (str): The name of the workflow.
            description (Optional[str]): A description of what this workflow does.
                If not provided, a default description will be generated.
        """
        self.name = name
        self.description = description or f"Workflow {name}"
        self.steps = {}  # Dict of step name to step
        self.start_step = None  # The first step to execute
        self.results = {}  # Dict of step name to result

    def add_step(self, step: AgentStep) -> None:
        """
        Add a step to the workflow.

        If this is the first step added, it will automatically be set as the
        starting step for the workflow.

        Args:
            step (AgentStep): The step to add.
        """
        self.steps[step.name] = step

        # If this is the first step added, set it as the start step
        if self.start_step is None:
            self.start_step = step

    def set_start_step(self, step: AgentStep) -> None:
        """
        Set the starting step for the workflow.

        Args:
            step (AgentStep): The step to start with.

        Raises:
            ValueError: If the step is not in the workflow.
        """
        if step.name not in self.steps:
            raise ValueError(f"Step {step.name} is not in the workflow")

        self.start_step = step

    def get_step(self, name: str) -> Optional[AgentStep]:
        """
        Get a step by name.

        Args:
            name (str): The name of the step to get.

        Returns:
            Optional[AgentStep]: The step if found, else None.
        """
        return self.steps.get(name)

    def get_result(self, step_name: str) -> Optional[Dict[str, Any]]:
        """
        Get the result of a specific step in the workflow.

        Args:
            step_name (str): The name of the step to get the result for.

        Returns:
            Optional[Dict[str, Any]]: The result of the step if available, else None.
        """
        return self.results.get(step_name)

    def starts_with(self, step: AgentStep) -> AgentStep:
        """
        Set the starting step for the workflow and return it for chaining.

        This method is a more fluent alternative to set_start_step that allows
        for method chaining with the 'then' method of AgentStep.

        Args:
            step (AgentStep): The step to start with.

        Returns:
            AgentStep: The starting step, for method chaining.

        Raises:
            ValueError: If the step is not in the workflow.
        """
        # Make sure the step is in the workflow
        if step.name not in self.steps:
            # If not, add it
            self.add_step(step)

        # Set it as the start step
        self.start_step = step

        # Return the step for method chaining
        return step

    def __str__(self) -> str:
        """
        Return a string representation of the workflow.

        Returns:
            str: A string representation of the workflow.
        """
        step_count = len(self.steps)
        start = self.start_step.name if self.start_step else "None"
        return f"AgentWorkflow(name='{self.name}', steps={step_count}, start='{start}')"

    def __repr__(self) -> str:
        """
        Return a detailed string representation of the workflow.

        Returns:
            str: A detailed string representation of the workflow.
        """
        steps_str = ", ".join([f"'{name}'" for name in self.steps.keys()])
        start = f"'{self.start_step.name}'" if self.start_step else "None"
        return f"AgentWorkflow(name='{self.name}', description='{self.description}', steps=[{steps_str}], start={start})"
