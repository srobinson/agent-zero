import json
from typing import List, Optional, Any, Generator, Dict, Callable, Tuple, Union

from agents_manager.Container import Container
from agents_manager.Agent import Agent


class AgentManager:
    """
    Manages a collection of agents and facilitates their execution.

    This class provides functionality to add, retrieve, and run agents with
    support for both streaming and non-streaming responses. It also handles
    tool execution.
    """

    def __init__(self) -> None:
        """
        Initialize the AgentManager with an empty list of agents.
        """
        self.agents: List[Agent] = []

    def add_agent(self, agent: Agent) -> None:
        """
        Add an agent to the manager's list if it doesn't already exist.

        Args:
            agent (Agent): The agent instance to add.

        Raises:
            ValueError: If the provided object is not an Agent instance.
        """
        if not isinstance(agent, Agent):
            raise ValueError("Only Agent instances can be added")

        _, existing_agent = self.get_agent(agent.name)
        if existing_agent is None:
            self.agents.append(agent)

    def get_agent(self, name: str) -> Tuple[Optional[int], Optional[Agent]]:
        """
        Retrieve an agent by name.

        Args:
            name (str): The name of the agent to find.

        Returns:
            Tuple[Optional[int], Optional[Agent]]: A tuple containing the index and agent if found,
                                                  else (None, None).
        """
        for index, agent in enumerate(self.agents):
            if agent.name == name:
                return index, agent
        return None, None

    def initialize_user_input(
        self, name: str, user_input: Optional[Any] = None
    ) -> Tuple[int, Agent]:
        """
        Initialize an agent with the given user input.

        Args:
            name (str): The name of the agent to initialize.
            user_input (Optional[Any]): User input to set for the agent.

        Returns:
            Tuple[int, Agent]: The index and initialized agent.

        Raises:
            ValueError: If no agent is found with the given name.
        """
        index, agent = self.get_agent(name)

        if agent is None:
            raise ValueError(f"No agent found with name: {name}")

        agent.set_system_message(agent.instruction)
        agent.set_tools(agent.tools)

        if user_input is not None:
            agent.set_user_message(user_input)

        return index, agent

    def _process_tool_calls(
        self, agent: Agent, tool_calls: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Process tool calls from an agent response.

        Args:
            agent (Agent): The agent that made the tool calls.
            tool_calls (List[Dict[str, Any]]): The tool calls to process.

        Returns:
            List[Dict[str, Any]]: The responses from the tools.
        """
        tool_responses = []

        for tool_call in tool_calls:
            output = agent.get_model().get_keys_in_tool_output(tool_call)
            call_id, function_name = output["id"], output["name"]

            # Parse arguments safely
            try:
                if isinstance(output["arguments"], str):
                    arguments = json.loads(output["arguments"])
                else:
                    arguments = output["arguments"]
            except json.JSONDecodeError:
                tool_responses.append(
                    {
                        "id": call_id,
                        "tool_result": "Error: Invalid JSON in arguments",
                        "name": function_name,
                    }
                )
                continue

            # Find and execute the appropriate tool
            tool_executed = False
            for tool in agent.tools:
                # Check if it's a callable function with matching name
                if (
                    isinstance(tool, Callable)
                    and hasattr(tool, "__name__")
                    and tool.__name__ == function_name
                ):
                    try:
                        tool_result = tool(**arguments)

                        # Return early if the tool result is an Agent
                        if isinstance(tool_result, Agent):
                            return [{"agent": tool_result}]

                        tool_responses.append(
                            {
                                "id": call_id,
                                "tool_result": str(tool_result),
                                "name": function_name,
                            }
                        )
                        tool_executed = True
                        break
                    except Exception as e:
                        tool_responses.append(
                            {
                                "id": call_id,
                                "tool_result": f"Error executing tool: {str(e)}",
                                "name": function_name,
                            }
                        )
                        tool_executed = True
                        break
                # Check if it's a Container with matching name
                elif isinstance(tool, Container) and tool.name == function_name:
                    try:
                        tool_result = tool.run(arguments)

                        # Return early if the tool result is an Agent
                        if isinstance(tool_result, Agent):
                            return [{"agent": tool_result}]

                        tool_responses.append(
                            {
                                "id": call_id,
                                "tool_result": str(tool_result),
                                "name": function_name,
                            }
                        )
                        tool_executed = True
                        break
                    except Exception as e:
                        tool_responses.append(
                            {
                                "id": call_id,
                                "tool_result": f"Error executing tool: {str(e)}",
                                "name": function_name,
                            }
                        )
                        tool_executed = True
                        break

            if not tool_executed:
                tool_responses.append(
                    {
                        "id": call_id,
                        "tool_result": f"Error: Tool '{function_name}' not found",
                        "name": function_name,
                    }
                )

        return tool_responses

    @staticmethod
    def _prepare_final_messages(
        agent: Agent,
        current_messages: List[Dict[str, Any]],
        tool_responses: List[Dict[str, Any]],
    ) -> None:
        """
        Prepare the final messages for an agent after tool execution.

        Args:
            agent (Agent): The agent to prepare messages for.
            current_messages (List[Dict[str, Any]]): The current message history.
            tool_responses (List[Dict[str, Any]]): The responses from tools.
        """
        tool_response = agent.get_model().get_tool_message(tool_responses)

        if isinstance(tool_response, dict):
            current_messages.append(tool_response)
        elif isinstance(tool_response, list):
            current_messages.extend(tool_response)

        agent.set_messages(current_messages)

    def run_agent(self, name: str, user_input: Optional[Any] = None) -> Dict[str, Any]:
        """
        Run a specific agent and get a non-streaming response.

        Args:
            name (str): The name of the agent to run.
            user_input (Optional[Any]): Additional user input to append to messages.

        Returns:
            Dict[str, Any]: The agent's response.

        Raises:
            ValueError: If no agent is found with the given name.
        """
        _, agent = self.initialize_user_input(name, user_input)
        response = agent.get_response()

        # If no tool calls, return the response directly
        if not response.get("tool_calls"):
            return response

        # Process tool calls
        tool_calls = response["tool_calls"]
        current_messages = agent.get_messages()

        # Add assistant message to history
        assistant_message = agent.get_model().get_assistant_message(response)
        if isinstance(assistant_message, dict):
            current_messages.append(assistant_message)
        elif isinstance(assistant_message, list):
            current_messages.extend(assistant_message)

        # Process tool calls
        tool_responses = self._process_tool_calls(agent, tool_calls)

        # Check if we need to chain to another agent
        for response in tool_responses:
            if "agent" in response:
                new_agent = response["agent"]
                if not self.get_agent(new_agent.name)[1]:
                    self.add_agent(new_agent)
                return self.run_agent(new_agent.name, user_input)

        # Prepare final messages and get response
        self._prepare_final_messages(agent, current_messages, tool_responses)
        final_response = agent.get_response()

        # Check for nested tool calls
        if final_response.get("tool_calls"):
            # Recursively handle nested tool calls
            return self.run_agent(name, None)

        return final_response

    def run_agent_stream(
        self, name: str, user_input: Optional[Any] = None
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Run a specific agent and get a streaming response.

        Args:
            name (str): The name of the agent to run.
            user_input (Optional[Any]): Additional user input to append to messages.

        Yields:
            Dict[str, Any]: Chunks of the agent's response.

        Raises:
            ValueError: If no agent is found with the given name.
        """
        position, agent = self.initialize_user_input(name, user_input)

        # Special case for the first agent with no tools
        if not agent.get_tools() and position == 0:
            yield from agent.get_stream_response()
            return

        # Get initial response to check for tool calls
        response = agent.get_response()
        if not response.get("tool_calls"):
            yield response
            return

        # Process tool calls
        tool_calls = response["tool_calls"]
        current_messages = agent.get_messages()

        # Add assistant message to history
        assistant_message = agent.get_model().get_assistant_message(response)
        if isinstance(assistant_message, dict):
            current_messages.append(assistant_message)
        elif isinstance(assistant_message, list):
            current_messages.extend(assistant_message)

        # Process tool calls
        tool_responses = self._process_tool_calls(agent, tool_calls)

        # Check if we need to chain to another agent
        for response in tool_responses:
            if "agent" in response:
                new_agent = response["agent"]
                if not self.get_agent(new_agent.name)[1]:
                    self.add_agent(new_agent)
                yield from self.run_agent_stream(new_agent.name, user_input)
                return

        # Prepare final messages and stream response
        self._prepare_final_messages(agent, current_messages, tool_responses)
        yield from agent.get_stream_response()
