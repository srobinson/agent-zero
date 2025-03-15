import os
import time
import sys
from dotenv import load_dotenv
from agents_manager.AgentZero import AgentZero
from agents_manager.Agent import Agent
from agents_manager.models.OpenAi import OpenAi
from agents_manager.models.Anthropic import Anthropic

# Load environment variables from .env file
load_dotenv()


def create_openai_agent():
    """Create an agent using OpenAI's model with streaming capability."""
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Warning: OPENAI_API_KEY not found in environment variables.")
            return None

        model = OpenAi(name="gpt-4o", api_key=api_key)

        agent = Agent(
            name="openai_streaming_agent",
            instruction="""You are a helpful assistant that provides detailed, thoughtful responses.
            When asked a question, break down your answer into clear sections with examples where appropriate.""",
            model=model,
        )

        print(f"Created OpenAI Streaming Agent")
        return agent
    except Exception as e:
        print(f"Error creating OpenAI agent: {e}")
        return None


def create_anthropic_agent():
    """Create an agent using Anthropic's model with streaming capability."""
    try:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print("Warning: ANTHROPIC_API_KEY not found in environment variables.")
            return None

        model = Anthropic(
            name="claude-3-5-sonnet-20241022", api_key=api_key, max_tokens=1024
        )

        agent = Agent(
            name="anthropic_streaming_agent",
            instruction="""You are a helpful assistant that provides detailed, thoughtful responses.
            When asked a question, break down your answer into clear sections with examples where appropriate.""",
            model=model,
        )

        print(f"Created Anthropic Streaming Agent")
        return agent
    except Exception as e:
        print(f"Error creating Anthropic agent: {e}")
        return None


def display_streaming_response(agent_zero, agent_name, query):
    """Display a streaming response from an agent."""
    print(f"\n{'=' * 80}")
    print(f"Streaming response from {agent_name}:")
    print(f"{'=' * 80}\n")

    # Track the full response for reporting
    full_response = ""

    # Stream the response
    try:
        for chunk in agent_zero.run_agent_stream(agent_name, query):
            if "content" in chunk and chunk["content"]:
                print(chunk["content"], end="", flush=True)
                full_response += chunk["content"]
    except Exception as e:
        print(f"\nError during streaming: {e}")
        return "", []

    # Print summary of the response
    print(f"\n\n{'=' * 80}")
    print(f"Response complete - {len(full_response)} characters")
    print(f"{'=' * 80}\n")

    return full_response, []


def compare_streaming_models(agent_zero):
    """Compare streaming responses from different models."""
    openai_agent = create_openai_agent()
    anthropic_agent = create_anthropic_agent()

    if not openai_agent or not anthropic_agent:
        print("Could not create all required agents. Please check your API keys.")
        return

    # Add agents to AgentZero
    agent_zero.add_agent(openai_agent)
    agent_zero.add_agent(anthropic_agent)

    query = input("Enter a question to see streaming responses from different models: ")

    # Get streaming responses from each agent
    print("\nProcessing your question with multiple models...\n")

    openai_response, _ = display_streaming_response(
        agent_zero, "openai_streaming_agent", query
    )
    anthropic_response, _ = display_streaming_response(
        agent_zero, "anthropic_streaming_agent", query
    )

    # Compare response lengths
    print("\nResponse Comparison:")
    print(f"OpenAI response length: {len(openai_response)} characters")
    print(f"Anthropic response length: {len(anthropic_response)} characters")


def demonstrate_tool_use_streaming(agent_zero):
    """Demonstrate tool use with streaming responses."""

    # Define a simple tool
    def calculate(expression: str) -> str:
        """
        Calculate the result of a mathematical expression.

        Args:
            expression (str): A mathematical expression (e.g., "2 + 2", "5 * 10")

        Returns:
            str: The result of the calculation
        """
        try:
            # Use eval safely with only mathematical operations
            allowed_names = {"abs": abs, "round": round, "min": min, "max": max}
            code = compile(expression, "<string>", "eval")

            for name in code.co_names:
                if name not in allowed_names:
                    raise NameError(f"The use of '{name}' is not allowed")

            return str(eval(expression, {"__builtins__": {}}, allowed_names))
        except Exception as e:
            return f"Error: {str(e)}"

    # Create an agent with the calculate tool
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Warning: OPENAI_API_KEY not found in environment variables.")
            return

        model = OpenAi(name="gpt-4o", api_key=api_key)

        agent = Agent(
            name="calculator_agent",
            instruction="""You are a math assistant that can perform calculations.
            When asked to calculate something, use the calculate tool to find the answer.
            Explain your reasoning step by step.""",
            model=model,
            tools=[calculate],
        )

        # Add the agent to AgentZero
        agent_zero.add_agent(agent)

        print(f"Created Calculator Agent with tool")

        # Get a calculation query
        query = input("Enter a math problem to solve with streaming output: ")

        # Display the streaming response
        print("\nSolving your math problem...\n")

        # Stream the response
        full_response = ""
        tool_calls_seen = []

        for chunk in agent_zero.run_agent_stream("calculator_agent", query):
            if "content" in chunk and chunk["content"]:
                print(chunk["content"], end="", flush=True)
                full_response += chunk["content"]

            # Check for tool calls in the chunk
            if "tool_calls" in chunk and chunk["tool_calls"]:
                for tool_call in chunk["tool_calls"]:
                    tool_name = None
                    if hasattr(tool_call, "function") and hasattr(
                        tool_call.function, "name"
                    ):
                        tool_name = tool_call.function.name
                    elif hasattr(tool_call, "name"):
                        tool_name = tool_call.name

                    if tool_name and tool_name not in tool_calls_seen:
                        tool_calls_seen.append(tool_name)
                        print(f"\n[Using tool: {tool_name}]\n")

    except Exception as e:
        print(f"Error demonstrating tool use: {e}")


def main():
    print("=== Streaming Responses Example ===")
    print(
        "This example demonstrates how to work with streaming outputs from language models.\n"
    )

    # Create AgentZero instance
    agent_zero = AgentZero()

    while True:
        print("\nChoose an example:")
        print("1. Compare streaming responses from different models")
        print("2. Demonstrate tool use with streaming")
        print("3. Exit")

        choice = input("\nEnter your choice (1-3): ")

        if choice == "1":
            compare_streaming_models(agent_zero)
        elif choice == "2":
            demonstrate_tool_use_streaming(agent_zero)
        elif choice == "3":
            print("\nExiting example. Goodbye!")
            break
        else:
            print("\nInvalid choice. Please try again.")


if __name__ == "__main__":
    main()
