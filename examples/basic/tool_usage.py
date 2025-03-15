import os
import json
import math
import signal
import requests
from datetime import datetime
from dotenv import load_dotenv
from agents_manager.AgentZero import AgentZero
from agents_manager.Agent import Agent
from agents_manager.models.OpenAi import OpenAi
from agents_manager.models.Anthropic import Anthropic

# Load environment variables from .env file
load_dotenv()

# Define some simple tools (functions) that the agent can use


def calculate(expression: str) -> str:
    """
    Evaluate a mathematical expression.

    Args:
        expression (str): A mathematical expression to evaluate (e.g., "2 + 2", "sin(30)", "sqrt(16)")

    Returns:
        str: The result of the evaluation
    """
    try:
        # Replace common mathematical functions with their math module equivalents
        expression = expression.replace("sin", "math.sin")
        expression = expression.replace("cos", "math.cos")
        expression = expression.replace("tan", "math.tan")
        expression = expression.replace("sqrt", "math.sqrt")
        expression = expression.replace("pi", "math.pi")
        expression = expression.replace("e", "math.e")

        # Safely evaluate the expression
        result = eval(expression, {"__builtins__": None}, {"math": math})
        return f"Result: {result}"
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"


def get_current_weather(location: str) -> str:
    """
    Get the current weather for a location.

    Args:
        location (str): The city and optionally country (e.g., "New York, US")

    Returns:
        str: Weather information for the location
    """
    try:
        # This is a mock implementation - in a real application, you would use a weather API
        weather_data = {
            "New York": {"temperature": 72, "condition": "Sunny", "humidity": 50},
            "London": {"temperature": 62, "condition": "Cloudy", "humidity": 80},
            "Tokyo": {"temperature": 85, "condition": "Partly Cloudy", "humidity": 70},
            "Sydney": {"temperature": 68, "condition": "Rainy", "humidity": 90},
        }

        # Normalize location name for lookup
        location_key = location.split(",")[0].strip()

        if location_key in weather_data:
            data = weather_data[location_key]
            return (
                f"Weather in {location_key}: {data['condition']}, "
                f"Temperature: {data['temperature']}°F, Humidity: {data['humidity']}%"
            )
        else:
            return f"Weather data not available for {location}"
    except Exception as e:
        return f"Error getting weather: {str(e)}"


def get_current_time(timezone: str) -> str:
    """
    Get the current time in a specific timezone.

    Args:
        timezone (str): The timezone (e.g., "UTC", "EST", "PST")

    Returns:
        str: The current time in the specified timezone
    """
    try:
        # This is a simplified implementation without actual timezone conversion
        now = datetime.now()
        return f"Current time ({timezone}): {now.strftime('%Y-%m-%d %H:%M:%S')}"
    except Exception as e:
        return f"Error getting time: {str(e)}"


def create_openai_agent_with_tools():
    """Create an OpenAI agent with tools."""
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Warning: OPENAI_API_KEY not found in environment variables.")
            return None

        model = OpenAi(name="gpt-4o", api_key=api_key)

        agent = Agent(
            name="openai_tool_assistant",
            instruction="You are a helpful assistant with access to tools. Use the appropriate tool when needed to answer user questions.",
            model=model,
            tools=[calculate, get_current_weather, get_current_time],
        )

        print(f"Model(name='{model.name}', type={model.__class__.__name__})")
        print(
            f"Agent(name='{agent.name}', model={model.__class__.__name__}, tools={len(agent.tools)})"
        )

        return agent
    except Exception as e:
        print(f"Error creating OpenAI agent: {e}")
        return None


def create_anthropic_agent_with_tools():
    """Create an Anthropic agent with tools."""
    try:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print("Warning: ANTHROPIC_API_KEY not found in environment variables.")
            return None

        model = Anthropic(
            name="claude-3-5-sonnet-20241022",
            api_key=api_key,
            max_tokens=1024,  # Required parameter for Anthropic
        )

        agent = Agent(
            name="claude_tool_assistant",
            instruction="You are a helpful assistant with access to tools. Use the appropriate tool when needed to answer user questions.",
            model=model,
            tools=[calculate, get_current_weather, get_current_time],
        )

        print(f"Model(name='{model.name}', type={model.__class__.__name__})")
        print(
            f"Agent(name='{agent.name}', model={model.__class__.__name__}, tools={len(agent.tools)})"
        )

        return agent
    except Exception as e:
        print(f"Error creating Anthropic agent: {e}")
        return None


def main():
    # Create AgentZero instance
    agent_zero = AgentZero()

    # Create an agent with tools
    openai_agent = create_openai_agent_with_tools()

    if not openai_agent:
        print("Failed to create OpenAI agent. Exiting.")
        return

    # Add the agent to AgentZero
    agent_zero.add_agent(openai_agent)

    # Example questions that require tool use
    questions = [
        "What is the square root of 144?",
        "What's the current weather in New York?",
        "What time is it in UTC?",  # Modified to provide a timezone
        "If I have a triangle with sides of length 3, 4, and 5, what is its area?",
    ]

    # Process each question
    for i, question in enumerate(questions):
        print(f"\n\n--- Question {i+1}: {question} ---")

        try:
            # Get the response using AgentZero
            response = agent_zero.run_agent("openai_tool_assistant", question)

            # Print the response
            print("\nResponse:")
            print(response["content"])

            # Print tool usage if any
            if response.get("tool_calls"):
                print("\nTool Calls:")
                for tool_call in response["tool_calls"]:
                    print(f"  Tool: {tool_call.function.name}")
                    print(f"  Arguments: {tool_call.function.arguments}")
        except Exception as e:
            print(f"Error processing question: {e}")

    # Try with Anthropic if available
    anthropic_agent = create_anthropic_agent_with_tools()
    if anthropic_agent:
        print("\n\n=== Testing Anthropic Agent with Tools ===")

        # Add the Anthropic agent to AgentZero
        agent_zero.add_agent(anthropic_agent)

        # Define timeout handler
        def timeout_handler(signum, frame):
            raise TimeoutError("Operation timed out")

        # Set a timeout of 30 seconds
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)

        try:
            # Test with one question
            test_question = "What is the cube root of 27?"
            print(f"\nQuestion: {test_question}")

            # Get the response using AgentZero
            response = agent_zero.run_agent("claude_tool_assistant", test_question)

            # Cancel the timeout
            signal.alarm(0)

            print("\nResponse:")
            print(response["content"])

            # Print tool usage if any
            if response.get("tool_calls"):
                print("\nTool Calls:")
                for tool_call in response["tool_calls"]:
                    print(f"  Tool: {tool_call.name}")
                    print(f"  Arguments: {tool_call.input}")
        except TimeoutError as e:
            print(f"Operation timed out: {e}")
        except Exception as e:
            print(f"Error with Anthropic agent: {e}")
        finally:
            # Cancel the timeout to be safe
            signal.alarm(0)


if __name__ == "__main__":
    main()
