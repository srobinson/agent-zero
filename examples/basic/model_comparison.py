import os
import time
from dotenv import load_dotenv
from agents_manager.AgentZero import AgentZero
from agents_manager.Agent import Agent
from agents_manager.models.OpenAi import OpenAi
from agents_manager.models.Anthropic import Anthropic

# Load environment variables from .env file
load_dotenv()


def create_openai_agent():
    """Create an agent using OpenAI's model."""
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Warning: OPENAI_API_KEY not found in environment variables.")
            return None

        model = OpenAi(name="gpt-4o", api_key=api_key)

        agent = Agent(
            name="openai_agent",
            instruction="You are a helpful assistant that provides clear, concise, and accurate answers.",
            model=model,
        )

        return agent
    except Exception as e:
        print(f"Error creating OpenAI agent: {e}")
        return None


def create_anthropic_agent():
    """Create an agent using Anthropic's Claude model."""
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
            name="claude_agent",
            instruction="You are a helpful assistant that provides clear, concise, and accurate answers.",
            model=model,
        )

        return agent
    except Exception as e:
        print(f"Error creating Anthropic agent: {e}")
        return None


def get_model_response(agent_zero, agent_name, question):
    """Get a response from a model for a specific question."""
    if not agent_name:
        return "Agent not available"

    try:
        # Set the user message and get the response
        start_time = time.time()
        response = agent_zero.run_agent(agent_name, question)
        end_time = time.time()

        # Calculate response time
        response_time = end_time - start_time

        return {
            "content": response["content"],
            "time": response_time,
            "tokens": len(response["content"].split()),  # Rough estimate of tokens
        }
    except Exception as e:
        return f"Error: {str(e)}"


def compare_models(questions):
    """Compare responses from different models for a set of questions."""
    # Create AgentZero instance
    agent_zero = AgentZero()

    # Create agents
    openai_agent = create_openai_agent()
    anthropic_agent = create_anthropic_agent()

    # Add agents to AgentZero
    if openai_agent:
        agent_zero.add_agent(openai_agent)
    if anthropic_agent:
        agent_zero.add_agent(anthropic_agent)

    # Initialize results table
    results = []

    # Process each question
    for i, question in enumerate(questions):
        print(f"\nProcessing question {i+1}/{len(questions)}: {question}")

        # Get responses from each available model
        openai_response = (
            get_model_response(agent_zero, "openai_agent", question)
            if openai_agent
            else "Not available"
        )
        anthropic_response = (
            get_model_response(agent_zero, "claude_agent", question)
            if anthropic_agent
            else "Not available"
        )

        # Add results to table
        results.append(
            {
                "question": question,
                "openai": openai_response,
                "anthropic": anthropic_response,
            }
        )

        # Print progress
        print(f"  ✓ Responses collected")

    return results


def display_results(results):
    """Display the comparison results in a readable format."""
    print("\n=== Model Comparison Results ===\n")

    for i, result in enumerate(results):
        question = result["question"]
        print(f"Question {i+1}: {question}")
        print("-" * 80)

        # Display OpenAI response if available
        if isinstance(result["openai"], dict):
            print(f"OpenAI (gpt-4o) - {result['openai']['time']:.2f}s:")
            print(f"  {result['openai']['content']}")
        else:
            print(f"OpenAI: {result['openai']}")

        print()

        # Display Anthropic response if available
        if isinstance(result["anthropic"], dict):
            print(
                f"Anthropic (claude-3-5-sonnet) - {result['anthropic']['time']:.2f}s:"
            )
            print(f"  {result['anthropic']['content']}")
        else:
            print(f"Anthropic: {result['anthropic']}")

        print("\n" + "=" * 80 + "\n")

    # Display performance summary using standard formatting
    print("Performance Summary:")
    print("-" * 100)

    # Print header
    header = f"{'Question':<30} | {'OpenAI Time':<12} | {'OpenAI Tokens':<14} | {'Anthropic Time':<14} | {'Anthropic Tokens':<15}"
    print(header)
    print("-" * 100)

    # Print each row
    for result in results:
        question_short = (
            result["question"][:27] + "..."
            if len(result["question"]) > 30
            else result["question"].ljust(30)
        )

        # Format OpenAI metrics
        if isinstance(result["openai"], dict):
            openai_time = f"{result['openai']['time']:.2f}s".ljust(12)
            openai_tokens = str(result["openai"]["tokens"]).ljust(14)
        else:
            openai_time = "N/A".ljust(12)
            openai_tokens = "N/A".ljust(14)

        # Format Anthropic metrics
        if isinstance(result["anthropic"], dict):
            anthropic_time = f"{result['anthropic']['time']:.2f}s".ljust(14)
            anthropic_tokens = str(result["anthropic"]["tokens"]).ljust(15)
        else:
            anthropic_time = "N/A".ljust(14)
            anthropic_tokens = "N/A".ljust(15)

        row = f"{question_short} | {openai_time} | {openai_tokens} | {anthropic_time} | {anthropic_tokens}"
        print(row)

    print("-" * 100)


def main():
    print("=== Model Comparison Example ===")
    print(
        "This example compares responses from different models on the same set of questions.\n"
    )

    # Define questions for comparison
    questions = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a short poem about artificial intelligence.",
        "What are the ethical considerations of using AI in healthcare?",
        "How would you solve the trolley problem?",
    ]

    # Compare models
    results = compare_models(questions)

    # Display results
    display_results(results)


if __name__ == "__main__":
    main()
