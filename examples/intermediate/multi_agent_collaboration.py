import os
import sys

from dotenv import load_dotenv

from agentflow.Agent import Agent
from main import AgentZero
from models.Anthropic import Anthropic
from models.OpenAi import OpenAi

# Load environment variables from .env file
load_dotenv()


def create_research_agent():
    """Create an agent specialized in research."""
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Warning: OPENAI_API_KEY not found in environment variables.")
            return None

        model = OpenAi(name="gpt-4o", api_key=api_key)

        agent = Agent(
            name="research_agent",
            instruction="""You are a research specialist. Your role is to:
1. Analyze the user's topic
2. Break it down into key aspects that need investigation
3. Provide a structured outline for research
4. Identify important questions that need answers

Focus on being thorough and systematic in your approach. Your output should be a well-structured research plan.""",
            model=model,
        )

        print(f"Created Research Agent with {model.__class__.__name__} model")
        return agent
    except Exception as e:
        print(f"Error creating research agent: {e}")
        return None


def create_writing_agent():
    """Create an agent specialized in writing."""
    try:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print("Warning: ANTHROPIC_API_KEY not found in environment variables.")
            return None

        model = Anthropic(
            name="claude-3-5-sonnet-20241022", api_key=api_key, max_tokens=1024
        )

        agent = Agent(
            name="writing_agent",
            instruction="""You are a writing specialist. Your role is to:
1. Take research outlines and information
2. Transform them into well-written, engaging content
3. Ensure the writing is clear, concise, and appropriate for the target audience
4. Maintain a consistent tone and style

Focus on creating high-quality content that effectively communicates the information provided.""",
            model=model,
        )

        print(f"Created Writing Agent with {model.__class__.__name__} model")
        return agent
    except Exception as e:
        print(f"Error creating writing agent: {e}")
        return None


def create_critique_agent():
    """Create an agent specialized in critique and improvement."""
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Warning: OPENAI_API_KEY not found in environment variables.")
            return None

        model = OpenAi(name="gpt-4o", api_key=api_key)

        agent = Agent(
            name="critique_agent",
            instruction="""You are a critique and improvement specialist. Your role is to:
1. Review content created by others
2. Identify strengths and weaknesses
3. Suggest specific improvements
4. Check for factual accuracy, clarity, and engagement

Be constructive in your feedback, highlighting both positive aspects and areas for improvement.""",
            model=model,
        )

        print(f"Created Critique Agent with {model.__class__.__name__} model")
        return agent
    except Exception as e:
        print(f"Error creating critique agent: {e}")
        return None


def stream_agent_response(agent_zero, agent_name, prompt):
    """
    Stream the response from an agent and return the full content.

    Args:
        agent_zero: The AgentZero instance
        agent_name: The name of the agent to run
        prompt: The prompt to send to the agent

    Returns:
        The full content of the agent's response
    """
    print(f"\nStreaming response from {agent_name}...")
    print("-" * 80)

    full_content = ""
    for chunk in agent_zero.run_agent_stream(agent_name, prompt):
        if "content" in chunk and chunk["content"]:
            content = chunk["content"]
            full_content += content
            sys.stdout.write(content)
            sys.stdout.flush()

    print("\n" + "-" * 80)
    return full_content


def main():
    print("=== Workflow Collaboration Example (Streaming) ===")
    print(
        "This example demonstrates how to use AgentZero to orchestrate collaboration between specialized agents with streaming responses.\n"
    )

    # Create the AgentZero instance
    agent_zero = AgentZero()

    # Create the specialized agents
    research_agent = create_research_agent()
    writing_agent = create_writing_agent()
    critique_agent = create_critique_agent()

    if not research_agent or not writing_agent or not critique_agent:
        print("Failed to create all required agents. Exiting.")
        return

    # Add all agents to AgentZero
    agent_zero.add_agent(research_agent)
    agent_zero.add_agent(writing_agent)
    agent_zero.add_agent(critique_agent)

    # Get the topic from user input
    topic = input(
        "Enter a topic for content creation (e.g., 'The impact of artificial intelligence on the future of work'): "
    )
    if not topic:
        topic = "The impact of artificial intelligence on the future of work"
        print(f"Using default topic: '{topic}'")

    print(f"\nStarting workflow on topic: '{topic}'\n")

    # Step 1: Research with streaming
    print("\n=== STEP 1: RESEARCH PLAN ===")
    research_prompt = f"Create a comprehensive research plan for the topic: {topic}"
    research_content = stream_agent_response(
        agent_zero, "research_agent", research_prompt
    )

    # Step 2: Writing with streaming
    print("\n=== STEP 2: ARTICLE DRAFT ===")
    writing_prompt = f"""Based on the following research plan, write an engaging article about {topic}:

{research_content}

Create a well-structured article with an introduction, body paragraphs, and conclusion."""
    writing_content = stream_agent_response(agent_zero, "writing_agent", writing_prompt)

    # Step 3: Critique with streaming
    print("\n=== STEP 3: CRITIQUE AND FEEDBACK ===")
    critique_prompt = f"""Review and provide constructive feedback on the following article about {topic}:

{writing_content}

Identify strengths, weaknesses, and suggest specific improvements."""
    critique_content = stream_agent_response(
        agent_zero, "critique_agent", critique_prompt
    )

    # Store results
    results = {
        "research": research_content,
        "writing": writing_content,
        "critique": critique_content,
    }

    print("\nWorkflow completed!")

    # Optional: Save results to files
    try:
        os.makedirs("output", exist_ok=True)
        with open(f"output/research_plan.txt", "w") as f:
            f.write(results["research"])
        with open(f"output/article_draft.txt", "w") as f:
            f.write(results["writing"])
        with open(f"output/critique.txt", "w") as f:
            f.write(results["critique"])
        print("Results saved to 'output' directory.")
    except Exception as e:
        print(f"Error saving results: {e}")


if __name__ == "__main__":
    main()
if __name__ == "__main__":
    main()
