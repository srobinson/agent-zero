import os
import sys
import time

from dotenv import load_dotenv

from agentflow.Agent import Agent
from main import AgentZero
from models.OpenAi import OpenAi

# Load environment variables from .env file
load_dotenv()


def create_agent(name, instruction):
    """Create an agent with the given name and instruction."""
    api_key = os.getenv("OPENAI_API_KEY")
    model = OpenAi(name="gpt-4o", api_key=api_key)

    agent = Agent(name=name, instruction=instruction, model=model)

    print(f"Created {name}")
    return agent


def main():
    print("=== Streaming Workflow Example ===")
    print(
        "This example demonstrates a workflow with streaming responses from each agent.\n"
    )

    # Create the AgentZero instance
    agent_zero = AgentZero()

    # Create specialized agents
    research_agent = create_agent(
        "research_agent",
        """You are a research specialist. When given a topic, provide a detailed analysis 
        covering key aspects, historical context, current trends, and future implications.""",
    )

    writing_agent = create_agent(
        "writing_agent",
        """You are a writing specialist. Take research information and transform it into 
        engaging, well-structured content with clear sections, examples, and a compelling narrative.""",
    )

    editing_agent = create_agent(
        "editing_agent",
        """You are an editing specialist. Review content for clarity, coherence, grammar, 
        style, and tone. Improve the writing while maintaining the original message.""",
    )

    # Add all agents to the manager
    agent_zero.add_agent(research_agent)
    agent_zero.add_agent(writing_agent)
    agent_zero.add_agent(editing_agent)

    # Get user input
    topic = input("Enter a topic for content creation: ")

    print(f"\nCreating content about '{topic}' with streaming output...\n")

    # Since we don't have the workflow API, we'll implement the workflow manually

    # Step 1: Research
    print(f"\n\n{'=' * 80}")
    print(f"STEP: RESEARCH - Research the topic")
    print(f"{'=' * 80}\n")

    research_prompt = f"Provide detailed research on the topic: {topic}"
    research_result = ""

    for chunk in agent_zero.run_agent_stream("research_agent", research_prompt):
        if "content" in chunk and chunk["content"]:
            print(chunk["content"], end="", flush=True)
            research_result += chunk["content"]

    # Step 2: Writing
    print(f"\n\n{'=' * 80}")
    print(f"STEP: WRITING - Write content based on research")
    print(f"{'=' * 80}\n")

    writing_prompt = f"""
    Create engaging, well-structured content based on this research:
    
    {research_result}
    
    Focus on the topic: {topic}
    """

    writing_result = ""
    for chunk in agent_zero.run_agent_stream("writing_agent", writing_prompt):
        if "content" in chunk and chunk["content"]:
            print(chunk["content"], end="", flush=True)
            writing_result += chunk["content"]

    # Step 3: Editing
    print(f"\n\n{'=' * 80}")
    print(f"STEP: EDITING - Edit and improve the content")
    print(f"{'=' * 80}\n")

    editing_prompt = f"""
    Edit and improve this content while maintaining the original message:
    
    {writing_result}
    """

    editing_result = ""
    for chunk in agent_zero.run_agent_stream("editing_agent", editing_prompt):
        if "content" in chunk and chunk["content"]:
            print(chunk["content"], end="", flush=True)
            editing_result += chunk["content"]

    # Workflow completion
    print(f"\n\n{'=' * 80}")
    print("WORKFLOW COMPLETED")
    print(f"{'=' * 80}\n")

    # Store results for reporting
    results = {
        "research": {"content": research_result},
        "writing": {"content": writing_result},
        "editing": {"content": editing_result},
    }

    print(
        f"Generated {sum(len(r.get('content', '')) for r in results.values())} total characters"
    )

    # Display a summary of the workflow results
    print("\nWorkflow Summary:")
    for step_name, result in results.items():
        content_length = len(result.get("content", ""))
        print(f"- {step_name.capitalize()}: {content_length} characters")

    print("\n\nFinal content has been created!")


if __name__ == "__main__":
    main()
if __name__ == "__main__":
    main()
