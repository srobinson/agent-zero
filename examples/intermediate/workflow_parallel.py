import os
import threading
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
    print("=== Parallel Workflow with Streaming Synthesis ===")
    print(
        "This example demonstrates how to process multiple tasks in parallel and then combine the results with streaming output.\n"
    )

    # Create the AgentZero instance
    agent_zero = AgentZero()

    # Create specialized agents
    market_research_agent = create_agent(
        "market_research_agent",
        "You are a market research specialist. Analyze market trends, consumer behavior, and competitive landscapes.",
    )

    technical_research_agent = create_agent(
        "technical_research_agent",
        "You are a technical research specialist. Analyze technical feasibility, implementation challenges, and technology trends.",
    )

    financial_research_agent = create_agent(
        "financial_research_agent",
        "You are a financial research specialist. Analyze costs, ROI, funding requirements, and financial projections.",
    )

    synthesis_agent = create_agent(
        "synthesis_agent",
        "You are a synthesis specialist. Combine and integrate information from multiple sources into a cohesive, comprehensive report.",
    )

    # Add all agents to the manager
    agent_zero.add_agent(market_research_agent)
    agent_zero.add_agent(technical_research_agent)
    agent_zero.add_agent(financial_research_agent)
    agent_zero.add_agent(synthesis_agent)

    # Get the product idea
    product_idea = input("Enter a product idea to analyze: ")

    print("\nRunning parallel research on your product idea...\n")

    # Store results for synthesis
    results = {}
    results_lock = threading.Lock()

    # Define a function to run an agent and store its results
    def run_agent(agent_name, query):
        # Get the response using AgentZero
        response = agent_zero.run_agent(agent_name, query)

        # Store the result for synthesis
        with results_lock:
            results[agent_name] = response
            print(f"✓ {agent_name} research completed")

    # Create and start threads for each research agent
    threads = [
        threading.Thread(
            target=run_agent,
            args=(
                "market_research_agent",
                f"Conduct market research for this product idea: {product_idea}",
            ),
        ),
        threading.Thread(
            target=run_agent,
            args=(
                "technical_research_agent",
                f"Analyze technical feasibility of this product idea: {product_idea}",
            ),
        ),
        threading.Thread(
            target=run_agent,
            args=(
                "financial_research_agent",
                f"Provide financial analysis for this product idea: {product_idea}",
            ),
        ),
    ]

    for thread in threads:
        thread.start()

    # Wait for all research to complete
    for thread in threads:
        thread.join()

    print("\nAll research completed. Synthesizing results with streaming output...\n")

    # Combine all research into a comprehensive report
    synthesis_prompt = f"""
    Create a comprehensive product analysis report by synthesizing the following research:
    
    MARKET RESEARCH:
    {results['market_research_agent']['content']}
    
    TECHNICAL RESEARCH:
    {results['technical_research_agent']['content']}
    
    FINANCIAL RESEARCH:
    {results['financial_research_agent']['content']}
    
    Format the report with clear sections, an executive summary, and recommendations.
    """

    # Print synthesis header
    print(f"\n{'=' * 30} SYNTHESIS {'=' * 30}\n")

    # Get the synthesis agent and set up for streaming
    _, synthesis_agent_obj = agent_zero.get_agent("synthesis_agent")
    synthesis_agent_obj.set_system_message(synthesis_agent_obj.instruction)
    synthesis_agent_obj.set_user_message(synthesis_prompt)

    # Stream the synthesis directly from the agent
    try:
        for chunk in synthesis_agent_obj.get_stream_response():
            # Try different ways to extract content from the chunk
            content = None
            if isinstance(chunk, dict):
                if "content" in chunk and chunk["content"]:
                    content = chunk["content"]
                elif (
                    "choices" in chunk
                    and chunk["choices"]
                    and "delta" in chunk["choices"][0]
                ):
                    delta = chunk["choices"][0]["delta"]
                    if "content" in delta and delta["content"]:
                        content = delta["content"]

            if content:
                print(content, end="", flush=True)
    except Exception as e:
        print(f"\nError during streaming: {e}")
        # Fall back to non-streaming if streaming fails
        print("\nFalling back to non-streaming synthesis...")
        response = agent_zero.run_agent("synthesis_agent", synthesis_prompt)
        print(response["content"])

    print(f"\n{'=' * 80}\n")
    print("\nParallel workflow with synthesis completed!")


if __name__ == "__main__":
    main()
if __name__ == "__main__":
    main()
