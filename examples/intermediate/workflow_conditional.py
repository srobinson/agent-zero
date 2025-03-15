import os

from dotenv import load_dotenv

from agentflow.Agent import Agent
from main import AgentZero
from models.OpenAi import OpenAi

# Load environment variables from .env file
load_dotenv()


def create_classifier_agent():
    """Create an agent that classifies user queries by topic."""
    api_key = os.getenv("OPENAI_API_KEY")
    model = OpenAi(name="gpt-4o", api_key=api_key)

    agent = Agent(
        name="classifier_agent",
        instruction="""You are a query classifier. Your job is to determine the primary category 
        of the user's question. Respond with ONLY ONE of these categories:
        - TECHNICAL (programming, software, hardware questions)
        - BUSINESS (strategy, marketing, finance questions)
        - CREATIVE (writing, design, art questions)
        - GENERAL (anything else)
        
        Respond with ONLY the category name, nothing else.""",
        model=model,
    )

    print(f"Created Classifier Agent")
    return agent


def create_technical_agent():
    """Create an agent specialized in technical topics."""
    api_key = os.getenv("OPENAI_API_KEY")
    model = OpenAi(name="gpt-4o", api_key=api_key)

    agent = Agent(
        name="technical_agent",
        instruction="""You are a technical expert specializing in programming, software development,
        and hardware. Provide detailed, accurate technical information with code examples when appropriate.""",
        model=model,
    )

    print(f"Created Technical Agent")
    return agent


def create_business_agent():
    """Create an agent specialized in business topics."""
    api_key = os.getenv("OPENAI_API_KEY")
    model = OpenAi(name="gpt-4o", api_key=api_key)

    agent = Agent(
        name="business_agent",
        instruction="""You are a business consultant with expertise in strategy, marketing, and finance.
        Provide practical, actionable business advice with real-world examples.""",
        model=model,
    )

    print(f"Created Business Agent")
    return agent


def create_creative_agent():
    """Create an agent specialized in creative topics."""
    api_key = os.getenv("OPENAI_API_KEY")
    model = OpenAi(name="gpt-4o", api_key=api_key)

    agent = Agent(
        name="creative_agent",
        instruction="""You are a creative professional with expertise in writing, design, and art.
        Provide imaginative, inspiring guidance with a focus on creative expression.""",
        model=model,
    )

    print(f"Created Creative Agent")
    return agent


def create_general_agent():
    """Create an agent for general knowledge questions."""
    api_key = os.getenv("OPENAI_API_KEY")
    model = OpenAi(name="gpt-4o", api_key=api_key)

    agent = Agent(
        name="general_agent",
        instruction="""You are a general knowledge assistant. Provide helpful, accurate information
        on a wide range of topics in a conversational, friendly manner.""",
        model=model,
    )

    print(f"Created General Agent")
    return agent


def is_technical(response):
    """Check if the response indicates a technical query."""
    return "TECHNICAL" in response["content"].upper()


def is_business(response):
    """Check if the response indicates a business query."""
    return "BUSINESS" in response["content"].upper()


def is_creative(response):
    """Check if the response indicates a creative query."""
    return "CREATIVE" in response["content"].upper()


def main():
    print("=== Conditional Workflow Example ===")
    print(
        "This example demonstrates how to create a workflow with conditional branching based on query classification.\n"
    )

    # Create the AgentZero instance
    agent_zero = AgentZero()

    # Create all agents
    classifier_agent = create_classifier_agent()
    technical_agent = create_technical_agent()
    business_agent = create_business_agent()
    creative_agent = create_creative_agent()
    general_agent = create_general_agent()

    # Add all agents to the manager
    agent_zero.add_agent(classifier_agent)
    agent_zero.add_agent(technical_agent)
    agent_zero.add_agent(business_agent)
    agent_zero.add_agent(creative_agent)
    agent_zero.add_agent(general_agent)

    # Get user query
    user_query = input("Enter your question: ")

    # Run the workflow manually since we don't have the workflow API
    print("\nProcessing your question...\n")

    # Step 1: Classify the query
    classification_result = agent_zero.run_agent("classifier_agent", user_query)

    # Step 2: Route to the appropriate specialist based on classification
    specialist_agent = "general_agent"  # Default

    if is_technical(classification_result):
        specialist_agent = "technical_agent"
    elif is_business(classification_result):
        specialist_agent = "business_agent"
    elif is_creative(classification_result):
        specialist_agent = "creative_agent"

    # Step 3: Get response from the specialist
    specialist_response = agent_zero.run_agent(specialist_agent, user_query)

    # Display the results
    print(f"\n=== Query Classification ===")
    print(f"Your question was classified as: {classification_result['content']}")

    print(
        f"\n=== {specialist_agent.replace('_agent', '').capitalize()} Specialist Response ==="
    )
    print(specialist_response["content"])

    print("\nWorkflow completed!")


if __name__ == "__main__":
    main()
    main()
    main()
