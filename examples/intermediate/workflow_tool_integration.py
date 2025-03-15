import os
import json
import requests
from dotenv import load_dotenv
from agents_manager.AgentZero import AgentZero
from agents_manager.Agent import Agent
from agents_manager.models.OpenAi import OpenAi

# Load environment variables from .env file
load_dotenv()


def search_wikipedia(query: str) -> str:
    """
    Search Wikipedia for information about a topic.

    Args:
        query (str): The topic to search for

    Returns:
        str: A summary of the Wikipedia article
    """
    try:
        # Format the query for the Wikipedia API
        search_url = "https://en.wikipedia.org/w/api.php"
        search_params = {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": query,
        }

        # Get search results
        search_response = requests.get(search_url, params=search_params)
        search_data = search_response.json()

        if not search_data.get("query", {}).get("search"):
            return f"No Wikipedia articles found for '{query}'."

        # Get the page ID of the first result
        page_id = search_data["query"]["search"][0]["pageid"]

        # Get the content of the page
        content_params = {
            "action": "query",
            "format": "json",
            "prop": "extracts",
            "exintro": True,
            "explaintext": True,
            "pageids": page_id,
        }

        content_response = requests.get(search_url, params=content_params)
        content_data = content_response.json()

        # Extract the summary
        page_content = content_data["query"]["pages"][str(page_id)]["extract"]
        page_title = content_data["query"]["pages"][str(page_id)]["title"]

        return f"Wikipedia: {page_title}\n\n{page_content}"

    except Exception as e:
        return f"Error searching Wikipedia: {str(e)}"


def get_weather(location: str) -> str:
    """
    Get the current weather for a location.

    Args:
        location (str): The location to get weather for

    Returns:
        str: A description of the current weather
    """
    # This is a mock implementation since we don't have a real API key
    weather_conditions = ["sunny", "cloudy", "rainy", "snowy", "windy"]
    temperatures = range(0, 35)

    import random

    condition = random.choice(weather_conditions)
    temperature = random.choice(temperatures)

    return f"Weather for {location}: {condition.capitalize()} with a temperature of {temperature}°C"


def create_research_agent():
    """Create an agent that researches topics using tools."""
    api_key = os.getenv("OPENAI_API_KEY")
    model = OpenAi(name="gpt-4o", api_key=api_key)

    agent = Agent(
        name="research_agent",
        instruction="""You are a research assistant that MUST use tools to gather information.
        When asked about ANY topic, ALWAYS use the search_wikipedia tool to find relevant information.
        When asked about weather in ANY location, ALWAYS use the get_weather tool.
        NEVER respond with generic messages like "How can I assist you today?".
        ALWAYS provide detailed information from the tools.""",
        model=model,
        tools=[search_wikipedia, get_weather],
    )

    print(f"Created Research Agent with tools")
    return agent


def create_summary_agent():
    """Create an agent that summarizes research findings."""
    api_key = os.getenv("OPENAI_API_KEY")
    model = OpenAi(name="gpt-4o", api_key=api_key)

    agent = Agent(
        name="summary_agent",
        instruction="""You are a summarization specialist. Take detailed information and create
        a concise, easy-to-understand summary highlighting the most important points.""",
        model=model,
    )

    print(f"Created Summary Agent")
    return agent


def create_recommendation_agent():
    """Create an agent that provides recommendations based on research."""
    api_key = os.getenv("OPENAI_API_KEY")
    model = OpenAi(name="gpt-4o", api_key=api_key)

    agent = Agent(
        name="recommendation_agent",
        instruction="""You are a recommendation specialist. Based on the provided information,
        offer practical, actionable recommendations and next steps.""",
        model=model,
    )

    print(f"Created Recommendation Agent")
    return agent


def main():
    print("=== Tool Integration Workflow Example ===")
    print(
        "This example demonstrates how to integrate external tools into a workflow.\n"
    )

    # Create the AgentZero instance
    agent_zero = AgentZero()

    # Create the agents
    research_agent = create_research_agent()
    summary_agent = create_summary_agent()
    recommendation_agent = create_recommendation_agent()

    # Add agents to the manager
    agent_zero.add_agent(research_agent)
    agent_zero.add_agent(summary_agent)
    agent_zero.add_agent(recommendation_agent)

    # Get user input
    topic = input(
        "Enter a topic to research (e.g., 'climate change', 'artificial intelligence'): "
    )
    location = input(
        "Enter a location for weather information (e.g., 'New York', 'Tokyo'): "
    )

    # Run the workflow manually since we don't have the workflow API
    print(f"\nResearching '{topic}' and weather in '{location}'...\n")

    # Step 1: Research
    print(f"\n{'=' * 30} RESEARCH {'=' * 30}\n")

    # Formulate a query that will trigger tool use
    research_query = f"I need you to do two things: 1) Search Wikipedia for information about {topic}. 2) Get the current weather in {location}. Please use your tools to find this information."

    research_result = agent_zero.run_agent("research_agent", research_query)
    print(research_result["content"])

    # Step 2: Summarize
    print(f"\n{'=' * 30} SUMMARY {'=' * 30}\n")

    summary_query = f"Please summarize the following research information:\n\n{research_result['content']}"
    summary_result = agent_zero.run_agent("summary_agent", summary_query)
    print(summary_result["content"])

    # Step 3: Recommendations
    print(f"\n{'=' * 30} RECOMMENDATIONS {'=' * 30}\n")

    recommendation_query = f"""
    Based on the following information, provide practical recommendations and next steps:
    
    RESEARCH:
    {research_result['content']}
    
    SUMMARY:
    {summary_result['content']}
    """

    recommendation_result = agent_zero.run_agent(
        "recommendation_agent", recommendation_query
    )
    print(recommendation_result["content"])

    # Store results for reporting
    results = {
        "research": research_result,
        "summary": summary_result,
        "recommendation": recommendation_result,
    }

    print(f"\n{'=' * 80}\n")
    print("Workflow completed!")


if __name__ == "__main__":
    main()
