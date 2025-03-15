import os

from dotenv import load_dotenv

from agentflow.Agent import Agent
from agentflow.Container import Container
from main import AgentZero
from models.OpenAi import OpenAi

# Load environment variables from .env file
load_dotenv()


def create_container_tools():
    """Create container tools for different tasks."""

    # Python code execution container
    python_container = Container(
        name="run_python",
        description="Executes Python code and returns the result.",
        image="python:3.9-slim",
        command='python -c "$CODE"',
        environment=[
            {"name": "CODE", "type": "string", "description": "Python code to execute"}
        ],
    )

    # Data processing container
    data_processor = Container(
        name="process_data",
        description="Process CSV data using pandas and return statistics.",
        image="jupyter/scipy-notebook:latest",
        command="python -c \"import pandas as pd; import json; import sys; data = pd.read_csv('$DATA_PATH'); result = data.describe().to_json(); print(result)\"",
        environment=[
            {
                "name": "DATA_PATH",
                "type": "string",
                "description": "Path to CSV data file",
            }
        ],
        volumes={
            "/path/to/data": {
                "bind": "/data",
                "mode": "ro",
            }  # Mount local data directory
        },
    )

    # Web scraping container
    web_scraper = Container(
        name="scrape_webpage",
        description="Scrapes content from a webpage and returns the text.",
        image="python:3.9-slim",
        command="pip install requests beautifulsoup4 && python -c \"import requests; from bs4 import BeautifulSoup; url = '$URL'; response = requests.get(url); soup = BeautifulSoup(response.text, 'html.parser'); print(soup.get_text())\"",
        environment=[
            {
                "name": "URL",
                "type": "string",
                "description": "URL of the webpage to scrape",
            }
        ],
    )

    return [python_container, data_processor, web_scraper]


def create_developer_agent():
    """Create an agent that can use container tools."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Warning: OPENAI_API_KEY not found in environment variables.")
        return None

    model = OpenAi(name="gpt-4o", api_key=api_key)

    # Create container tools
    container_tools = create_container_tools()

    agent = Agent(
        name="developer_agent",
        instruction="""You are a developer assistant with access to Docker container tools.
        You can execute Python code, process data, and scrape web content using these tools.
        
        Available container tools:
        1. run_python - Executes Python code and returns the result
        2. process_data - Process CSV data using pandas and return statistics
        3. scrape_webpage - Scrapes content from a webpage and returns the text
        
        When asked to perform tasks related to code execution, data processing, or web scraping,
        use the appropriate container tool. Be specific in your tool usage and explain your approach.
        """,
        model=model,
        tools=container_tools,
    )

    print(f"Created Developer Agent with container tools")
    return agent


def main():
    print("=== Container Tools Example ===")
    print(
        "This example demonstrates how to use Docker containers as tools for agents.\n"
    )

    # Create the AgentZero instance
    agent_zero = AgentZero()

    # Create the developer agent with container tools
    developer_agent = create_developer_agent()
    if not developer_agent:
        print("Failed to create developer agent. Exiting.")
        return

    # Add the agent to AgentZero
    agent_zero.add_agent(developer_agent)

    # Example tasks that would use container tools
    tasks = [
        "Write a Python function to calculate the Fibonacci sequence and test it with n=10",
        "Process the sales_data.csv file and give me the statistical summary",
        "Scrape the content from https://example.com and summarize the main points",
    ]

    print("Example tasks that would use container tools:")
    for i, task in enumerate(tasks, 1):
        print(f"{i}. {task}")

    # Get user input for which task to run
    try:
        choice = int(input("\nEnter the number of the task you want to run (1-3): "))
        if choice < 1 or choice > len(tasks):
            print(f"Invalid choice. Using default task 1.")
            choice = 1
    except ValueError:
        print(f"Invalid input. Using default task 1.")
        choice = 1

    selected_task = tasks[choice - 1]
    print(f"\nRunning task: {selected_task}\n")

    # Stream the agent's response
    print("Agent response (streaming):")
    print("-" * 80)

    full_response = ""
    for chunk in agent_zero.run_agent_stream("developer_agent", selected_task):
        if "content" in chunk and chunk["content"]:
            print(chunk["content"], end="", flush=True)
            full_response += chunk["content"]

    print("\n" + "-" * 80)

    # Note about Docker requirements
    print(
        "\nNote: This example requires Docker to be installed and running on your system."
    )
    print(
        "The container tools are defined but actual execution depends on Docker availability."
    )
    print(
        "You may need to adjust volume paths and permissions based on your system configuration."
    )

    # Optional: Save the response to a file
    try:
        os.makedirs("output", exist_ok=True)
        with open("output/container_tool_response.txt", "w") as f:
            f.write(full_response)
        print("\nResponse saved to 'output/container_tool_response.txt'")
    except Exception as e:
        print(f"\nError saving response: {e}")


if __name__ == "__main__":
    main()
if __name__ == "__main__":
    main()
