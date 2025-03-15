"""
Getting Started with Agents Manager

This example demonstrates the most basic usage of the agents_manager library.
It shows how to create a simple agent and have a conversation with it.
"""

import os
from dotenv import load_dotenv
from agents_manager.AgentZero import AgentZero
from agents_manager.Agent import Agent
from agents_manager.models.OpenAi import OpenAi

# Load environment variables from .env file
load_dotenv()


def main():
    print("=== Getting Started with Agents Manager ===")

    # Step 1: Create a model
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment variables.")
        print("Please create a .env file with your API key.")
        return

    model = OpenAi(name="gpt-3.5-turbo", api_key=api_key)

    # Step 2: Create an agent
    assistant = Agent(
        name="assistant",
        instruction="You are a helpful assistant that provides clear and concise answers.",
        model=model,
    )

    # Step 3: Create AgentZero (the entry point)
    agent_zero = AgentZero()

    # Step 4: Add the agent to AgentZero
    agent_zero.add_agent(assistant)

    # Step 5: Have a conversation
    print("\nYou can now chat with the assistant. Type 'exit' to quit.\n")

    while True:
        # Get user input
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Goodbye!")
            break

        # Get response from the agent
        print("\nAssistant: ", end="")

        # Stream the response for a better user experience
        full_response = ""
        for chunk in agent_zero.run_agent_stream("assistant", user_input):
            if "content" in chunk and chunk["content"]:
                print(chunk["content"], end="", flush=True)
                full_response += chunk["content"]
        print("\n")


if __name__ == "__main__":
    main()
