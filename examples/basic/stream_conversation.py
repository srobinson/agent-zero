import os
import sys
import time
from dotenv import load_dotenv
from agents_manager.AgentZero import AgentZero
from agents_manager.Agent import Agent
from agents_manager.models.OpenAi import OpenAi

# Load environment variables from .env file
load_dotenv()


def create_conversation_agent():
    """Create a simple conversation agent using OpenAI."""
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Error: OPENAI_API_KEY not found in environment variables.")
            return None

        model = OpenAi(name="gpt-4o", api_key=api_key)

        agent = Agent(
            name="conversation_assistant",
            instruction="You are a friendly assistant named Alex. You have a warm, conversational tone and enjoy helping users. You remember details from earlier in the conversation and can refer back to them. Keep your responses concise but helpful.",
            model=model,
        )

        print(f"Model(name='{model.name}', type={model.__class__.__name__})")
        print(
            f"Agent(name='{agent.name}', model={model.__class__.__name__}, tools={len(agent.tools)})"
        )

        return agent
    except Exception as e:
        print(f"Error creating conversation agent: {e}")
        return None


def simulate_conversation(agent_zero, agent_name):
    """Simulate a conversation with the agent."""
    conversation = [
        "Hi there! My name is Jamie. Can you help me plan a weekend trip?",
        "I'm thinking about going to either the mountains or the beach. Which would you recommend for relaxation?",
        "The beach sounds perfect. I'm traveling with my dog. Are there usually pet-friendly accommodations?",
        "Great! I'll need some activities too. What are some relaxing things to do at the beach besides swimming?",
        "Those all sound wonderful. One last question - what should I pack for a beach weekend?",
    ]

    # Keep track of the full conversation history
    full_history = []

    for i, user_message in enumerate(conversation):
        print(f"\n--- Turn {i+1} ---")
        # Output agent and model information for each turn
        _, agent = agent_zero.get_agent(agent_name)
        model_name = agent.get_model().name
        model_type = agent.get_model().__class__.__name__
        print(f"[Agent: {agent.name} | Model: {model_name} ({model_type})]")
        print(f"User: {user_message}")
        print(f"Assistant: ", end="")

        try:
            # For the first message, just use the standard run_agent_stream
            # This will automatically set the user message
            if i == 0:
                # Get the streaming response
                accumulated_response = ""
                for chunk in agent_zero.run_agent_stream(agent_name, user_message):
                    if "content" in chunk and chunk["content"]:
                        print(chunk["content"], end="", flush=True)
                        accumulated_response += chunk["content"]

                # Add a newline after the streaming response
                print()

                # Add the user message and assistant's response to the history
                full_history.append({"role": "user", "content": user_message})
                full_history.append(
                    {"role": "assistant", "content": accumulated_response}
                )
            else:
                # For subsequent messages, we need to maintain the conversation history
                # Add the new user message to history
                full_history.append({"role": "user", "content": user_message})

                # Set the complete message history on the agent
                agent.set_messages(
                    [{"role": "system", "content": agent.instruction}] + full_history
                )

                # Get the streaming response (without passing user_message since it's already in history)
                accumulated_response = ""
                for chunk in agent_zero.run_agent_stream(agent_name):
                    if "content" in chunk and chunk["content"]:
                        print(chunk["content"], end="", flush=True)
                        accumulated_response += chunk["content"]

                # Add a newline after the streaming response
                print()

                # Add the assistant's response to the history
                full_history.append(
                    {"role": "assistant", "content": accumulated_response}
                )

        except Exception as e:
            print(f"Error in conversation: {e}")
            break

    print("\nConversation completed!")


def main():
    print("=== Simple Streaming Conversation Example ===")
    print(
        "This example demonstrates a basic back-and-forth conversation with an agent using streaming responses.\n"
    )

    # Create AgentZero instance
    agent_zero = AgentZero()

    # Create the conversation agent
    agent = create_conversation_agent()

    if not agent:
        print("Failed to create agent. Exiting.")
        return

    # Add the agent to AgentZero
    agent_zero.add_agent(agent)

    # Simulate a conversation
    simulate_conversation(agent_zero, "conversation_assistant")


if __name__ == "__main__":
    main()
