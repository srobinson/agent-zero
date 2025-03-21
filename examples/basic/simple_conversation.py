import os

from dotenv import load_dotenv

from agentflow.Agent import Agent
from main import AgentZero
from models.OpenAi import OpenAi

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

        try:
            # For the first message, just use the standard run_agent
            if i == 0:
                # Get the response
                response = agent_zero.run_agent(agent_name, user_message)

                # Print the assistant's response
                print(f"Assistant: {response['content']}")

                # Add the user message and assistant's response to the history
                full_history.append({"role": "user", "content": user_message})
                full_history.append(
                    {"role": "assistant", "content": response["content"]}
                )
            else:
                # For subsequent messages, we need to maintain the conversation history
                # Add the new user message to history
                full_history.append({"role": "user", "content": user_message})

                # Set the complete message history on the agent
                agent.set_messages(
                    [{"role": "system", "content": agent.instruction}] + full_history
                )

                # Get the response (without passing user_message since it's already in history)
                response = agent_zero.run_agent(agent_name)

                # Print the assistant's response
                print(f"Assistant: {response['content']}")

                # Add the assistant's response to the history
                full_history.append(
                    {"role": "assistant", "content": response["content"]}
                )

        except Exception as e:
            print(f"Error in conversation: {e}")
            break

    print("\nConversation completed!")


def main():
    print("=== Simple Conversation Example ===")
    print(
        "This example demonstrates a basic back-and-forth conversation with an agent.\n"
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
if __name__ == "__main__":
    main()
