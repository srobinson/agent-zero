import os

from dotenv import load_dotenv

from agentflow.Agent import Agent
from agentflow.Workflow import AgentStep, AgentWorkflow
from main import AgentZero
from models.OpenAi import OpenAi

# Load environment variables from .env file
load_dotenv()

# Create the AgentZero instance
agent_zero = AgentZero()

# Create models
api_key = os.getenv("OPENAI_API_KEY")
small_model = OpenAi(
    name="gpt-3.5-turbo", api_key=api_key, max_tokens=2048
)  # Reduced max_tokens
large_model = OpenAi(
    name="gpt-4", api_key=api_key, max_tokens=4096
)  # Reduced max_tokens

# Create agents with more specific instructions
research_agent = Agent(
    name="research_agent",
    instruction="""You are a research assistant. Your task is to gather comprehensive information on a given topic.
    When given a topic, provide detailed information including definitions, causes, effects, statistics, and current knowledge.
    DO NOT ask for clarification - assume the topic is clear and provide the best information you can.
    Always respond with substantive research content.
    Keep your response concise and under 1000 words.""",  # Added length constraint
    model=small_model,
)

analysis_agent = Agent(
    name="analysis_agent",
    instruction="""You are an analysis expert. Your task is to analyze information and provide insights.
    When given research information, identify patterns, implications, and draw meaningful conclusions.
    Focus on critical analysis rather than summarization.
    Always provide substantive analytical content.
    Keep your response concise and under 800 words.""",  # Added length constraint
    model=small_model,
)

summary_agent = Agent(
    name="summary_agent",
    instruction="""You are a summarization expert. Your task is to create concise summaries.
    When given analyzed information, distill it into key points that capture the essence of the topic.
    Create a clear, structured summary that is easy to understand.
    Always provide a complete, well-organized summary.
    Keep your summary under 500 words.""",  # Added length constraint
    model=large_model,
)

# Add agents to AgentZero
agent_zero.add_agent(research_agent)
agent_zero.add_agent(analysis_agent)
agent_zero.add_agent(summary_agent)

# Create workflow
research_workflow = AgentWorkflow(name="research_workflow")

# Create steps
research_step = AgentStep(
    research_agent, name="research", description="Research information"
)
analysis_step = AgentStep(
    analysis_agent, name="analysis", description="Analyze information"
)
summary_step = AgentStep(
    summary_agent, name="summary", description="Summarize findings"
)

# Add steps to workflow
research_workflow.add_step(research_step)
research_workflow.add_step(analysis_step)
research_workflow.add_step(summary_step)

# Define transitions
research_step.then(analysis_step)
analysis_step.then(summary_step)

# Set starting step
research_workflow.set_start_step(research_step)

print(f"Workflow created: {research_workflow}")
print(f"Steps: {list(research_workflow.steps.keys())}")
print(f"Starting step: {research_workflow.start_step.name}")


# Function to truncate text to avoid token limit issues
def truncate_text(text, max_chars=3000):
    """Truncate text to avoid token limit issues."""
    if len(text) > max_chars:
        return text[:max_chars] + "... [truncated for token limit]"
    return text


# Now we can run the workflow using AgentZero
def run_workflow(workflow, topic):
    """Run a workflow with AgentZero."""
    print(f"\nRunning workflow on topic: {topic}")

    # Get the starting step
    current_step_name = workflow.start_step.name
    results = {}

    # Manual step sequence since we're having issues with next_steps
    step_sequence = ["research", "analysis", "summary"]
    current_index = 0

    while current_index < len(step_sequence):
        current_step_name = step_sequence[current_index]
        print(f"\nExecuting step: {current_step_name}")

        # Get the current step object
        current_step = workflow.steps[current_step_name]

        # Prepare the input for the current step
        if current_step_name == "research":
            # First step gets the original topic
            input_text = f"Research this topic thoroughly: {topic}"
        else:
            # Subsequent steps get results from previous steps
            previous_step = step_sequence[current_index - 1]
            previous_result = results[previous_step]

            # Truncate previous result to avoid token limit issues
            truncated_result = truncate_text(previous_result)

            input_text = f"Based on the following information about {topic}:\n\n{truncated_result}\n\nPerform your task for the topic: {topic}"

        try:
            # Run the agent for this step
            agent_name = current_step.agent.name
            response = agent_zero.run_agent(agent_name, input_text)

            # Store the result
            results[current_step_name] = response["content"]

            # Print the result
            print(f"\nResult from {current_step_name}:")
            print(f"{'-' * 40}")
            print(
                response["content"][:500] + "..."
                if len(response["content"]) > 500
                else response["content"]
            )
            print(f"{'-' * 40}")

        except Exception as e:
            print(f"\nError in step {current_step_name}: {str(e)}")
            print("Attempting to continue with a simplified input...")

            # Fallback: Try with a much shorter input
            simplified_input = f"Regarding {topic}, please provide your {current_step_name} based on what you know about this topic."
            try:
                response = agent_zero.run_agent(agent_name, simplified_input)
                results[current_step_name] = response["content"]

                print(f"\nResult from {current_step_name} (fallback):")
                print(f"{'-' * 40}")
                print(
                    response["content"][:500] + "..."
                    if len(response["content"]) > 500
                    else response["content"]
                )
                print(f"{'-' * 40}")
            except Exception as e2:
                print(f"Fallback also failed: {str(e2)}")
                results[current_step_name] = (
                    f"Error: Could not complete {current_step_name} step due to token limits."
                )

        # Move to the next step in our manual sequence
        current_index += 1

    print("\nWorkflow completed!")
    return results


# Example usage
if __name__ == "__main__":
    topic = input("Enter a topic for the research workflow: ")
    results = run_workflow(research_workflow, topic)
    topic = input("Enter a topic for the research workflow: ")
    results = run_workflow(research_workflow, topic)
