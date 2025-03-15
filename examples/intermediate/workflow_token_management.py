import os
import sys

from dotenv import load_dotenv

from agentflow.Agent import Agent
from agentflow.utils import estimate_tokens
from main import AgentZero
from models.OpenAi import OpenAi

# Load environment variables from .env file
load_dotenv()


# Custom summarization function that ensures complete output
def complete_summarize(text, max_tokens, model):
    """
    Summarize text to fit within token limit, ensuring the summary is complete.
    """
    from agents_manager.utils import estimate_tokens

    # If text already fits, return it unchanged
    if estimate_tokens(text) <= max_tokens:
        return text

    # Create a summarization agent with explicit instructions for completeness
    summarizer = Agent(
        name="summarizer",
        instruction=f"""You are a summarization specialist. Summarize the following text to fit within {max_tokens} tokens.
        
        IMPORTANT REQUIREMENTS:
        1. Your summary MUST be complete with no trailing ellipses or unfinished sentences
        2. Your summary MUST have a proper conclusion
        3. Your summary MUST preserve the most important information
        4. Your summary MUST be coherent and readable
        
        Ensure your summary is a complete, self-contained piece of text.
        """,
        model=model,
    )

    # Set the text to summarize
    summarizer.set_system_message(summarizer.instruction)
    summarizer.set_user_message(text)

    # Get the summary
    response = summarizer.get_response()
    summary = response.get("content", "")

    # Print the summarization result
    print(f"\nSummarization Result:")
    print(f"Original length: {len(text)} characters ({estimate_tokens(text)} tokens)")
    print(
        f"Summary length: {len(summary)} characters ({estimate_tokens(summary)} tokens)"
    )
    print(f"Reduction: {100 - (len(summary) / len(text) * 100):.1f}%")

    return summary


def main():
    print("=== Token Management Workflow Example ===")
    print(
        "This example demonstrates how to manage token limits between agents with different context sizes.\n"
    )

    # Create the AgentZero instance
    agent_zero = AgentZero()

    # Create agents with different context sizes
    large_context_agent = Agent(
        name="large_context_agent",
        instruction="""You are a research specialist. Provide comprehensive, detailed information on the given topic.
        
        IMPORTANT: Your response MUST be complete with a proper conclusion. Never end with ellipses or incomplete thoughts.
        """,
        model=OpenAi(
            name="gpt-4o",
            api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=8192,  # Large context
        ),
    )

    small_context_agent = Agent(
        name="small_context_agent",
        instruction="""You are a summarization specialist. Create concise, informative summaries that capture the key points.
        
        IMPORTANT: Your summary MUST be complete with a proper conclusion. Never end with ellipses or incomplete thoughts.
        """,
        model=OpenAi(
            name="gpt-3.5-turbo",
            api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=2048,  # Small context
        ),
    )

    final_agent = Agent(
        name="final_agent",
        instruction="""You are a content specialist. Create engaging, complete content from the provided summary.
        
        IMPORTANT REQUIREMENTS:
        1. Your content MUST have a clear beginning, middle, and end
        2. Your content MUST be complete with a proper conclusion
        3. NEVER end with ellipses or incomplete thoughts
        4. Your content should be well-structured and engaging
        """,
        model=OpenAi(
            name="gpt-3.5-turbo",
            api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=4096,  # Medium context
        ),
    )

    # Add agents to the manager
    agent_zero.add_agent(large_context_agent)
    agent_zero.add_agent(small_context_agent)
    agent_zero.add_agent(final_agent)

    # Define a custom function to run the workflow with streaming
    def run_workflow_with_streaming():
        # Create a dictionary to store results
        results = {}

        # Step 1: Research
        print("\nProcessing step: research")

        # Get the topic
        topic = "The history and future of artificial intelligence"

        # Get the research agent
        current_input = f"Research this topic in detail: {topic}. Provide a comprehensive overview with historical context, current state, and future prospects."

        print(f"\n{'=' * 30} STREAMING RESEARCH {'=' * 30}\n")

        # Stream the response
        research_response = ""
        for chunk in agent_zero.run_agent_stream("large_context_agent", current_input):
            if chunk.get("content"):
                content = chunk["content"]
                research_response += content
                sys.stdout.write(content)
                sys.stdout.flush()

        print(f"\n\n{'=' * 80}\n")
        results["research"] = {"content": research_response}
        print(f"Completed step: research")

        # Step 2: Summary
        print("\nProcessing step: summary")

        # Get the summary agent
        current_input = research_response

        # Check if we need to manage tokens
        _, agent = agent_zero.get_agent("small_context_agent")
        max_input_tokens = agent.model.kwargs.get("max_tokens", 2048) // 2
        print(f"Managing tokens for input (limit: {max_input_tokens})")

        # Use our custom summarization function
        if estimate_tokens(current_input) > max_input_tokens:
            print(f"Input exceeds token limit. Summarizing...")
            current_input = complete_summarize(
                current_input, max_input_tokens, large_context_agent.model
            )

        print(f"\n{'=' * 30} STREAMING SUMMARY {'=' * 30}\n")

        # Stream the response
        summary_response = ""
        for chunk in agent_zero.run_agent_stream(
            "small_context_agent",
            f"Create a concise summary of the following text:\n\n{current_input}",
        ):
            if chunk.get("content"):
                content = chunk["content"]
                summary_response += content
                sys.stdout.write(content)
                sys.stdout.flush()

        print(f"\n\n{'=' * 80}\n")
        results["summary"] = {"content": summary_response}
        print(f"Completed step: summary")

        # Step 3: Content
        print("\nProcessing step: content")

        # Get the content agent
        current_input = summary_response

        # Check if we need to manage tokens
        _, agent = agent_zero.get_agent("final_agent")
        max_input_tokens = agent.model.kwargs.get("max_tokens", 4096) // 2
        print(f"Managing tokens for input (limit: {max_input_tokens})")

        # Use our custom summarization function if needed
        if estimate_tokens(current_input) > max_input_tokens:
            print(f"Input exceeds token limit. Summarizing...")
            current_input = complete_summarize(
                current_input, max_input_tokens, large_context_agent.model
            )

        print(f"\n{'=' * 30} STREAMING CONTENT {'=' * 30}\n")

        # Stream the response
        content_response = ""
        for chunk in agent_zero.run_agent_stream(
            "final_agent",
            f"Create engaging, complete content based on this summary:\n\n{current_input}",
        ):
            if chunk.get("content"):
                content = chunk["content"]
                content_response += content
                sys.stdout.write(content)
                sys.stdout.flush()

        print(f"\n\n{'=' * 80}\n")
        results["content"] = {"content": content_response}
        print(f"Completed step: content")

        return results

    # Run workflow with token management
    results = run_workflow_with_streaming()

    # Print the final results
    print("\n\nWorkflow Results:\n")
    for step_name, result in results.items():
        print(f"\n--- {step_name.upper()} ---")
        content = result.get("content", "")
        print(
            f"Content length: {len(content)} characters ({estimate_tokens(content)} tokens)"
        )
        print(content[:1024] + ("..." if len(content) > 1024 else ""))


if __name__ == "__main__":
    main()
if __name__ == "__main__":
    main()
