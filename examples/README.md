# Agents Manager Examples

This directory contains example scenarios demonstrating how to use the Agents Manager library for various AI agent applications. These examples range from basic usage to advanced multi-agent workflows.

## Directory Structure

```
examples/
├── basic/
│   ├── simple_conversation.py - Basic interaction with different models
│   ├── tool_usage.py - Using Python functions as tools
│   └── model_comparison.py - Compare responses across different models
├── intermediate/
│   ├── multi_agent_collaboration.py - Agents working together
│   ├── container_tools.py - Using Docker containers as tools
│   └── streaming_responses.py - Working with streaming outputs
├── advanced/
│   ├── research_workflow.py - Multi-step research process
│   ├── code_generation.py - From requirements to code
│   └── data_processing.py - ETL with agents
└── README.md - This file
```

## Prerequisites

Before running the examples, make sure you have:

1. Installed the Agents Manager library:

   ```bash
   pip install agents-manager
   ```

2. Set up API keys for the LLM providers you want to use. Create a `.env` file in the examples directory with your API keys:

   ```
   OPENAI_API_KEY=your_openai_key_here
   ANTHROPIC_API_KEY=your_anthropic_key_here
   GOOGLE_API_KEY=your_google_key_here
   ```

3. For container examples, ensure Docker is installed and running on your system.

## Model-Specific Setup Instructions

### OpenAI

1. **Get an API Key**:

   - Go to [OpenAI API Keys](https://platform.openai.com/api-keys)
   - Create a new API key
   - Add the key to your `.env` file as `OPENAI_API_KEY`

2. **Model Selection**:

   - The examples use "gpt-4o" by default
   - Other options include "gpt-4-turbo", "gpt-3.5-turbo", etc.
   - Different models have different capabilities and pricing

3. **Required Parameters**:
   - `name`: The model name (e.g., "gpt-4o")
   - `api_key`: Your OpenAI API key

### Anthropic

1. **Get an API Key**:

   - Go to [Anthropic Console](https://console.anthropic.com/)
   - Create an account and generate an API key
   - Add the key to your `.env` file as `ANTHROPIC_API_KEY`

2. **Model Selection**:

   - The examples use "claude-3-5-sonnet-20241022" by default
   - Other options include "claude-3-opus", "claude-3-haiku", etc.
   - Check [Anthropic's documentation](https://docs.anthropic.com/claude/reference/selecting-a-model) for the latest models

3. **Required Parameters**:
   - `name`: The model name (e.g., "claude-3-5-sonnet-20241022")
   - `api_key`: Your Anthropic API key
   - `max_tokens`: Maximum number of tokens to generate (required by Anthropic)

### Google Genai

1. **Get an API Key**:

   - Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create an API key
   - Add the key to your `.env` file as `GOOGLE_API_KEY`

2. **Enable the API**:

   - Visit the [Google Cloud Console](https://console.cloud.google.com/)
   - Navigate to "APIs & Services" > "Library"
   - Search for "Generative Language API" and enable it
   - If you get a PERMISSION_DENIED error, follow the URL in the error message to enable the API

3. **Model Selection**:

   - The examples use "gemini-2.0-flash" by default
   - Other options include "gemini-2.0-pro", "gemini-1.5-pro", etc.

4. **Required Parameters**:
   - `name`: The model name (e.g., "gemini-2.0-flash")
   - `api_key`: Your Google API key

## Running the Examples

### Basic Examples

#### Simple Conversation

This example demonstrates how to create agents with different model backends and have a simple conversation with each.

```bash
cd examples/basic
python simple_conversation.py
```

Expected output:

```
Model(name='gpt-4o', type=OpenAi)
Agent(name='openai_assistant', model=OpenAi, tools=0)

Asking OpenAI agent...

=== OpenAI Response ===
[OpenAI's response to the question]

Model(name='claude-3-5-sonnet-20241022', type=Anthropic)
Agent(name='claude_assistant', model=Anthropic, tools=0)

Asking Anthropic agent...

=== Anthropic Response ===
[Anthropic's response to the question]
```

#### Tool Usage

This example shows how to create and use function-based tools with an agent.

```bash
cd examples/basic
python tool_usage.py
```

#### Model Comparison

This example runs the same prompt across different models and compares the results.

```bash
cd examples/basic
python model_comparison.py
```

## Troubleshooting

If you encounter issues running the examples:

### OpenAI Issues

- **Authentication Error**: Verify your API key is correct and has not expired
- **Rate Limits**: OpenAI has rate limits based on your account tier
- **Model Access**: Some models may require specific access permissions

### Anthropic Issues

- **Missing max_tokens**: Anthropic requires the `max_tokens` parameter
- **Authentication Error**: Verify your API key is correct
- **Rate Limits**: Anthropic has rate limits based on your account tier

### Google Genai Issues

- **PERMISSION_DENIED Error**: You need to enable the Generative Language API in your Google Cloud project
- **Invalid API Key**: Ensure you're using an API key from Google AI Studio, not a regular Google Cloud API key
- **Project Configuration**: Make sure your project is properly set up in Google Cloud

### General Issues

- **API Key Errors**: Ensure your API keys are correctly set in the `.env` file
- **Model Availability**: Some models may require specific access or subscriptions
- **Docker Issues**: For container examples, ensure Docker is running and you have sufficient permissions
- **Rate Limits**: If you hit rate limits, add delays between API calls or use a different API key

## Contributing

If you create an interesting example using Agents Manager, consider contributing it back to the repository!

## License

These examples are provided under the same license as the Agents Manager library.
