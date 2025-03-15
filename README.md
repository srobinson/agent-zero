# Agents Manager

[![PyPI version](https://badge.fury.io/py/agents-manager.svg)](https://badge.fury.io/py/agents-manager)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python Version](https://img.shields.io/badge/python-3.7%2B-blue)
[![Downloads](https://img.shields.io/pypi/dm/agents-manager.svg)](https://pypi.org/project/agents-manager/)

A lightweight Python package for managing multi-agent orchestration. Easily define agents with custom instructions, tools, and models, and orchestrate their interactions seamlessly. Perfect for building modular, collaborative AI systems.

## Features

- Define agents with specific roles and instructions
- Assign models to agents (e.g., OpenAI models)
- Equip agents with tools and containers for performing tasks
- Seamlessly orchestrate interactions between multiple agents

## Supported Models

- OpenAI
- Grok
- DeepSeek
- Anthropic
- Llama
- Genai

## Installation

Install the package via pip:

```sh
pip install agents-manager
```

## Quick Start

```python
from agents_manager import Agent, AgentManager
from agents_manager.models import OpenAi, Anthropic, Genai

from dotenv import load_dotenv

load_dotenv()

# Define the OpenAi model
openaiModel = OpenAi(name="gpt-4o-mini")

#Define the Anthropic model
anthropicModel = Anthropic(
        name="claude-3-5-sonnet-20241022",
        max_tokens= 1024,
        stream=True,
    )

#Define the Genai model
genaiModel = Genai(name="gemini-2.0-flash-001")

def multiply(a: int, b: int) -> int:
    """
    Multiply two numbers.
    """
    return a * b


def transfer_to_agent_3_for_math_calculation() -> Agent:
    """
    Transfer to agent 3 for math calculation.
    """
    return agent3


def transfer_to_agent_2_for_math_calculation() -> Agent:
    """
    Transfer to agent 2 for math calculation.    
    """
    agent2.set_instruction("You can change the instruction here")
    return agent2

# Define agents
agent3 = Agent(
    name="agent3",
    instruction="You are a maths teacher, explain properly how you calculated the answer.",
    model=genaiModel,
    tools=[multiply]
)

agent2 = Agent(
    name="agent2",
    instruction="You are a maths calculator bro",
    model=anthropicModel,
    tools=[transfer_to_agent_3_for_math_calculation]
)

agent1 = Agent(
    name="agent1",
    instruction="You are a helpful assistant",
    model=openaiModel,
    tools=[transfer_to_agent_2_for_math_calculation]
)

# Initialize Agent Manager and run agent
agent_manager = AgentManager()
agent_manager.add_agent(agent1)

response = agent_manager.run_agent("agent1", "What is 2 multiplied by 3?")
print(response["content"])
```

You can run for stream response as well.
```python
response_stream = agent_manager.run_agent_stream("agent1", [
    {"role": "user", "content": "What is 2 multiplied by 3?"},
])
for chunk in response_stream:
    print(chunk["content"], end="")
```

You can also pass container as tool to the agent.
```python
from agents_manager import Agent, AgentManager, Container

...

agent4 = Agent(
    name="agent4",
    instruction="You are a helpful assistant",
    model=model,
    tools=[Container(
        name="hello",
        description="A simple hello world container",
        image="hello-world:latest",
    )]
)
```

You can also pass the result of the container to the next agent with result variable.
```python
from agents_manager import Agent, Container

...

agent5 = Agent(
    name="agent1",
    instruction="You are a helpful assistant",
    model=model,
    tools=[Container(
        name="processing",
        description="Container to do some processing...",
        image="docker/xxxx:latest",
        environment=[
            {"name": "input1", "type": "integer"},
            {"name": "input2", "type": "integer"}
        ],
        authenticate={
            "username": "xxxxx",
            "password": "xxxxx",
            "registry": "xxxxx"
        },
        return_to={
            "agent": agent6,
            "instruction": "The result is: {result}"
        },
    )]
)
```


You can also run the agent with a dictionary as the input content.
```python

response = agent_manager.run_agent("agent1", {"role": "user", "content": "What is 2 multiplied by 3?"})

```

You can also run the agent with a list of history of messages as the input.
```python
response = agent_manager.run_agent("agent1", [
    {"role": "user", "content": "What is 2 multiplied by 3?"},
])
```



## More models
```python
from agents_manager.models import Grok, DeepSeek, Llama

#Define the Grok model
modelGrok = Grok(name="grok-2-latest")


#Define the DeepSeek model
modelDeepSeek = DeepSeek(name="deepseek-chat")


#Define the Llama model
modelLlama = Llama(name="llama3.1-70b")

```


## Troubleshooting

1. While using Genai model with functions, if you get the following error:

```python
google.genai.errors.ClientError: 400 INVALID_ARGUMENT. {'error': {'code': 400, 'message': '* GenerateContentRequest.tools[0].function_declarations[0].parameters.properties: should be non-empty for OBJECT type\n', 'status': 'INVALID_ARGUMENT'}}

```
It is because google genai does not support functions without parameters. You can fix this by providing a dummy parameter. Please let me know if you have a better solution for this. 

2. If you get the following error while running the container tool:
```python
Error: Error while fetching server API version: ('Connection aborted.', FileNotFoundError(2, 'No such file or directory'))
```

It is because the docker daemon is not running. You can fix this by starting the docker daemon.
and export the following environment variable:

```bash
#linux
export DOCKER_HOST=unix:///var/run/docker.sock

#colima
export DOCKER_HOST=unix://$HOME/.colima/default/docker.sock
```


## How It Works

1. **Define Agents**: Each agent has a name, a specific role (instruction), and a model.
2. **Assign Tools**: Agents can be assigned tools (functions and containers) to perform tasks.
3. **Create an Agent Manager**: The `AgentManager` manages the orchestration of agents.
4. **Run an Agent**: Start an agent to process a request and interact with other agents as needed.




## Use Cases

- AI-powered automation systems
- Multi-agent chatbots
- Complex workflow orchestration
- Research on AI agent collaboration

## Contributing

Contributions are welcome! Feel free to submit issues and pull requests.

## License

MIT License

