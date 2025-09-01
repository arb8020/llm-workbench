# Rollouts

Clean agent framework for AI conversations and tool use.

## Quick Start

### Basic Text Conversations

For simple AI analysis tasks without tools:

```python
import asyncio
from rollouts import (
    Endpoint, Actor, AgentState, Message, Trajectory, RunConfig,
    BasicEnvironment, run_agent, stdout_handler
)

async def simple_analysis():
    # Create messages
    system_msg = Message(role="system", content="You are a helpful analyst.")
    user_msg = Message(role="user", content="Analyze this data...")
    
    # Set up AI endpoint
    endpoint = Endpoint(
        provider="anthropic",  # or "openai"
        model="claude-3-5-sonnet-20241022",
        api_key="your-api-key",
        temperature=0.1
    )
    
    # Create agent with BasicEnvironment (no tools)
    trajectory = Trajectory(messages=[system_msg, user_msg])
    actor = Actor(trajectory=trajectory, endpoint=endpoint)
    environment = BasicEnvironment()  # Clean, no distracting tools
    
    agent_state = AgentState(
        actor=actor,
        environment=environment,
        max_turns=1
    )
    
    # Run analysis
    run_config = RunConfig(on_chunk=stdout_handler)
    final_states = await run_agent(agent_state, run_config)
    
    return final_states[-1].actor.trajectory.messages[-1].content

# Run it
response = asyncio.run(simple_analysis())
print(response)
```

### Tool-Based Conversations

For AI that can use calculator tools:

```python
from rollouts import CalculatorEnvironment

# Replace BasicEnvironment with CalculatorEnvironment
environment = CalculatorEnvironment()
# Now AI can use add, subtract, multiply, divide, clear tools
```

## Environments

- **`BasicEnvironment`** - No tools, clean text conversations (recommended for analysis)
- **`CalculatorEnvironment`** - Provides calculator tools (add, subtract, multiply, etc.)
- **`SearchEnvironment`** - Advanced search and decomposition tools

## Examples

- **`examples/oracle_analysis.py`** - AI-powered code analysis using BasicEnvironment
- **`examples/simple_calculator.py`** - Calculator tool usage example

## Environment Usage Patterns

### When to use BasicEnvironment
```python
from rollouts import BasicEnvironment
environment = BasicEnvironment()
```
- Text analysis tasks
- Code reviews
- Document processing
- Single-shot AI conversations
- Any time you want clean responses without tool confusion

### When to use CalculatorEnvironment
```python
from rollouts import CalculatorEnvironment
environment = CalculatorEnvironment()
```
- Math problem solving
- Data calculations
- Financial computations
- Any task requiring arithmetic

The key insight: **Choose your environment based on whether you want the AI to have access to tools or just provide text responses.**

## Core API

```python
# Essential imports
from rollouts import (
    Endpoint,           # AI model configuration
    Actor,              # Combines messages + endpoint  
    AgentState,         # Combines actor + environment + settings
    Message,            # Individual conversation message
    Trajectory,         # List of messages
    RunConfig,          # Execution configuration
    BasicEnvironment,   # No-tools environment
    run_agent,          # Main execution function
    stdout_handler      # Simple output handler
)
```

## Architecture

Rollouts follows a clean functional architecture:

1. **Messages** → **Trajectory** (conversation history)
2. **Trajectory** + **Endpoint** → **Actor** (AI agent) 
3. **Actor** + **Environment** → **AgentState** (complete state)
4. **AgentState** + **RunConfig** → **run_agent()** (execution)

This design makes it easy to:
- Checkpoint and resume conversations
- Test with different AI models
- Add/remove tools via environments
- Handle complex multi-turn interactions