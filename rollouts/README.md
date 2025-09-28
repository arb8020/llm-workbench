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

### Type-Safe Environment Access

After running your agent, access environment-specific data using type narrowing:

```python
# For calculator environments
final_state = states[-1]
if isinstance(final_state.environment, CalculatorEnvironment):
    print(f"Result: {final_state.environment.current_value}")

# For search environments wrapping calculator
if hasattr(final_state.environment, 'inner_env') and \
   isinstance(final_state.environment.inner_env, CalculatorEnvironment):
    print(f"Result: {final_state.environment.inner_env.current_value}")
```

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

### When to use SearchEnvironment
```python
from rollouts import SearchEnvironment, create_search_config
calc_env = CalculatorEnvironment()
search_config = create_search_config(max_depth=2)
environment = SearchEnvironment(calc_env, search_config)
```
- Complex problem decomposition
- Multi-step reasoning tasks
- Exploring alternative solution paths

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

### Protocol-Based Environments

Environments use Python Protocols for composition over inheritance:

```python
from typing import Protocol
from dataclasses import dataclass

# Define your custom environment
@dataclass
class MyCustomEnvironment:
    """Custom environment following the Environment Protocol."""
    my_state: str = "initial"

    def get_tools(self) -> List[Tool]:
        return [my_custom_tool()]

    async def exec_tool(self, tool_call, state, config, store=None):
        # Handle your custom tools
        pass

    # ... implement other Protocol methods

# Use with type-safe access
final_env = final_state.environment
if isinstance(final_env, MyCustomEnvironment):
    print(f"My state: {final_env.my_state}")  # ✅ Type-safe
```

This design makes it easy to:
- Checkpoint and resume conversations
- Test with different AI models
- Add/remove tools via environments
- Handle complex multi-turn interactions
- **Create custom environments without inheritance**
- **Compose environments together (SearchEnvironment wrapping others)**
- **Access environment state in a type-safe way**