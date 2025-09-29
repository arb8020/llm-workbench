# Option A: Protocol + User Unions for Type-Safe Extensible Environments

## Overview

The library provides built-in environments as a tagged union for internal type safety, while exposing a `Protocol` for extensibility. Users can create custom environments that satisfy the protocol and optionally create their own tagged unions for type safety in their code.

## Library Code Structure

### Core Types (rollouts/dtypes.py)

```python
from typing import Protocol, Union, Dict, Any, runtime_checkable
from dataclasses import dataclass
from typing_extensions import Literal

# =======================
# Built-in Environment Types (Tagged Union)
# =======================

@dataclass
class CalculatorEnv:
    """Calculator environment with numeric state."""
    type: Literal["calculator"] = "calculator"
    current_value: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {"type": "calculator", "current_value": self.current_value}
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'CalculatorEnv':
        return CalculatorEnv(current_value=data["current_value"])

@dataclass
class BinarySearchEnv:
    """Binary search environment."""
    type: Literal["binary_search"] = "binary_search"
    range_min: int
    range_max: int
    answer: int
    turns: int = 0
    correct: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "binary_search",
            "range_min": self.range_min,
            "range_max": self.range_max,
            "answer": self.answer,
            "turns": self.turns,
            "correct": self.correct
        }
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'BinarySearchEnv':
        return BinarySearchEnv(**{k: v for k, v in data.items() if k != "type"})

@dataclass
class SearchWrapperEnv:
    """Wrapper that adds search to another environment."""
    type: Literal["search_wrapper"] = "search_wrapper"
    inner_env: 'Environment'  # Can wrap ANY environment via protocol
    depth: int
    search_config: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "search_wrapper",
            "inner_env": self.inner_env.to_dict(),
            "depth": self.depth,
            "search_config": self.search_config
        }

# Library's tagged union for built-in environments
BuiltinEnv = Union[CalculatorEnv, BinarySearchEnv, SearchWrapperEnv]

# =======================
# Protocol for ALL Environments
# =======================

@runtime_checkable
class Environment(Protocol):
    """Protocol that all environments must satisfy.
    
    This allows both built-in and user-defined environments
    to work with the rollout framework.
    """
    type: str  # Discriminator field for type narrowing
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize environment state to dictionary."""
        ...
    
    def get_tools(self) -> List[Tool]:
        """Return available tools for this environment."""
        return []
    
    def get_display_string(self) -> str:
        """Return human-readable state summary."""
        # Default implementation
        return f"{self.type}: {self.to_dict()}"
```

### Framework Functions (rollouts/runner.py)

```python
from typing import Any
from rollouts.dtypes import Environment, BuiltinEnv, CalculatorEnv, BinarySearchEnv

@dataclass
class RolloutResult:
    """Result of running a rollout."""
    environment: Environment  # Works with ANY environment
    transcript: str
    total_tokens: int
    
def run_rollout(env: Environment, agent: Agent) -> RolloutResult:
    """Run a rollout with any environment satisfying the protocol.
    
    Args:
        env: Any environment implementing the Environment protocol
        agent: Agent to run
        
    Returns:
        RolloutResult with final environment state
    """
    # Framework doesn't need to know specific environment type
    tools = env.get_tools()
    # ... run the rollout ...
    return RolloutResult(environment=env, transcript="...", total_tokens=100)

def display_environment(env: Environment) -> str:
    """Display any environment, with special handling for built-ins.
    
    This shows how the library can still have type-safe handling
    of its own types while accepting any environment.
    """
    # Type narrowing for built-in types - fully type safe!
    if isinstance(env, CalculatorEnv):
        return f"🧮 Calculator: {env.current_value}"
    elif isinstance(env, BinarySearchEnv):
        status = "✓" if env.correct else "..."
        return f"🔍 Binary Search [{env.range_min}, {env.range_max}]: {status}"
    elif isinstance(env, SearchWrapperEnv):
        # Recursive handling
        inner = display_environment(env.inner_env)
        return f"🌳 Search (depth={env.depth}): {inner}"
    else:
        # Generic handling for user environments
        return env.get_display_string()

def extract_final_value(env: Environment) -> Any:
    """Extract the 'final value' from any environment."""
    if isinstance(env, CalculatorEnv):
        return env.current_value
    elif isinstance(env, BinarySearchEnv):
        return env.answer if env.correct else None
    elif isinstance(env, SearchWrapperEnv):
        return extract_final_value(env.inner_env)
    else:
        # For unknown types, try to get from dict
        data = env.to_dict()
        return data.get("value") or data.get("result") or data
```

### CLI Integration (rollouts/cli/agent_cli.py)

```python
def process_rollout_result(result: RolloutResult) -> None:
    """Process rollout result for CLI display."""
    env = result.environment
    
    # Display using the environment's method
    print(f"Final state: {display_environment(env)}")
    
    # Save to registry - works with any environment
    state_dict = env.to_dict()
    save_to_registry(state_dict)
    
    # Try to extract a final value
    final_value = extract_final_value(env)
    if final_value is not None:
        print(f"Result: {final_value}")
```

## User Code Examples

### Example 1: Custom Chess Environment

```python
# user_project/environments.py
from dataclasses import dataclass
from typing import Literal, Dict, Any, List, Union
from rollouts.dtypes import Environment, CalculatorEnv, BinarySearchEnv

@dataclass
class ChessEnv:
    """Custom chess environment."""
    type: Literal["chess"] = "chess"
    board_state: str  # FEN notation
    move_count: int
    white_to_move: bool
    captured_pieces: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "chess",
            "board_state": self.board_state,
            "move_count": self.move_count,
            "white_to_move": self.white_to_move,
            "captured_pieces": self.captured_pieces
        }
    
    def get_tools(self) -> List[Tool]:
        return [ChessTool()]  # Custom chess-specific tools
    
    def get_display_string(self) -> str:
        turn = "White" if self.white_to_move else "Black"
        return f"♟️ Chess - Move {self.move_count} ({turn} to play)"

# Create a union for this project's type safety
ProjectEnv = Union[CalculatorEnv, BinarySearchEnv, ChessEnv]

def analyze_game(env: ProjectEnv) -> str:
    """Type-safe analysis of environments in this project."""
    if env.type == "calculator":
        return f"Math result: {env.current_value}"  # ✅ Type safe!
    elif env.type == "binary_search":
        return f"Found {env.answer} in {env.turns} turns"  # ✅ Type safe!
    elif env.type == "chess":
        pieces = len(env.captured_pieces)  # ✅ Type safe!
        return f"Chess game: {pieces} pieces captured"
```

### Example 2: Using with Framework

```python
from rollouts.runner import run_rollout, display_environment
from rollouts.dtypes import SearchWrapperEnv
from user_project.environments import ChessEnv

# Create custom environment
chess_env = ChessEnv(
    board_state="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR",
    move_count=0,
    white_to_move=True
)

# Works seamlessly with framework ✅
result = run_rollout(chess_env, agent)
print(display_environment(result.environment))

# Can even wrap in search!
search_chess = SearchWrapperEnv(
    inner_env=chess_env,  # Protocol allows any environment
    depth=0,
    search_config={"max_depth": 3}
)

result2 = run_rollout(search_chess, agent)
```

### Example 3: Mixed Environment Processing

```python
from typing import List
from rollouts.dtypes import Environment, CalculatorEnv
from user_project.environments import ChessEnv, ProjectEnv

def process_batch(environments: List[Environment]) -> None:
    """Process a batch of mixed environment types."""
    for env in environments:
        # Framework functions work with all
        print(display_environment(env))
        
        # Can still type-narrow when needed
        if isinstance(env, CalculatorEnv):
            print(f"  Math: {env.current_value}")
        elif isinstance(env, ChessEnv):
            print(f"  Chess: {env.move_count} moves")

# Mix built-in and custom environments
batch = [
    CalculatorEnv(current_value=42.0),
    ChessEnv(board_state="...", move_count=10, white_to_move=False),
    BinarySearchEnv(range_min=0, range_max=100, answer=42)
]

process_batch(batch)  # ✅ All work!
```

## Key Benefits

### 1. **Type Safety Where It Matters**
- Library code has full type safety for built-in environments
- Users get type safety for their own environments via custom unions
- Type narrowing with `isinstance()` works perfectly

### 2. **True Extensibility**
- Users just implement the protocol - no registration needed
- Custom environments work with all framework functions
- Can even wrap custom environments with built-in wrappers

### 3. **Zero Magic**
- No dynamic type generation
- No TypeVars proliferating through the codebase
- Standard Python patterns that IDEs understand

### 4. **Progressive Enhancement**
- Users can start with just the protocol (loose typing)
- Can add their own tagged unions for stricter typing
- Can mix and match library and custom environments

### 5. **Clean Separation**
- Library doesn't need to know about user types
- User doesn't need to modify library code
- Each layer has appropriate type strictness

## Migration Path

To migrate the existing codebase:

1. **Convert base `Environment` class to Protocol**
2. **Convert existing environments to dataclasses with `type` field**
3. **Update display/access code to use type narrowing**
4. **Remove all `.current_value` direct access in favor of helper functions**

Example migration for problem code:

```python
# OLD - Type error
print(f"🧮 Final calculator value: {final_state.environment.current_value}")

# NEW - Type safe
from rollouts.runner import extract_final_value
value = extract_final_value(final_state.environment)
if value is not None:
    print(f"🧮 Final value: {value}")

# OR using display function
print(display_environment(final_state.environment))
```

## Summary

This approach gives you:
- **Library type safety** via tagged unions for built-ins
- **User extensibility** via protocols
- **No TypeVar complexity** spreading through the codebase  
- **Clean, pythonic code** that type checkers understand
- **Progressive typing** - users can be as strict or loose as they want

The key insight is that the library provides patterns and building blocks, while users compose them for their specific needs. The protocol acts as the contract between library and user code, while tagged unions provide type safety within each domain.
