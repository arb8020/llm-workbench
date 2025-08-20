#!/usr/bin/env python3
"""
Agent CLI - Manage background agents with tmux integration

Commands:
    agent-cli launch <task> [--name NAME] [--env ENV]    # Launch new agent
    agent-cli status [NAME]                              # Check agent status
    agent-cli list                                       # List all agents
    agent-cli watch NAME                                 # Attach to agent tmux
    agent-cli respond NAME <response>                    # Send response to blocked agent
    agent-cli kill NAME                                  # Kill specific agent
    agent-cli cleanup                                    # Kill all agents

Examples:
    agent-cli launch "solve quadratic x^2 + 5x + 6 = 0" --name solver
    agent-cli status solver
    agent-cli respond solver "y"
    agent-cli watch solver
"""

import argparse
import json
import os
import subprocess
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

try:
    import libtmux
    LIBTMUX_AVAILABLE = True
except ImportError:
    LIBTMUX_AVAILABLE = False

@dataclass
class AgentInfo:
    """Information about a running agent."""
    name: str
    task: str
    session_name: str
    pipe_path: str
    env_type: str
    status: str  # running, blocked, completed, error
    created_at: str
    last_updated: str
    result: Optional[str] = None
    error: Optional[str] = None
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'AgentInfo':
        return cls(**data)

class AgentRegistry:
    """Manages the registry of running agents."""
    
    def __init__(self, registry_path: str = "/tmp/agent_cli_registry.json"):
        self.registry_path = registry_path
        
    def load(self) -> Dict[str, AgentInfo]:
        """Load agent registry from disk."""
        if not os.path.exists(self.registry_path):
            return {}
        
        try:
            with open(self.registry_path, 'r') as f:
                data = json.load(f)
                return {name: AgentInfo.from_dict(info) for name, info in data.items()}
        except Exception:
            return {}
    
    def save(self, agents: Dict[str, AgentInfo]):
        """Save agent registry to disk."""
        try:
            data = {name: info.to_dict() for name, info in agents.items()}
            with open(self.registry_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save registry: {e}")
    
    def add_agent(self, info: AgentInfo):
        """Add agent to registry."""
        agents = self.load()
        agents[info.name] = info
        self.save(agents)
    
    def update_agent(self, name: str, **updates):
        """Update agent info."""
        agents = self.load()
        if name in agents:
            for key, value in updates.items():
                if hasattr(agents[name], key):
                    setattr(agents[name], key, value)
            agents[name].last_updated = time.strftime("%Y-%m-%d %H:%M:%S")
            self.save(agents)
    
    def remove_agent(self, name: str):
        """Remove agent from registry."""
        agents = self.load()
        if name in agents:
            del agents[name]
            self.save(agents)
    
    def get_agent(self, name: str) -> Optional[AgentInfo]:
        """Get agent info by name."""
        agents = self.load()
        return agents.get(name)
    
    def list_agents(self) -> List[AgentInfo]:
        """List all agents."""
        agents = self.load()
        return list(agents.values())

class AgentCLI:
    """Main CLI handler."""
    
    def __init__(self):
        self.registry = AgentRegistry()
        
    def create_agent_session(self, name: str, task: str, env_type: str = "calculator") -> Optional[AgentInfo]:
        """Create a new agent tmux session."""
        if not LIBTMUX_AVAILABLE:
            print("❌ libtmux required. Install with: pip install libtmux")
            return None
        
        # Generate unique identifiers
        session_name = f"agent_{name}"
        pipe_path = f"/tmp/agent_{name}_pipe"
        
        # Clean up existing session/pipe
        subprocess.run(["tmux", "kill-session", "-t", session_name], 
                      capture_output=True, text=True)
        if os.path.exists(pipe_path):
            os.unlink(pipe_path)
        
        # Create named pipe
        try:
            os.mkfifo(pipe_path)
        except OSError as e:
            print(f"❌ Could not create pipe: {e}")
            return None
        
        # Create agent script based on environment type
        if env_type == "calculator":
            agent_script = self._create_calculator_script(task, pipe_path, name)
        elif env_type == "search":
            agent_script = self._create_search_script(task, pipe_path, name)
        else:
            print(f"❌ Unknown environment type: {env_type}")
            return None
        
        # Launch tmux session with environment variables
        cmd = [
            "tmux", "new-session", "-d", "-s", session_name,
            "python", "-c", agent_script
        ]
        
        # Pass current environment to subprocess
        env = os.environ.copy()
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        
        if result.returncode != 0:
            print(f"❌ Failed to create tmux session: {result.stderr}")
            if os.path.exists(pipe_path):
                os.unlink(pipe_path)
            return None
        
        # Create agent info
        agent_info = AgentInfo(
            name=name,
            task=task,
            session_name=session_name,
            pipe_path=pipe_path,
            env_type=env_type,
            status="running",
            created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            last_updated=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
        # Register agent
        self.registry.add_agent(agent_info)
        
        return agent_info
    
    def _create_calculator_script(self, task: str, pipe_path: str, name: str = "agent") -> str:
        """Create Python script for calculator agent."""
        api_key = os.environ.get('ANTHROPIC_API_KEY', '')
        return f'''
import asyncio
import sys
import os
import time
sys.path.append("{os.getcwd()}")

# Set up environment variables explicitly
os.environ["ANTHROPIC_API_KEY"] = "{api_key}"

# Import agent framework
from rollouts import *

class SimpleCalc(Environment):
    def __init__(self):
        self.value = 0
        
    async def serialize(self):
        return {{"value": self.value}}
        
    @staticmethod 
    async def deserialize(data):
        calc = SimpleCalc()
        calc.value = data["value"]
        return calc
        
    def requires_confirmation(self, tc):
        return tc.name == "divide"
        
    def get_tools(self):
        return [
            Tool(type="function", function=ToolFunction(
                name="add", description="Add number", 
                parameters=ToolFunctionParameter(
                    type="object", 
                    properties={{"value": {{"type": "number"}}}}
                ),
                required=["value"]
            )),
            Tool(type="function", function=ToolFunction(
                name="subtract", description="Subtract number",
                parameters=ToolFunctionParameter(
                    type="object",
                    properties={{"value": {{"type": "number"}}}}
                ),
                required=["value"]
            )),
            Tool(type="function", function=ToolFunction(
                name="multiply", description="Multiply by number",
                parameters=ToolFunctionParameter(
                    type="object",
                    properties={{"value": {{"type": "number"}}}}
                ),
                required=["value"]
            )),
            Tool(type="function", function=ToolFunction(
                name="divide", description="Divide by number (needs confirm)",
                parameters=ToolFunctionParameter(
                    type="object",
                    properties={{"value": {{"type": "number"}}}}
                ),
                required=["value"]
            )),
            Tool(type="function", function=ToolFunction(
                name="clear", description="Reset to zero",
                parameters=ToolFunctionParameter(type="object", properties={{}})
            )),
            Tool(type="function", function=ToolFunction(
                name="complete", description="Finish calculation",
                parameters=ToolFunctionParameter(
                    type="object", 
                    properties={{"summary": {{"type": "string"}}}},
                ),
                required=["summary"]
            ))
        ]
        
    async def exec_tool(self, tc, state=None, run_config=None, checkpoint_store=None):
        if tc.name == "add":
            val = tc.args.get("value", 0)
            self.value += val
            return ToolResult(tc.id, True, f"Added {{val}}, total: {{self.value}}")
        elif tc.name == "subtract":
            val = tc.args.get("value", 0)
            self.value -= val
            return ToolResult(tc.id, True, f"Subtracted {{val}}, total: {{self.value}}")
        elif tc.name == "multiply":
            val = tc.args.get("value", 1)
            self.value *= val
            return ToolResult(tc.id, True, f"Multiplied by {{val}}, total: {{self.value}}")
        elif tc.name == "divide":
            val = tc.args.get("value", 1)
            if val == 0:
                return ToolResult(tc.id, False, error="Cannot divide by zero")
            self.value /= val
            return ToolResult(tc.id, True, f"Divided by {{val}}, total: {{self.value}}")
        elif tc.name == "clear":
            self.value = 0
            return ToolResult(tc.id, True, "Reset to 0")
        elif tc.name == "complete":
            summary = tc.args.get("summary", "Calculation completed")
            return ToolResult(tc.id, True, f"{{summary}}. Final result: {{self.value}}", 
                            stop_reason=StopReason.TASK_COMPLETED)
        return ToolResult(tc.id, False, error="Unknown tool")

async def tmux_input(prompt):
    print(f"\\n📥 BLOCKED: {{prompt}}")
    print(f"💡 Use: agent-cli respond {name} <response>")
    
    # Write status update
    try:
        with open("/tmp/agent_cli_registry.json", "r") as f:
            import json
            registry = json.load(f)
        if "{name}" in registry:
            registry["{name}"]["status"] = "blocked"
            registry["{name}"]["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
            with open("/tmp/agent_cli_registry.json", "w") as f:
                json.dump(registry, f, indent=2)
    except:
        pass
    
    # Wait for pipe input
    for _ in range(3000):  # 5 minute timeout
        try:
            with open("{pipe_path}", "r") as f:
                response = f.read().strip()
                if response:
                    print(f"✅ Got: {{response}}")
                    # Update status back to running
                    try:
                        with open("/tmp/agent_cli_registry.json", "r") as f:
                            registry = json.load(f)
                        if "{name}" in registry:
                            registry["{name}"]["status"] = "running"
                            registry["{name}"]["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
                            with open("/tmp/agent_cli_registry.json", "w") as f:
                                json.dump(registry, f, indent=2)
                    except:
                        pass
                    return response
        except:
            pass
        await asyncio.sleep(0.1)
    
    print("⏰ Timeout waiting for response")
    return "timeout"

async def tmux_confirm(tc, state, run_config):
    if not state.environment.requires_confirmation(tc):
        return state, ToolConfirmResult(proceed=True)
        
    print(f"\\n⚠️  CONFIRM: {{tc.name}}({{tc.args}})?")
    response = await run_config.on_input("Proceed? (y/n): ")
    
    if response.startswith("y"):
        return state, ToolConfirmResult(proceed=True)
    else:
        return state, ToolConfirmResult(proceed=False)

async def main():
    print(f"🤖 Agent '{name}' starting...")
    print("=" * 60)
    print(f"Task: {task}")
    print("=" * 60)
    
    # Create agent
    calc = SimpleCalc() 
    
    trajectory = Trajectory(messages=[
        Message(role="system", content=f"""You are a helpful calculator assistant. 

Your task: {task}

Available tools:
- add(value): Add a number to current value
- subtract(value): Subtract a number from current value  
- multiply(value): Multiply current value by a number
- divide(value): Divide current value by a number (requires confirmation)
- clear(): Reset current value to zero
- complete(summary): Mark calculation as finished

Work step by step and explain what you're doing. When finished, use complete() with a summary.""", tool_calls=[]),
        Message(role="user", content=f"Please complete this task: {task}", tool_calls=[])
    ])
    
    actor = Actor(trajectory, Endpoint("anthropic", "claude-4-sonnet-20250514", api_key="{api_key}"))
    state = AgentState(actor=actor, environment=calc, max_turns=20, turn_idx=0)
    
    # Custom handlers
    async def chunk_handler(chunk):
        if chunk.kind == "token":
            print(chunk.data.get("text", ""), end="", flush=True)
        elif chunk.kind == "tool_call_complete":
            print(f"\\n🔧 {{chunk.data.get('name')}}({{chunk.data.get('args')}})")
        elif chunk.kind == "tool_result":
            status = "✓" if chunk.data.get("ok") else "✗"
            print(f"  {{status}} {{chunk.data.get('content', '')}}")
    
    config = RunConfig(
        on_chunk=chunk_handler,
        on_input=tmux_input,
        confirm_tool=tmux_confirm
    )
    
    try:
        print("🚀 Starting execution...")
        states = await run_agent(state, config)
        final_state = states[-1]
        
        # Update registry with result
        try:
            with open("/tmp/agent_cli_registry.json", "r") as f:
                registry = json.load(f)
            if "{name}" in registry:
                registry["{name}"]["status"] = "completed"
                registry["{name}"]["result"] = str(final_state.environment.value)
                registry["{name}"]["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
                with open("/tmp/agent_cli_registry.json", "w") as f:
                    json.dump(registry, f, indent=2)
        except:
            pass
        
        print(f"\\n\\n🎯 COMPLETED!")
        print(f"   Final result: {{final_state.environment.value}}")
        print(f"   Status: {{final_state.stop}}")
        
    except Exception as e:
        print(f"\\n❌ Error: {{e}}")
        # Update registry with error
        try:
            with open("/tmp/agent_cli_registry.json", "r") as f:
                registry = json.load(f)
            if "{name}" in registry:
                registry["{name}"]["status"] = "error"
                registry["{name}"]["error"] = str(e)
                registry["{name}"]["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
                with open("/tmp/agent_cli_registry.json", "w") as f:
                    json.dump(registry, f, indent=2)
        except:
            pass
    
    print("\\n💤 Agent finished. Session will stay open for inspection.")
    print(f"🔍 Use 'agent-cli kill {name}' to clean up when done.")
    
    # Keep session alive
    while True:
        await asyncio.sleep(60)

if __name__ == "__main__":
    asyncio.run(main())
'''
    
    def _create_search_script(self, task: str, pipe_path: str, name: str = "agent") -> str:
        """Create Python script for search agent with sub-agent support."""
        api_key = os.environ.get('ANTHROPIC_API_KEY', '')
        return f'''
import asyncio
import sys
import os
import time
import json
import uuid
sys.path.append("{os.getcwd()}")

# Set up environment variables explicitly
os.environ["ANTHROPIC_API_KEY"] = "{api_key}"

# Import search agent framework
from rollouts import *

try:
    # SearchEnvironment is already imported from rollouts
except ImportError as e:
    print(f"❌ Import error: {{e}}")
    print("Make sure search_agent.py is in the Python path")
    exit(1)

try:
    import libtmux
    LIBTMUX_AVAILABLE = True
except ImportError:
    LIBTMUX_AVAILABLE = False

class TmuxSubAgentConfig:
    """Configuration for tmux-enabled sub-agents."""
    def __init__(self, parent_name: str):
        self.parent_name = parent_name
        self.server = libtmux.Server() if LIBTMUX_AVAILABLE else None
        self.sub_agents = {{}}  # name -> {{session, pipe_path}}
    
    def create_sub_agent_session(self, sub_name: str) -> tuple[str, str]:
        """Create tmux session and pipe for sub-agent."""
        if not self.server:
            return None, None
            
        session_name = f"sub_{{self.parent_name}}_{{sub_name}}"
        pipe_path = f"/tmp/sub_{{self.parent_name}}_{{sub_name}}_pipe"
        
        # Clean up existing
        existing = self.server.find_where({{"session_name": session_name}})
        if existing:
            existing.kill()
        if os.path.exists(pipe_path):
            os.unlink(pipe_path)
        
        try:
            # Create session and pipe
            session = self.server.new_session(session_name, detach=True)
            os.mkfifo(pipe_path)
            
            # Set up window
            pane = session.active_pane
            pane.send_keys(f'echo "🔍 Sub-agent: {{sub_name}}"')
            pane.send_keys(f'echo "Parent: {{self.parent_name}}"')
            pane.send_keys(f'echo "Pipe: {{pipe_path}}"')
            pane.send_keys('')
            
            self.sub_agents[sub_name] = {{
                "session": session,
                "pipe_path": pipe_path,
                "session_name": session_name
            }}
            
            # Update registry with sub-agent info
            self.update_registry(sub_name, "created")
            
            return session_name, pipe_path
        except Exception as e:
            print(f"Failed to create sub-agent {{sub_name}}: {{e}}")
            return None, None
    
    def update_registry(self, sub_name: str, status: str, **kwargs):
        """Update agent registry with sub-agent info."""
        try:
            with open("/tmp/agent_cli_registry.json", "r") as f:
                registry = json.load(f)
            
            if "{{name}}" not in registry:
                return
                
            if "sub_agents" not in registry["{{name}}"]:
                registry["{{name}}"]["sub_agents"] = {{}}
            
            registry["{{name}}"]["sub_agents"][sub_name] = {{
                "status": status,
                "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
                **kwargs
            }}
            
            with open("/tmp/agent_cli_registry.json", "w") as f:
                json.dump(registry, f, indent=2)
        except Exception as e:
            print(f"Failed to update registry: {{e}}")
    
    def create_sub_agent_handlers(self, sub_name: str):
        """Create tmux handlers for specific sub-agent."""
        if sub_name not in self.sub_agents:
            return None, None
        
        sub_info = self.sub_agents[sub_name]
        session = sub_info["session"]
        pipe_path = sub_info["pipe_path"]
        pane = session.active_pane
        
        async def sub_chunk_handler(chunk):
            try:
                if chunk.kind == "token":
                    text = chunk.data.get("text", "").replace('"', '\\\\"')
                    pane.send_keys(f'echo -n "{{text}}"', enter=False)
                elif chunk.kind == "tool_call_complete":
                    pane.send_keys(f'echo "\\n🔧 {{chunk.data.get(\\'name\\')}}({{chunk.data.get(\\'args\\')}})"')
                elif chunk.kind == "tool_result":
                    status = "✓" if chunk.data.get("ok") else "✗"
                    pane.send_keys(f'echo "  {{status}} {{chunk.data.get(\\'content\\', \\'\\')[:100]}}"')
            except Exception as e:
                pane.send_keys(f'echo "❌ Chunk error: {{e}}"')
        
        async def sub_input_handler(prompt):
            try:
                pane.send_keys(f'echo "\\n📥 SUB-AGENT BLOCKED: {{prompt}}"')
                pane.send_keys(f'echo "💡 Use: agent-cli respond-sub {{name}} {{sub_name}} <response>"')
                
                # Update registry
                self.update_registry(sub_name, "blocked", prompt=prompt)
                
                # Wait for pipe input
                for _ in range(3000):  # 5 minute timeout
                    try:
                        with open(pipe_path, "r") as f:
                            response = f.read().strip()
                            if response:
                                pane.send_keys(f'echo "✅ Got: {{response}}"')
                                self.update_registry(sub_name, "running")
                                return response
                    except:
                        pass
                    await asyncio.sleep(0.1)
                
                pane.send_keys(f'echo "⏰ Sub-agent timeout"')
                return "timeout"
            except Exception as e:
                pane.send_keys(f'echo "❌ Input error: {{e}}"')
                return "error"
        
        async def sub_confirm_handler(tc, state, run_config):
            if not state.environment.requires_confirmation(tc):
                return state, ToolConfirmResult(proceed=True)
            
            pane.send_keys(f'echo "\\n⚠️  SUB-AGENT CONFIRM: {{tc.name}}({{tc.args}})?"')
            response = await run_config.on_input("Proceed? (y/n): ")
            
            if response.startswith("y"):
                return state, ToolConfirmResult(proceed=True)
            else:
                return state, ToolConfirmResult(proceed=False)
        
        return sub_chunk_handler, sub_input_handler, sub_confirm_handler

# Custom SearchConfig that creates tmux sub-agents
class TmuxSearchConfig:
    def __init__(self, base_config, sub_agent_config):
        self.base_config = base_config
        self.sub_agent_config = sub_agent_config
    
    def __getattr__(self, name):
        return getattr(self.base_config, name)
    
    def transform_run_config(self, parent_config: RunConfig, sub_name: str = None):
        """Transform config for sub-agents with tmux support."""
        if not sub_name or not self.sub_agent_config.server:
            # Fallback to base behavior but need to handle the new signature
            if hasattr(self.base_config, 'transform_run_config'):
                try:
                    # Try with sub_name parameter first
                    return self.base_config.transform_run_config(parent_config, sub_name)
                except TypeError:
                    # Fall back to old signature
                    return self.base_config.transform_run_config(parent_config)
            return parent_config
        
        # Create tmux session for sub-agent
        session_name, pipe_path = self.sub_agent_config.create_sub_agent_session(sub_name)
        if not session_name:
            # Fallback to base behavior
            if hasattr(self.base_config, 'transform_run_config'):
                try:
                    return self.base_config.transform_run_config(parent_config, sub_name)
                except TypeError:
                    return self.base_config.transform_run_config(parent_config)
            return parent_config
        
        # Create custom handlers for this sub-agent
        chunk_handler, input_handler, confirm_handler = self.sub_agent_config.create_sub_agent_handlers(sub_name)
        
        # Start with base config settings, then override with tmux handlers
        try:
            base_config = self.base_config.transform_run_config(parent_config)
        except Exception as e:
            print(f"Warning: Failed to get base config, using parent: {{e}}")
            base_config = parent_config
        
        return RunConfig(
            on_chunk=chunk_handler,
            on_input=input_handler,
            confirm_tool=confirm_handler,
            handle_tool_error=base_config.handle_tool_error,
            on_step_start=base_config.on_step_start,
            handle_stop=base_config.handle_stop,
            handle_no_tool=base_config.handle_no_tool,
            user_message_for_thinking=base_config.user_message_for_thinking
        )

async def tmux_input(prompt):
    print(f"\\n📥 MAIN AGENT BLOCKED: {{prompt}}")
    print(f"💡 Use: agent-cli respond {name} <response>")
    
    # Update registry
    try:
        with open("/tmp/agent_cli_registry.json", "r") as f:
            registry = json.load(f)
        if "{{name}}" in registry:
            registry["{{name}}"]["status"] = "blocked"
            registry["{{name}}"]["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
            with open("/tmp/agent_cli_registry.json", "w") as f:
                json.dump(registry, f, indent=2)
    except:
        pass
    
    # Wait for pipe input
    for _ in range(3000):
        try:
            with open("{{pipe_path}}", "r") as f:
                response = f.read().strip()
                if response:
                    print(f"✅ Got: {{response}}")
                    try:
                        with open("/tmp/agent_cli_registry.json", "r") as f:
                            registry = json.load(f)
                        if "{{name}}" in registry:
                            registry["{{name}}"]["status"] = "running"
                            registry["{{name}}"]["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
                            with open("/tmp/agent_cli_registry.json", "w") as f:
                                json.dump(registry, f, indent=2)
                    except:
                        pass
                    return response
        except:
            pass
        await asyncio.sleep(0.1)
    return "timeout"

async def main():
    print(f"🔍 Search Agent '{name}' starting...")
    print("=" * 60)
    print(f"Task: {task}")
    print("=" * 60)
    
    # Create sub-agent configuration for tmux
    sub_agent_config = TmuxSubAgentConfig("{name}")
    
    # Create search-enabled calculator
    calculator_env = CalculatorEnvironment()
    base_search_config = create_search_config(
        context_passer_name="summary",
        max_depth=2,
        autonomous_subagents=True,
    )
    
    # Wrap with tmux support
    tmux_search_config = TmuxSearchConfig(base_search_config, sub_agent_config)
    search_env = SearchEnvironment(calculator_env, tmux_search_config)
    
    trajectory = Trajectory(messages=[
        Message(role="system", content=f"""You are a mathematical problem solver with advanced search capabilities.

Your task: {task}

You have access to:
- Basic calculator operations (add, subtract, multiply, divide, clear, complete_task)  
- Advanced search operations:
  - branch(approaches): Try different solution methods (only one needs to succeed)
  - decompose(subproblems): Break problem into parts (all parts must be solved)

Use search operations to solve complex problems systematically. Each sub-agent will get its own tmux session for debugging.""", tool_calls=[]),
        Message(role="user", content=f"Please solve: {task}", tool_calls=[])
    ])
    
    actor = Actor(trajectory, Endpoint("anthropic", "claude-4-sonnet-20250514", api_key="{api_key}"))
    state = AgentState(actor=actor, environment=search_env, max_turns=25, turn_idx=0)
    
    # Main agent handlers
    async def chunk_handler(chunk):
        if chunk.kind == "token":
            print(chunk.data.get("text", ""), end="", flush=True)
        elif chunk.kind == "tool_call_complete":
            print(f"\\n🔧 {{chunk.data.get('name')}}({{chunk.data.get('args')}})")
        elif chunk.kind == "tool_result":
            status = "✓" if chunk.data.get("ok") else "✗"
            print(f"  {{status}} {{chunk.data.get('content', '')}}")
    
    config = RunConfig(
        on_chunk=chunk_handler,
        on_input=tmux_input
    )
    
    try:
        print("🚀 Starting search agent execution...")
        states = await run_agent(state, config)
        final_state = states[-1]
        
        # Update registry with result
        try:
            with open("/tmp/agent_cli_registry.json", "r") as f:
                registry = json.load(f)
            if "{{name}}" in registry:
                registry["{{name}}"]["status"] = "completed"
                if hasattr(final_state.environment, 'inner_env'):
                    registry["{{name}}"]["result"] = str(final_state.environment.inner_env.current_value)
                registry["{{name}}"]["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
                with open("/tmp/agent_cli_registry.json", "w") as f:
                    json.dump(registry, f, indent=2)
        except:
            pass
        
        print(f"\\n\\n🎯 SEARCH AGENT COMPLETED!")
        if hasattr(final_state.environment, 'inner_env'):
            print(f"   Final result: {{final_state.environment.inner_env.current_value}}")
        print(f"   Status: {{final_state.stop}}")
        
    except Exception as e:
        print(f"\\n❌ Error: {{e}}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
'''
    
    def detect_agent_blocking(self, agent_info: AgentInfo) -> tuple[bool, str]:
        """Detect if agent is blocked by analyzing tmux content."""
        try:
            result = subprocess.run(
                ["tmux", "capture-pane", "-t", agent_info.session_name, "-p"],
                capture_output=True, text=True
            )
            
            if result.returncode != 0:
                return False, ""
            
            content = result.stdout.strip().lower()
            
            # Look for blocking indicators
            blocking_patterns = [
                ("📥", "waiting for input"),
                ("blocked:", "blocked on operation"),
                ("confirm:", "waiting for confirmation"),
                ("proceed?", "waiting for confirmation"),
                ("choice:", "waiting for user choice"),
                ("(y/n)", "waiting for yes/no response"),
                ("respond via:", "waiting for pipe input"),
                ("echo", "waiting for pipe response"),
                ("use: agent-cli respond", "waiting for cli response")
            ]
            
            for pattern, description in blocking_patterns:
                if pattern.lower() in content:
                    return True, description
            
            # Check if content hasn't changed recently (might be stuck)
            lines = content.split('\n')
            if lines:
                last_line = lines[-1].strip()
                # If last line contains "finished" or "completed", it's done
                if any(word in last_line.lower() for word in ["finished", "completed", "done"]):
                    return False, ""
                # If session shows no activity signs, might be blocked
                if not any(sign in content for sign in ["🔧", "✓", "✗", "starting", "running"]):
                    return True, "possibly stuck or waiting"
            
            return False, ""
            
        except Exception as e:
            return False, f"error checking: {e}"
    
    def check_agent_status(self, name: str) -> Optional[AgentInfo]:
        """Check if agent is still running and update status."""
        agent_info = self.registry.get_agent(name)
        if not agent_info:
            return None
        
        # Check if tmux session exists
        result = subprocess.run(
            ["tmux", "has-session", "-t", agent_info.session_name],
            capture_output=True, text=True
        )
        
        if result.returncode != 0:
            # Session doesn't exist - mark as completed or error
            if agent_info.status == "running" or agent_info.status == "blocked":
                self.registry.update_agent(name, status="error", error="Session terminated unexpectedly")
            return self.registry.get_agent(name)
        
        # If agent is marked as running, check if it's actually blocked
        if agent_info.status == "running":
            is_blocked, reason = self.detect_agent_blocking(agent_info)
            if is_blocked:
                self.registry.update_agent(name, status="blocked", error=reason)
                return self.registry.get_agent(name)
        
        # Check if agent completed by looking for completion markers
        if agent_info.status in ["running", "blocked"]:
            try:
                result = subprocess.run(
                    ["tmux", "capture-pane", "-t", agent_info.session_name, "-p"],
                    capture_output=True, text=True
                )
                if result.returncode == 0:
                    content = result.stdout.strip().lower()
                    if "🎯 completed!" in content or "final result:" in content:
                        self.registry.update_agent(name, status="completed")
                        return self.registry.get_agent(name)
            except Exception:
                pass
        
        return agent_info
    
    def send_response(self, name: str, response: str) -> bool:
        """Send response to blocked agent."""
        agent_info = self.registry.get_agent(name)
        if not agent_info:
            print(f"❌ Agent '{name}' not found")
            return False
        
        if not os.path.exists(agent_info.pipe_path):
            print(f"❌ Pipe not found: {agent_info.pipe_path}")
            return False
        
        try:
            with open(agent_info.pipe_path, 'w') as f:
                f.write(response + '\\n')
            print(f"✅ Sent '{response}' to agent '{name}'")
            return True
        except Exception as e:
            print(f"❌ Failed to send response: {e}")
            return False
    
    def send_sub_response(self, parent_name: str, sub_name: str, response: str) -> bool:
        """Send response to blocked sub-agent."""
        pipe_path = f"/tmp/sub_{parent_name}_{sub_name}_pipe"
        
        if not os.path.exists(pipe_path):
            print(f"❌ Sub-agent pipe not found: {pipe_path}")
            return False
        
        try:
            with open(pipe_path, 'w') as f:
                f.write(response + '\\n')
            print(f"✅ Sent '{response}' to sub-agent '{parent_name}/{sub_name}'")
            return True
        except Exception as e:
            print(f"❌ Failed to send response to sub-agent: {e}")
            return False
    
    def kill_agent(self, name: str) -> bool:
        """Kill agent and clean up."""
        agent_info = self.registry.get_agent(name)
        if not agent_info:
            print(f"❌ Agent '{name}' not found")
            return False
        
        # Kill tmux session
        subprocess.run(["tmux", "kill-session", "-t", agent_info.session_name],
                      capture_output=True, text=True)
        
        # Remove pipe
        if os.path.exists(agent_info.pipe_path):
            try:
                os.unlink(agent_info.pipe_path)
            except Exception:
                pass
        
        # Remove from registry
        self.registry.remove_agent(name)
        
        print(f"✅ Killed agent '{name}'")
        return True
    
    def list_agents(self):
        """List all agents with their status."""
        agents = self.registry.list_agents()
        
        if not agents:
            print("No agents running")
            return
        
        print("🤖 AGENTS")
        print("="*80)
        print(f"{'NAME':<15} {'STATUS':<10} {'TASK':<30} {'CREATED':<20}")
        print("-"*80)
        
        for agent in agents:
            # Update status
            self.check_agent_status(agent.name)
            # Re-fetch to get updated status
            updated_agent = self.registry.get_agent(agent.name)
            if updated_agent:
                status_color = {
                    "running": "🟢",
                    "blocked": "🟡", 
                    "completed": "✅",
                    "error": "❌"
                }.get(updated_agent.status, "❓")
                
                task_short = updated_agent.task[:28] + ".." if len(updated_agent.task) > 30 else updated_agent.task
                
                print(f"{updated_agent.name:<15} {status_color} {updated_agent.status:<8} {task_short:<30} {updated_agent.created_at}")
                
                if updated_agent.status == "blocked":
                    print(f"{'':>15} 💡 Use: agent-cli respond {updated_agent.name} <response>")
                elif updated_agent.status == "completed" and updated_agent.result:
                    print(f"{'':>15} 🎯 Result: {updated_agent.result}")
                elif updated_agent.status == "error" and updated_agent.error:
                    print(f"{'':>15} ❌ Error: {updated_agent.error[:40]}")
                
                # Show sub-agents if they exist
                agent_dict = updated_agent.to_dict()
                if "sub_agents" in agent_dict:
                    for sub_name, sub_info in agent_dict["sub_agents"].items():
                        sub_status_color = {
                            "running": "🟢",
                            "blocked": "🟡",
                            "completed": "✅", 
                            "error": "❌",
                            "created": "🔵"
                        }.get(sub_info.get("status", "unknown"), "❓")
                        
                        print(f"{'':>5} └─ {sub_name:<10} {sub_status_color} {sub_info.get('status', 'unknown'):<8}")
                        
                        if sub_info.get("status") == "blocked":
                            print(f"{'':>15} 💡 Use: agent-cli respond-sub {updated_agent.name} {sub_name} <response>")
        
        print("-"*80)
    
    def watch_agent(self, name: str, follow: bool = True):
        """Watch agent output like tail -f."""
        agent_info = self.registry.get_agent(name)
        if not agent_info:
            print(f"❌ Agent '{name}' not found")
            return
        
        # Check if tmux session exists
        result = subprocess.run(
            ["tmux", "has-session", "-t", agent_info.session_name],
            capture_output=True, text=True
        )
        
        if result.returncode != 0:
            print(f"❌ Agent session '{agent_info.session_name}' not found")
            return
        
        if follow:
            print(f"📺 Watching agent '{name}' (like tail -f)...")
            print("💡 Press Ctrl+C to stop watching")
            print("=" * 60)
            
            try:
                # Start by showing recent history
                result = subprocess.run(
                    ["tmux", "capture-pane", "-t", agent_info.session_name, "-p"],
                    capture_output=True, text=True
                )
                if result.returncode == 0:
                    # Show last 20 lines
                    lines = result.stdout.strip().split('\n')
                    for line in lines[-20:]:
                        print(line)
                
                print("-" * 60)
                print("📡 Following new output...")
                
                # Follow new output
                import time
                last_content = ""
                
                while True:
                    result = subprocess.run(
                        ["tmux", "capture-pane", "-t", agent_info.session_name, "-p"],
                        capture_output=True, text=True
                    )
                    
                    if result.returncode != 0:
                        print(f"\\n❌ Lost connection to agent '{name}'")
                        break
                    
                    current_content = result.stdout.strip()
                    
                    # Only show new lines
                    if current_content != last_content:
                        current_lines = current_content.split('\\n')
                        last_lines = last_content.split('\\n') if last_content else []
                        
                        # Find new lines
                        if len(current_lines) > len(last_lines):
                            new_lines = current_lines[len(last_lines):]
                            for line in new_lines:
                                print(line)
                        elif current_content != last_content:
                            # Content changed, show diff (simple approach)
                            print("\\n" + "-" * 40 + " UPDATE " + "-" * 40)
                            for line in current_lines[-5:]:  # Show last 5 lines on update
                                print(line)
                        
                        last_content = current_content
                    
                    time.sleep(0.5)  # Check every 500ms
                    
            except KeyboardInterrupt:
                print(f"\\n\\n📺 Stopped watching agent '{name}'")
            except Exception as e:
                print(f"\\n❌ Error watching agent: {e}")
        else:
            # Just show current state (non-following)
            result = subprocess.run(
                ["tmux", "capture-pane", "-t", agent_info.session_name, "-p"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                print(f"📺 Current state of agent '{name}':")
                print("=" * 60)
                print(result.stdout.strip())
                print("=" * 60)

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Agent CLI - Manage background agents")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Launch command
    launch_parser = subparsers.add_parser("launch", help="Launch new agent")
    launch_parser.add_argument("task", help="Task for the agent to complete")
    launch_parser.add_argument("--name", help="Agent name (auto-generated if not provided)")
    launch_parser.add_argument("--env", choices=["calculator", "search"], default="calculator", 
                              help="Environment type")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Check agent status")
    status_parser.add_argument("name", help="Agent name")
    
    # List command
    subparsers.add_parser("list", help="List all agents")
    
    # Watch command
    watch_parser = subparsers.add_parser("watch", help="Watch agent output like tail -f")
    watch_parser.add_argument("name", help="Agent name")
    watch_parser.add_argument("--no-follow", action="store_true", help="Show current state without following")
    
    # Respond command
    respond_parser = subparsers.add_parser("respond", help="Send response to blocked agent")
    respond_parser.add_argument("name", help="Agent name")
    respond_parser.add_argument("response", help="Response to send")
    
    # Respond to sub-agent command
    respond_sub_parser = subparsers.add_parser("respond-sub", help="Send response to blocked sub-agent")
    respond_sub_parser.add_argument("parent", help="Parent agent name")
    respond_sub_parser.add_argument("subagent", help="Sub-agent name")
    respond_sub_parser.add_argument("response", help="Response to send")
    
    # Kill command
    kill_parser = subparsers.add_parser("kill", help="Kill specific agent")
    kill_parser.add_argument("name", help="Agent name")
    
    # Cleanup command
    subparsers.add_parser("cleanup", help="Kill all agents")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    cli = AgentCLI()
    
    if args.command == "launch":
        name = args.name or f"agent_{int(time.time())}"
        print(f"🚀 Launching agent '{name}'...")
        
        agent_info = cli.create_agent_session(name, args.task, args.env)
        if agent_info:
            print(f"✅ Agent '{name}' launched successfully!")
            print(f"📺 Watch: agent-cli watch {name}")
            print(f"📊 Status: agent-cli status {name}")
            print(f"🔗 Respond: agent-cli respond {name} <response>")
        
    elif args.command == "status":
        agent_info = cli.check_agent_status(args.name)
        if agent_info:
            print(f"🤖 Agent: {agent_info.name}")
            print(f"📋 Task: {agent_info.task}")
            print(f"📊 Status: {agent_info.status}")
            print(f"🕐 Created: {agent_info.created_at}")
            print(f"🕑 Updated: {agent_info.last_updated}")
            
            if agent_info.status == "blocked":
                print(f"💡 Respond: agent-cli respond {agent_info.name} <response>")
            elif agent_info.status == "completed" and agent_info.result:
                print(f"🎯 Result: {agent_info.result}")
            elif agent_info.status == "error" and agent_info.error:
                print(f"❌ Error: {agent_info.error}")
        else:
            print(f"❌ Agent '{args.name}' not found")
    
    elif args.command == "list":
        cli.list_agents()
    
    elif args.command == "watch":
        cli.watch_agent(args.name, follow=not args.no_follow)
    
    elif args.command == "respond":
        cli.send_response(args.name, args.response)
    
    elif args.command == "respond-sub":
        cli.send_sub_response(args.parent, args.subagent, args.response)
    
    elif args.command == "kill":
        cli.kill_agent(args.name)
    
    elif args.command == "cleanup":
        agents = cli.registry.list_agents()
        killed = 0
        for agent in agents:
            if cli.kill_agent(agent.name):
                killed += 1
        print(f"✅ Cleaned up {killed} agents")

if __name__ == "__main__":
    main()