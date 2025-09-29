"""
Advanced search environment for rollouts framework.

Adds conjunctive (decompose) and disjunctive (branch) search to any Environment.
Adds conjunctive (decompose) and disjunctive (branch) search to any Environment.
"""

from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Callable
import asyncio
import json

from ..dtypes import (
    Message, Trajectory, StopReason, Tool, ToolFunction, ToolFunctionParameter, ToolCall, ToolResult,
    AgentState, RunConfig, Environment, default_confirm_tool
)
from ..agents import (
    run_agent
)


# ── Search Configuration ──────────────────────────────────────────────────────

@dataclass
class SearchConfig:
    """Configuration for search behavior."""
    context_passer: Callable[[AgentState, Dict], AgentState]
    context_passer_name: str  # Name for registry lookup during serialization
    autonomous_subagents: bool = True
    max_depth: int = 3
    timeout_per_branch: float = 300.0  # 5 minutes per branch
    debug_sequential: bool = False  # Run searches sequentially for debugging
    
    def transform_run_config(self, parent_config: RunConfig, sub_name: Optional[str] = None) -> RunConfig:
        """Transform parent RunConfig for sub-agents."""
        if self.autonomous_subagents:
            return replace(parent_config,
                confirm_tool=default_confirm_tool,
                on_step_start=lambda s: s,  # Disable warnings
                handle_no_tool=lambda s, cfg: inject_tool_reminder_handler(s)
            )
        else:
            return parent_config


# ── No Tool Handlers ──────────────────────────────────────────────────────────

def inject_tool_reminder_handler(state: AgentState) -> AgentState:
    """Inject a reminder to use tools when no tools are called."""
    reminder = Message(
        role="user",
        content="Please use the available tools to complete your task. You must actively use tools rather than just providing text responses. Use the tools step by step to solve the problem.",
        tool_calls=[]
    )
    new_trajectory = replace(
        state.actor.trajectory,
        messages=state.actor.trajectory.messages + [reminder]
    )
    return replace(state, actor=replace(state.actor, trajectory=new_trajectory))


# ── Context Passing Strategies ────────────────────────────────────────────────

def default_context_passer(parent_state: AgentState, branch_spec: Dict) -> AgentState:
    """Create fresh sub-agent state with just the branch description."""
    sys_msg = Message(
        role="system", 
        content=f"""You are solving a subproblem. Your specific task: {branch_spec['description']}

IMPORTANT: You must use the available tools to complete this task directly. Do NOT use search tools (branch or decompose) - solve the problem using the basic tools available. Use the tools step by step to solve the problem. When you have completed the task, use any completion tool available (like complete_task) to mark it as finished."""
    )
    user_msg = Message(
        role="user", 
        content=branch_spec['description']
    )
    
    new_trajectory = Trajectory(messages=[sys_msg, user_msg])
    new_actor = replace(parent_state.actor, trajectory=new_trajectory)
    
    return replace(parent_state, 
                   actor=new_actor,
                   turn_idx=0,
                   pending_tool_calls=[],
                   next_tool_idx=0)


def inherit_context_passer(parent_state: AgentState, branch_spec: Dict) -> AgentState:
    """Inherit parent trajectory and add branch-specific message."""
    branch_msg = Message(
        role="user", 
        content=f"""Now focus specifically on: {branch_spec['description']}

IMPORTANT: Use the available tools to complete this specific task. Do not just provide a text response - actively use the tools to solve the problem step by step. When finished, use any completion tool available to mark the task as complete."""
    )
    
    new_trajectory = replace(
        parent_state.actor.trajectory,
        messages=parent_state.actor.trajectory.messages + [branch_msg]
    )
    new_actor = replace(parent_state.actor, trajectory=new_trajectory)
    
    return replace(parent_state, 
                   actor=new_actor, 
                   turn_idx=0,
                   pending_tool_calls=[],
                   next_tool_idx=0)


def summary_context_passer(parent_state: AgentState, branch_spec: Dict) -> AgentState:
    """Pass a summary of recent conversation plus branch task."""
    # Simple summary of last few messages
    recent_messages = parent_state.actor.trajectory.messages[-4:]
    summary_parts = []
    for msg in recent_messages:
        if msg.role == "user":
            summary_parts.append(f"User: {msg.content}")
        elif msg.role == "assistant" and msg.content:
            # Truncate long assistant messages
            content = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
            summary_parts.append(f"Assistant: {content}")
    
    context_summary = "\n".join(summary_parts)
    
    sys_msg = Message(
        role="system",
        content=f"""You are solving a subproblem as part of a larger conversation.

Recent context:
{context_summary}

Your specific task: {branch_spec['description']}

IMPORTANT: Use the available tools to complete this task. Do not just provide text responses - actively use the tools to solve the problem step by step."""
    )
    user_msg = Message(
        role="user", 
        content=branch_spec['description']
    )
    
    new_trajectory = Trajectory(messages=[sys_msg, user_msg])
    new_actor = replace(parent_state.actor, trajectory=new_trajectory)
    
    return replace(parent_state, 
                   actor=new_actor,
                   turn_idx=0,
                   pending_tool_calls=[],
                   next_tool_idx=0)


# Context passer registry
CONTEXT_PASSER_REGISTRY = {
    "default": default_context_passer,
    "inherit": inherit_context_passer,
    "summary": summary_context_passer,
}


def create_search_config(context_passer_name: str, **kwargs) -> SearchConfig:
    """Create SearchConfig with context passer from registry."""
    if context_passer_name not in CONTEXT_PASSER_REGISTRY:
        raise ValueError(f"Unknown context passer: {context_passer_name}. Available: {list(CONTEXT_PASSER_REGISTRY.keys())}")
    
    return SearchConfig(
        context_passer=CONTEXT_PASSER_REGISTRY[context_passer_name],
        context_passer_name=context_passer_name,
        **kwargs
    )


# ── Search Environment ────────────────────────────────────────────────────────

@dataclass
class SearchEnvironment:
    """
    Environment wrapper that adds search capabilities to any inner environment.
    
    Adds two tools:
    - 'branch': Try different approaches (disjunctive - first success wins)
    - 'decompose': Break into subproblems (conjunctive - all must succeed)
    """
    
    """Environment wrapper that adds search capabilities via composition."""
    inner_env: Environment
    search_config: SearchConfig
    depth: int = 0
    
    async def serialize(self) -> dict:
        """Serialize inner environment state and SearchConfig."""
        inner_data = await self.inner_env.serialize() if hasattr(self.inner_env, 'serialize') else {}
        return {
            "inner_env_data": inner_data,
            "inner_env_class": self.inner_env.__class__.__name__,
            "search_config": {
                "context_passer_name": self.search_config.context_passer_name,
                "autonomous_subagents": self.search_config.autonomous_subagents,
                "max_depth": self.search_config.max_depth,
                "timeout_per_branch": self.search_config.timeout_per_branch,
                "debug_sequential": self.search_config.debug_sequential,
            },
            "depth": self.depth
        }
    
    @staticmethod
    async def deserialize(data: dict) -> 'SearchEnvironment':
        """Deserialize search environment, reconstructing SearchConfig from registry."""
        # Simple environment registry
        from .calculator import CalculatorEnvironment
        env_registry = {
            "CalculatorEnvironment": CalculatorEnvironment,
        }
        
        inner_env_class = env_registry.get(data["inner_env_class"])
        if not inner_env_class:
            raise ValueError(f"Unknown environment class: {data['inner_env_class']}")
            
        if hasattr(inner_env_class, 'deserialize'):
            inner_env = await inner_env_class.deserialize(data["inner_env_data"])
        else:
            inner_env = inner_env_class()
        
        # Reconstruct SearchConfig from serialized data + registry
        search_config = create_search_config(
            context_passer_name=data["search_config"]["context_passer_name"],
            autonomous_subagents=data["search_config"]["autonomous_subagents"],
            max_depth=data["search_config"]["max_depth"],
            timeout_per_branch=data["search_config"]["timeout_per_branch"],
            debug_sequential=data["search_config"].get("debug_sequential", False),
        )
        
        return SearchEnvironment(inner_env, search_config, data["depth"])
    
    def get_tools(self) -> List[Tool]:
        """Combine inner environment tools with search tools."""
        base_tools = self.inner_env.get_tools()
        
        # Only add search tools if we haven't hit max depth
        if self.depth < self.search_config.max_depth:
            search_tools = [
                Tool(
                    type="function",
                    function=ToolFunction(
                        name="branch",
                        description="Try different approaches to solve the problem (only one needs to succeed)",
                        parameters=ToolFunctionParameter(
                            type="object",
                            properties={
                                "approaches": {
                                    "type": "array",
                                    "description": "List of different approaches to try",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "name": {
                                                "type": "string",
                                                "description": "Name for this approach"
                                            },
                                            "description": {
                                                "type": "string", 
                                                "description": "What this approach will try"
                                            }
                                        },
                                        "required": ["name", "description"]
                                    }
                                }
                            }
                        ),
                        required=["approaches"]
                    )
                ),
                Tool(
                    type="function",
                    function=ToolFunction(
                        name="decompose",
                        description="Break the problem into independent subproblems that will be solved in parallel by separate agent instances (no communication between subproblems! think of this like mapreduce). All subproblems must succeed for this tool to return success, but partial results are returned even if some fail, showing which succeeded/failed.",
                        parameters=ToolFunctionParameter(
                            type="object",
                            properties={
                                "subproblems": {
                                    "type": "array",
                                    "description": "List of subproblems to solve",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "name": {
                                                "type": "string",
                                                "description": "Name for this subproblem"
                                            },
                                            "description": {
                                                "type": "string",
                                                "description": "What this subproblem needs to solve"
                                            }
                                        },
                                        "required": ["name", "description"]
                                    }
                                }
                            }
                        ),
                        required=["subproblems"]
                    )
                )
            ]
            return base_tools + search_tools
        
        return base_tools
    
    async def exec_tool(self, tool_call: ToolCall, current_state: AgentState, 
                       run_config: RunConfig, checkpoint_store=None) -> ToolResult:
        """Execute tool - delegate to inner environment or handle search tools."""
        
        if tool_call.name == "branch":
            return await self._exec_branch(tool_call, current_state, run_config)
        elif tool_call.name == "decompose":
            return await self._exec_decompose(tool_call, current_state, run_config)
        else:
            # Delegate to inner environment
            return await self.inner_env.exec_tool(tool_call, current_state, run_config, checkpoint_store)
    
    def requires_confirmation(self, tool_call: ToolCall) -> bool:
        """Check if tool requires confirmation."""
        if tool_call.name in ["branch", "decompose"]:
            return False  # Search tools don't require confirmation by default
        return self.inner_env.requires_confirmation(tool_call)
    
    async def _exec_branch(self, tool_call: ToolCall, parent_state: AgentState, run_config: RunConfig) -> ToolResult:
        """Execute branch tool - try approaches until one succeeds."""
        # Check depth limit
        if self.depth >= self.search_config.max_depth:
            return ToolResult(
                call_id=tool_call.id,
                ok=False,
                error=f"Maximum search depth ({self.search_config.max_depth}) reached"
            )
            
        try:
            args = json.loads(tool_call.args) if isinstance(tool_call.args, str) else tool_call.args
            approaches = args["approaches"]
            
            print(f"🔍 Branching: Trying {len(approaches)} approaches...")
            
            # Try each approach
            for i, approach in enumerate(approaches):
                print(f"  🌟 Approach {i+1}: {approach['name']}")
                
                try:
                    # Create sub-agent state
                    sub_state = self.search_config.context_passer(parent_state, approach)
                    
                    # Create deeper search environment using serialization pattern
                    env_data = await parent_state.environment.serialize()
                    fresh_env = await parent_state.environment.__class__.deserialize(env_data)
                    if isinstance(fresh_env, SearchEnvironment):
                        fresh_env = SearchEnvironment(fresh_env.inner_env, fresh_env.search_config, fresh_env.depth + 1)
                    sub_state = replace(sub_state, environment=fresh_env)
                    
                    # Configure run config for sub-agent
                    sub_config = self.search_config.transform_run_config(run_config)
                    
                    # Run sub-agent with timeout
                    sub_states = await asyncio.wait_for(
                        run_agent(sub_state, sub_config),
                        timeout=self.search_config.timeout_per_branch
                    )
                    
                    final_sub_state = sub_states[-1]

                    # Check if successful - only TASK_COMPLETED or no stop reason counts as success
                    # MAX_TURNS means the agent timed out without completing, which is a failure
                    if not final_sub_state.stop or final_sub_state.stop == StopReason.TASK_COMPLETED:
                        print(f"  ✅ Approach '{approach['name']}' succeeded!")

                        # Extract result from sub-agent
                        last_message = final_sub_state.actor.trajectory.messages[-1]
                        result_content = last_message.content if last_message.role == "assistant" else f"Completed approach: {approach['name']}"

                        return ToolResult(
                            call_id=tool_call.id,
                            ok=True,
                            content=f"Branch '{approach['name']}' succeeded: {result_content}"
                        )
                    elif final_sub_state.stop == StopReason.MAX_TURNS:
                        print(f"  ⏰ Approach '{approach['name']}' hit max turns without completing")
                        continue
                
                except asyncio.TimeoutError:
                    print(f"  ⏰ Approach '{approach['name']}' timed out")
                    continue
                except Exception as e:
                    print(f"  ❌ Approach '{approach['name']}' failed: {str(e)}")
                    continue
            
            # All approaches failed
            return ToolResult(
                call_id=tool_call.id,
                ok=False,
                error=f"All {len(approaches)} approaches failed"
            )
            
        except Exception as e:
            return ToolResult(
                call_id=tool_call.id,
                ok=False,
                error=f"Branch execution error: {str(e)}"
            )
    
    async def _exec_decompose(self, tool_call: ToolCall, parent_state: AgentState, run_config: RunConfig) -> ToolResult:
        """Execute decompose tool - solve all subproblems."""
        # Check depth limit
        if self.depth >= self.search_config.max_depth:
            return ToolResult(
                call_id=tool_call.id,
                ok=False,
                error=f"Maximum search depth ({self.search_config.max_depth}) reached"
            )
            
        try:
            args = json.loads(tool_call.args) if isinstance(tool_call.args, str) else tool_call.args
            subproblems = args["subproblems"]
            
            print(f"🔍 Decomposing: Solving {len(subproblems)} subproblems...")
            
            results = []
            
            # Solve each subproblem
            for i, subproblem in enumerate(subproblems):
                print(f"  🧩 Subproblem {i+1}: {subproblem['name']}")
                
                try:
                    # Create sub-agent state
                    sub_state = self.search_config.context_passer(parent_state, subproblem)
                    
                    # Create deeper search environment using serialization pattern
                    env_data = await parent_state.environment.serialize()
                    fresh_env = await parent_state.environment.__class__.deserialize(env_data)
                    if isinstance(fresh_env, SearchEnvironment):
                        fresh_env = SearchEnvironment(fresh_env.inner_env, fresh_env.search_config, fresh_env.depth + 1)
                    sub_state = replace(sub_state, environment=fresh_env)
                    
                    # Configure run config for sub-agent
                    sub_config = self.search_config.transform_run_config(run_config)
                    
                    # Run sub-agent with timeout
                    sub_states = await asyncio.wait_for(
                        run_agent(sub_state, sub_config),
                        timeout=self.search_config.timeout_per_branch
                    )
                    
                    final_sub_state = sub_states[-1]

                    # Extract result
                    last_message = final_sub_state.actor.trajectory.messages[-1]
                    result_content = last_message.content if last_message.role == "assistant" else f"Completed subproblem: {subproblem['name']}"

                    # Only TASK_COMPLETED or no stop reason counts as success
                    # MAX_TURNS means timeout without completing, which is a failure
                    is_success = not final_sub_state.stop or final_sub_state.stop == StopReason.TASK_COMPLETED

                    if final_sub_state.stop == StopReason.MAX_TURNS:
                        result_content = "Hit max turns without completing"
                        is_success = False

                    results.append({
                        "name": subproblem["name"],
                        "result": result_content,
                        "success": is_success
                    })

                    if results[-1]["success"]:
                        print(f"  ✅ Subproblem '{subproblem['name']}' completed")
                    else:
                        print(f"  ❌ Subproblem '{subproblem['name']}' failed")
                        
                except asyncio.TimeoutError:
                    print(f"  ⏰ Subproblem '{subproblem['name']}' timed out")
                    results.append({
                        "name": subproblem["name"],
                        "result": "Timed out",
                        "success": False
                    })
                except Exception as e:
                    print(f"  ❌ Subproblem '{subproblem['name']}' error: {str(e)}")
                    results.append({
                        "name": subproblem["name"],
                        "result": f"Error: {str(e)}",
                        "success": False
                    })
            
            # Check if all succeeded
            successful_count = sum(1 for r in results if r["success"])
            
            if successful_count == len(subproblems):
                result_summary = "All subproblems completed:\n" + "\n".join(
                    f"- {r['name']}: {r['result']}" for r in results
                )
                return ToolResult(
                    call_id=tool_call.id,
                    ok=True,
                    content=result_summary
                )
            else:
                result_summary = f"Only {successful_count}/{len(subproblems)} subproblems succeeded:\n" + "\n".join(
                    f"- {r['name']}: {'✅' if r['success'] else '❌'} {r['result']}" for r in results
                )
                return ToolResult(
                    call_id=tool_call.id,
                    ok=False,
                    error=result_summary
                )
                
        except Exception as e:
            return ToolResult(
                call_id=tool_call.id,
                ok=False,
                error=f"Decompose execution error: {str(e)}"
            )
