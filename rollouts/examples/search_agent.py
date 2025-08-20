"""
search_agent.py - Search capabilities for agent framework

Adds conjunctive (decompose) and disjunctive (branch) search to any Environment.
"""

from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Callable
import asyncio
import traceback

from rollouts import (
    Environment, Tool, ToolFunction, ToolFunctionParameter, ToolCall, ToolResult,
    StopReason, Message, Trajectory, Endpoint, Actor, AgentState, RunConfig,
    run_agent, stdout_handler, confirm_tool_with_feedback,
    handle_tool_error, handle_stop_max_turns, inject_tool_reminder,
    default_confirm_tool
)

# ── Search Configuration ──────────────────────────────────────────────────────

def create_search_config(context_passer_name: str, **kwargs) -> 'SearchConfig':
    """Create SearchConfig with context passer from registry."""
    if context_passer_name not in CONTEXT_PASSER_REGISTRY:
        raise ValueError(f"Unknown context passer: {context_passer_name}. Available: {list(CONTEXT_PASSER_REGISTRY.keys())}")
    
    return SearchConfig(
        context_passer=CONTEXT_PASSER_REGISTRY[context_passer_name],
        context_passer_name=context_passer_name,
        **kwargs
    )

@dataclass
class SearchConfig:
    """Configuration for search behavior."""
    context_passer: Callable[[AgentState, Dict], AgentState]  # Required - no default
    context_passer_name: str  # Name for registry lookup during serialization
    autonomous_subagents: bool = True
    max_depth: int = 3
    timeout_per_branch: float = 300.0  # 5 minutes per branch
    debug_sequential: bool = False  # Run searches sequentially for debugging
    # TODO: Add more configuration options:
    # TODO: - max_concurrent_subagents: int = None  # Rate limiting
    # TODO: - cost_limit: Optional[float] = None  # Budget control
    # TODO: - result_extractor: Callable[[AgentState], Any] = default_result_extractor
    # TODO: - success_criteria: Callable[[AgentState], bool] = default_success_criteria
    # TODO: - retry_failed_branches: bool = False
    
    def transform_run_config(self, parent_config: RunConfig, sub_name: Optional[str] = None) -> RunConfig:
        """Transform parent RunConfig for sub-agents."""
        if self.autonomous_subagents:
            return RunConfig(
                on_chunk=parent_config.on_chunk,
                confirm_tool=default_confirm_tool,  # Use the default async handler
                handle_tool_error=parent_config.handle_tool_error,
                on_step_start=lambda s: s,  # Disable warnings  
                handle_stop=parent_config.handle_stop,
                handle_no_tool=inject_tool_reminder_handler,  # Remind to use tools instead of stopping
                user_message_for_thinking=parent_config.user_message_for_thinking
            )
        else:
            # TODO: Route confirmations back to parent (future feature)
            # TODO: This would require a way to send confirmation requests back up the tree
            # TODO: Maybe use asyncio queues or callback functions?
            # TODO: For now, non-autonomous sub-agents use parent config unchanged
            return parent_config

# ── No Tool Handlers ──────────────────────────────────────────────────────────

async def inject_tool_reminder_handler(state: AgentState, run_config: RunConfig) -> AgentState:
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

IMPORTANT: You must use the available tools to complete this task. Do not just provide a text response - actively use the tools to solve the problem step by step. When you have completed the task, use any completion tool available (like complete_task) to mark it as finished."""
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

IMPORTANT: Use the available tools to complete this specific task. Do not just provide a text response - actively use the tools to solve the problem step by step. When finished, use any completion tool available to mark the task as complete."""
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

# ── Context Passer Registry ───────────────────────────────────────────────────

CONTEXT_PASSER_REGISTRY = {
    "default": default_context_passer,
    "inherit": inherit_context_passer,
    "summary": summary_context_passer,
}

# ── Search Environment ────────────────────────────────────────────────────────

class SearchEnvironment(Environment):
    """
    Environment wrapper that adds search capabilities to any inner environment.
    
    Adds two tools:
    - 'branch': Try different approaches (disjunctive - first success wins)
    - 'decompose': Break into subproblems (conjunctive - all must succeed)
    """
    
    def __init__(self, inner_env: Environment, search_config: SearchConfig, depth: int = 0):
        self.inner_env = inner_env
        self.search_config = search_config
        self.depth = depth
    
    async def serialize(self) -> dict:
        """Serialize inner environment state and SearchConfig (using registry for context_passer)."""
        inner_data = await self.inner_env.serialize()
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
        # TODO: Need environment registry to recreate inner_env
        # This is a limitation - we need to register environment classes somewhere
        # For now, assume CalculatorEnvironment is available
        env_registry = {
            "CalculatorEnvironment": CalculatorEnvironment,
            # TODO: Add other environments here as needed
            # TODO: This should be a global registry, not hardcoded here
        }
        
        inner_env_class = env_registry[data["inner_env_class"]]
        inner_env = await inner_env_class.deserialize(data["inner_env_data"])
        
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
                        description="Break the problem into subproblems that must all be solved",
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
        else:
            return base_tools  # No search tools at max depth
    
    async def exec_tool(self, tool_call: ToolCall, current_state: AgentState,
                       run_config: RunConfig, checkpoint_store = None) -> ToolResult:
        """Execute tool call - handle search operations or delegate to inner environment."""
        if tool_call.name == "branch":
            return await self._handle_branch(tool_call, current_state, run_config, checkpoint_store)
        elif tool_call.name == "decompose":
            return await self._handle_decompose(tool_call, current_state, run_config, checkpoint_store)
        else:
            # Delegate to inner environment
            return await self.inner_env.exec_tool(tool_call, current_state, run_config, checkpoint_store)
    
    async def _handle_branch(self, tool_call: ToolCall, current_state: AgentState,
                           run_config: RunConfig, checkpoint_store) -> ToolResult:
        """Handle disjunctive branching (first success wins)."""
        approaches = tool_call.args.get("approaches", [])
        if not approaches:
            return ToolResult(
                call_id=tool_call.id,
                ok=False,
                error="No approaches specified"
            )
        
        if self.depth >= self.search_config.max_depth:
            return ToolResult(
                call_id=tool_call.id,
                ok=False,
                error=f"Maximum search depth ({self.search_config.max_depth}) reached"
            )
        
        try:
            if self.search_config.debug_sequential:
                result = await self._run_disjunctive_search_debug(current_state, approaches, run_config, checkpoint_store)
            else:
                result = await self._run_disjunctive_search(current_state, approaches, run_config, checkpoint_store)
            
            if result.get("success"):
                return ToolResult(
                    call_id=tool_call.id,
                    ok=True,
                    content=f"✓ Branch search succeeded! Winning approach: {result.get('name', 'unknown')}\nResult: {result.get('result', 'completed')}"
                )
            else:
                return ToolResult(
                    call_id=tool_call.id,
                    ok=False,
                    error=f"All branch approaches failed: {result.get('message', 'unknown error')}"
                )
        except Exception as e:
            return ToolResult(
                call_id=tool_call.id,
                ok=False,
                error=f"Branch search failed: {str(e)}"
            )
    
    async def _handle_decompose(self, tool_call: ToolCall, current_state: AgentState,
                              run_config: RunConfig, checkpoint_store) -> ToolResult:
        """Handle conjunctive decomposition (all must succeed)."""
        subproblems = tool_call.args.get("subproblems", [])
        if not subproblems:
            return ToolResult(
                call_id=tool_call.id,
                ok=False,
                error="No subproblems specified"
            )
        
        if self.depth >= self.search_config.max_depth:
            return ToolResult(
                call_id=tool_call.id,
                ok=False,
                error=f"Maximum search depth ({self.search_config.max_depth}) reached"
            )
        
        try:
            if self.search_config.debug_sequential:
                results = await self._run_conjunctive_search_debug(current_state, subproblems, run_config, checkpoint_store)
            else:
                results = await self._run_conjunctive_search(current_state, subproblems, run_config, checkpoint_store)
            success_count = sum(1 for r in results if r.get("success", False))
            all_succeeded = success_count == len(results)
            
            # Format results summary
            summary_lines = [f"Decomposition Results ({success_count}/{len(results)} succeeded)"]
            summary_lines.append("─" * 50)
            
            for r in results:
                status = "✓" if r.get("success") else "✗"
                name = r.get("name", "unknown")
                if r.get("success"):
                    summary_lines.append(f"{status} {name}: {r.get('result', 'completed')}")
                else:
                    error_msg = r.get("error", r.get("reason", "unknown error"))
                    summary_lines.append(f"{status} {name}: {error_msg}")
            
            return ToolResult(
                call_id=tool_call.id,
                ok=all_succeeded,
                content="\n".join(summary_lines) if all_succeeded else f"Only {success_count}/{len(results)} subproblems succeeded",
                error=None if all_succeeded else f"Only {success_count}/{len(results)} subproblems succeeded"
            )
        except Exception as e:
            return ToolResult(
                call_id=tool_call.id,
                ok=False,
                error=f"Decomposition failed: {str(e)}"
            )
    
    async def _run_single_subagent(self, parent_state: AgentState, branch_spec: Dict,
                                 run_config: RunConfig, checkpoint_store) -> Dict[str, Any]:
        """Run a single sub-agent with search capabilities."""
        try:
            # Create sub-agent state using context passer
            sub_state = self.search_config.context_passer(parent_state, branch_spec)
            
            # DESIGN DECISION: Use same serialization pattern as process_pending_tools
            # This creates a fresh copy of the entire SearchEnvironment (including inner state)
            # rather than manually managing inner_env serialization. This is simpler and
            # follows the established pattern in the codebase.
            env_data = await parent_state.environment.serialize()
            fresh_env = await parent_state.environment.__class__.deserialize(env_data)
            
            # Increment depth for sub-agent (since we're going one level deeper)
            if isinstance(fresh_env, SearchEnvironment):
                fresh_env = SearchEnvironment(fresh_env.inner_env, fresh_env.search_config, fresh_env.depth + 1)
            
            # Update sub-state with fresh environment
            sub_state = replace(sub_state, environment=fresh_env)
            
            # Extract sub-agent name for identification
            sub_name = branch_spec.get('name', f'unnamed_{id(sub_state)}')
            
            # Transform run config for sub-agent
            sub_run_config = self.search_config.transform_run_config(run_config, sub_name)
            
            # Create unique session ID for sub-agent checkpoints
            # TODO: Better session ID strategy - need to track parent session properly
            # TODO: Should sub-agents share checkpoints with parent or have separate ones?
            if checkpoint_store:
                parent_session = getattr(checkpoint_store, '_current_session', 'unknown')
                sub_session_id = f"{parent_session}_sub_{branch_spec.get('name', 'unnamed')}_{id(sub_state)}"
            else:
                sub_session_id = None
            
            
            # Run sub-agent
            states = await run_agent(sub_state, sub_run_config, sub_session_id)
            final_state = states[-1]
            
            # Extract result from final environment state
            # TODO: Better result extraction - this is very specific to calculator
            # TODO: Maybe environments should have a get_result() method?
            # TODO: Or result extraction could be configurable in SearchConfig?
            result_description = "Task completed"
            
            # Handle both SearchEnvironment and direct CalculatorEnvironment
            env = final_state.environment
            if hasattr(env, 'inner_env') and hasattr(env.inner_env, 'current_value'):
                # SearchEnvironment wrapping calculator
                result_description = f"Final calculation result: {env.inner_env.current_value}"
            elif hasattr(env, 'current_value'):
                # Direct CalculatorEnvironment
                result_description = f"Final calculation result: {env.current_value}"
            
            if final_state.stop == StopReason.TASK_COMPLETED:
                return {
                    "name": branch_spec.get("name", "unknown"),
                    "success": True,
                    "result": result_description,
                    "final_state": final_state
                }
            elif final_state.stop is None:
                # Completed normally (hit max turns but finished)
                return {
                    "name": branch_spec.get("name", "unknown"),
                    "success": True,
                    "result": result_description,
                    "final_state": final_state
                }
            else:
                return {
                    "name": branch_spec.get("name", "unknown"),
                    "success": False,
                    "reason": f"Stopped due to: {final_state.stop}",
                    "final_state": final_state,
                    "debug_info": {
                        "turn_idx": final_state.turn_idx,
                        "max_turns": final_state.max_turns,
                        "pending_tools": len(final_state.pending_tool_calls),
                        "stop_reason": str(final_state.stop)
                    }
                }
                
        except Exception as e:
            return {
                "name": branch_spec.get("name", "unknown"),
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    async def _run_disjunctive_search(self, parent_state: AgentState, approaches: List[Dict],
                                    run_config: RunConfig, checkpoint_store) -> Dict[str, Any]:
        """Race approaches, return first success."""
        
        tasks = [
            self._run_single_subagent(parent_state, approach, run_config, checkpoint_store)
            for approach in approaches
        ]

        
        # Use asyncio.as_completed for first-success-wins
        for task in asyncio.as_completed(tasks):
            result = await task
            print('got result: ', result)
            if result.get("success"):
                # Cancel remaining tasks
                for remaining_task in tasks:
                    if not remaining_task.done():
                        remaining_task.cancel()
                return result
        
        # If we get here, all failed
        return {"success": False, "message": "All approaches failed"}
    
    async def _run_conjunctive_search(self, parent_state: AgentState, subproblems: List[Dict],
                                    run_config: RunConfig, checkpoint_store) -> List[Dict[str, Any]]:
        """Run all subproblems concurrently (all must succeed)."""
        tasks = [
            self._run_single_subagent(parent_state, subproblem, run_config, checkpoint_store)
            for subproblem in subproblems
        ]
        
        # Wait for all to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to failures
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "name": subproblems[i].get("name", f"subproblem_{i}"),
                    "success": False,
                    "error": str(result)
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    # ── Debug Search Methods ──────────────────────────────────────────────────
    
    async def _run_disjunctive_search_debug(self, parent_state: AgentState, approaches: List[Dict],
                                          run_config: RunConfig, checkpoint_store) -> Dict[str, Any]:
        """DEBUG: Try approaches sequentially with detailed logging."""
        print(f"🔍 DEBUG: Starting disjunctive search with {len(approaches)} approaches (sequential)")
        
        for i, approach in enumerate(approaches):
            print(f"\n📍 DEBUG: Trying approach {i+1}/{len(approaches)}: {approach.get('name', 'unnamed')}")
            print(f"   Description: {approach.get('description', 'no description')}")
            
            result = await self._run_single_subagent(parent_state, approach, run_config, checkpoint_store)
            print(f"   DEBUG Result: {result}")
            
            if result.get("success"):
                print(f"✅ DEBUG: Approach '{approach.get('name')}' succeeded! Stopping search.")
                return result
            else:
                error_msg = result.get('error') or result.get('reason', 'unknown')
                print(f"❌ DEBUG: Approach '{approach.get('name')}' failed: {error_msg}")
                if 'traceback' in result:
                    print(f"   Traceback: {result['traceback']}")
        
        print("💥 DEBUG: All approaches failed in disjunctive search")
        return {"success": False, "message": "All approaches failed"}
    
    async def _run_conjunctive_search_debug(self, parent_state: AgentState, subproblems: List[Dict],
                                          run_config: RunConfig, checkpoint_store) -> List[Dict[str, Any]]:
        """DEBUG: Run all subproblems sequentially with detailed logging."""
        print(f"🔍 DEBUG: Starting conjunctive search with {len(subproblems)} subproblems (sequential)")
        
        results = []
        for i, subproblem in enumerate(subproblems):
            print(f"\n📍 DEBUG: Solving subproblem {i+1}/{len(subproblems)}: {subproblem.get('name', 'unnamed')}")
            print(f"   Description: {subproblem.get('description', 'no description')}")
            
            try:
                result = await self._run_single_subagent(parent_state, subproblem, run_config, checkpoint_store)
                print(f"   DEBUG Result: {result}")
                results.append(result)
                
                if result.get("success"):
                    print(f"✅ DEBUG: Subproblem '{subproblem.get('name')}' succeeded")
                else:
                    error_msg = result.get('error') or result.get('reason', 'unknown')
                    print(f"❌ DEBUG: Subproblem '{subproblem.get('name')}' failed: {error_msg}")
                    if 'traceback' in result:
                        print(f"   Traceback: {result['traceback']}")
                        
            except Exception as e:
                print(f"💥 DEBUG: Exception in subproblem '{subproblem.get('name')}': {str(e)}")
                results.append({
                    "name": subproblem.get("name", f"subproblem_{i}"),
                    "success": False,
                    "error": str(e)
                })
        
        success_count = sum(1 for r in results if r.get("success", False))
        print(f"\n🎯 DEBUG: Conjunctive search complete: {success_count}/{len(results)} succeeded")
        
        return results

# ── Demo Usage ─────────────────────────────────────────────────────────────────

# TODO: Add more comprehensive demos:
# TODO: - Multi-level search (search within search)
# TODO: - Different context passing strategies comparison
# TODO: - Performance/cost analysis of search vs linear approaches
# TODO: - Integration with other environments beyond calculator

async def demo_quadratic_solver():
    """Demo: Solve x^2 + 5x + 6 = 0 using search-enabled calculator."""
    from rollouts import FileCheckpointStore
    
    # Create search-enabled calculator
    calculator_env = CalculatorEnvironment()
    search_config = create_search_config(
        context_passer_name="default",  # Choose context passing strategy by name
        max_depth=2,
        autonomous_subagents=True,
        # debug_sequential=True,
    )
    search_env = SearchEnvironment(calculator_env, search_config)
    
    # Set up initial state
    sys_msg = Message(
        role="system",
        content="""You are a mathematical problem solver with advanced capabilities. You have access to:

Basic calculator operations:
- add(value): Add a number to current value
- subtract(value): Subtract a number from current value  
- multiply(value): Multiply current value by a number
- divide(value): Divide current value by a number
- clear(): Reset current value to zero
- complete_task(summary): Mark task as complete with summary

Advanced search operations:
- branch(approaches): Try different solution methods (only one needs to succeed)
- decompose(subproblems): Break problem into parts (all parts must be solved)

For quadratic equations like x² + 5x + 6 = 0, you could:
1. Use 'branch' to try different methods: factoring, quadratic formula, completing the square
2. Use 'decompose' to break into steps: find discriminant, calculate roots, verify solutions

Be systematic and verify your answers."""
    )
    user_msg = Message(
        role="user", 
        content="Solve the quadratic equation x² + 5x + 6 = 0. Find all real solutions and verify them."
    )
    
    trajectory = Trajectory(messages=[sys_msg, user_msg])
    endpoint = Endpoint(
        provider="anthropic",
        model="claude-4-sonnet-20250514",
        api_base="https://api.anthropic.com"
    )
    actor = Actor(trajectory=trajectory, endpoint=endpoint)
    
    initial_state = AgentState(
        actor=actor,
        environment=search_env,
        turn_idx=0,
        max_turns=25
    )
    
    # Set up run config
    checkpoint_store = FileCheckpointStore("/tmp/search-demo-checkpoints")
    run_config = RunConfig(
        on_chunk=stdout_handler,
        confirm_tool=confirm_tool_with_feedback,
        handle_tool_error=handle_tool_error,
        handle_stop=handle_stop_max_turns,
        handle_no_tool=inject_tool_reminder,  
        checkpoint_store=checkpoint_store
    )
    
    print("🚀 Starting quadratic equation solver with search capabilities...")
    print("📋 Problem: x² + 5x + 6 = 0")
    print("🔍 The agent can use 'branch' and 'decompose' to explore different solution strategies")
    print("─" * 80)
    
    # Run the demo
    states = await run_agent(initial_state, run_config)
    
    # Show final results
    final_state = states[-1]
    print("\n" + "="*80)
    print("📊 DEMO RESULTS")
    print("="*80)
    
    if final_state.stop == StopReason.TASK_COMPLETED:
        print("✅ Task completed successfully!")
    elif final_state.stop == StopReason.MAX_TURNS:
        print("⏰ Reached maximum turns")
    else:
        print(f"🛑 Stopped due to: {final_state.stop}")
    
    print(f"🔢 Final calculator value: {final_state.environment.inner_env.current_value}")
    print(f"💬 Total conversation turns: {final_state.turn_idx}")
    print(f"📝 Total messages: {len(final_state.actor.trajectory.messages)}")
    
    # Count tool usage
    tool_calls = sum(len(msg.tool_calls) for msg in final_state.actor.trajectory.messages)
    search_calls = sum(1 for msg in final_state.actor.trajectory.messages 
                      for tc in msg.tool_calls 
                      if tc.name in ['branch', 'decompose'])
    
    print(f"🔧 Total tool calls: {tool_calls}")
    print(f"🌳 Search operations used: {search_calls}")
    
    return states

# ── Calculator Environment ────────────────────────────────────────────────────

class CalculatorEnvironment(Environment):
    def __init__(self, current_value: float = 0.0):
        self.current_value = current_value
    
    async def serialize(self) -> dict:
        return {"current_value": self.current_value}
    
    @staticmethod
    async def deserialize(data: dict) -> 'CalculatorEnvironment':
        return CalculatorEnvironment(current_value=data["current_value"])
    
    def get_tools(self) -> List[Tool]:
        return [
            Tool(
                type="function",
                function=ToolFunction(
                    name="add",
                    description="Add a number to the current value",
                    parameters=ToolFunctionParameter(
                        type="object",
                        properties={"value": {"type": "number", "description": "Number to add"}}
                    ),
                    required=["value"]
                )
            ),
            Tool(
                type="function",
                function=ToolFunction(
                    name="subtract",
                    description="Subtract a number from the current value",
                    parameters=ToolFunctionParameter(
                        type="object",
                        properties={"value": {"type": "number", "description": "Number to subtract"}}
                    ),
                    required=["value"]
                )
            ),
            Tool(
                type="function",
                function=ToolFunction(
                    name="multiply",
                    description="Multiply the current value by a number",
                    parameters=ToolFunctionParameter(
                        type="object",
                        properties={"value": {"type": "number", "description": "Number to multiply by"}}
                    ),
                    required=["value"]
                )
            ),
            Tool(
                type="function",
                function=ToolFunction(
                    name="divide",
                    description="Divide the current value by a number",
                    parameters=ToolFunctionParameter(
                        type="object",
                        properties={"value": {"type": "number", "description": "Number to divide by"}}
                    ),
                    required=["value"]
                )
            ),
            Tool(
                type="function",
                function=ToolFunction(
                    name="clear",
                    description="Reset the current value to zero",
                    parameters=ToolFunctionParameter(
                        type="object",
                        properties={}
                    ),
                    required=[]
                )
            ),
            Tool(
                type="function",
                function=ToolFunction(
                    name="complete_task",
                    description="Signal that the calculation task is complete",
                    parameters=ToolFunctionParameter(
                        type="object",
                        properties={
                            "summary": {"type": "string", "description": "Summary of calculations performed"},
                            "final_result": {"type": "number", "description": "Final calculation result"}
                        }
                    ),
                    required=["summary"]
                )
            ),
        ]
    
    def requires_confirmation(self, tool_call: ToolCall) -> bool:
        # e.g. only confirm "divide" calls:
        return tool_call.name == "divide"

    async def exec_tool(self, tool_call: ToolCall, current_state: 'AgentState',
                       run_config: 'RunConfig', checkpoint_store = None) -> ToolResult:
        """Execute tool call, mutating environment state"""
        try:
            if tool_call.name == "add":
                value = tool_call.args.get("value")
                if value is None:
                    return ToolResult(
                        call_id=tool_call.id,
                        ok=False,
                        content="",
                        error="Missing required parameter 'value'"
                    )
                self.current_value += value
                return ToolResult(
                    call_id=tool_call.id,
                    ok=True,
                    content=f"Added {value}. Current value: {self.current_value}"
                )
            
            elif tool_call.name == "subtract":
                value = tool_call.args.get("value")
                if value is None:
                    return ToolResult(
                        call_id=tool_call.id,
                        ok=False,
                        content="",
                        error="Missing required parameter 'value'"
                    )
                self.current_value -= value
                return ToolResult(
                    call_id=tool_call.id,
                    ok=True,
                    content=f"Subtracted {value}. Current value: {self.current_value}"
                )
            
            elif tool_call.name == "multiply":
                value = tool_call.args.get("value")
                if value is None:
                    return ToolResult(
                        call_id=tool_call.id,
                        ok=False,
                        content="",
                        error="Missing required parameter 'value'"
                    )
                self.current_value *= value
                return ToolResult(
                    call_id=tool_call.id,
                    ok=True,
                    content=f"Multiplied by {value}. Current value: {self.current_value}"
                )
            
            elif tool_call.name == "divide":
                value = tool_call.args.get("value")
                if value is None:
                    return ToolResult(
                        call_id=tool_call.id,
                        ok=False,
                        content="",
                        error="Missing required parameter 'value'"
                    )
                if value == 0:
                    return ToolResult(
                        call_id=tool_call.id,
                        ok=False,
                        content="",
                        error="Cannot divide by zero"
                    )
                self.current_value /= value
                return ToolResult(
                    call_id=tool_call.id,
                    ok=True,
                    content=f"Divided by {value}. Current value: {self.current_value}"
                )
            
            elif tool_call.name == "clear":
                self.current_value = 0.0
                return ToolResult(
                    call_id=tool_call.id,
                    ok=True,
                    content="Reset to zero. Current value: 0.0"
                )
            
            elif tool_call.name == "complete_task":
                summary = tool_call.args.get("summary", "Task completed")
                final_result = tool_call.args.get("final_result", self.current_value)
                
                return ToolResult(
                    call_id=tool_call.id,
                    ok=True,
                    content=f"Calculation task completed: {summary}. Final result: {final_result}",
                    stop_reason=StopReason.TASK_COMPLETED  # This will stop the agent!
                )
            
            else:
                return ToolResult(
                    call_id=tool_call.id,
                    ok=False,
                    content="",
                    error=f"Unknown operation: {tool_call.name}"
                )
                
        except Exception as e:
            return ToolResult(
                call_id=tool_call.id,
                ok=False,
                content="",
                error=str(e)
            )

if __name__ == "__main__":
    asyncio.run(demo_quadratic_solver())
