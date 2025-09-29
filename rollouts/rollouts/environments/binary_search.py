import asyncio
import argparse
from typing import List

from ..dtypes import (
    Message, Trajectory, Endpoint, Actor, AgentState, Environment,
    Tool, ToolFunction, ToolFunctionParameter, ToolCall, ToolResult, StopReason,
    RunConfig,
)
from ..checkpoints import FileCheckpointStore
from ..agents import (
    stdout_handler,
    confirm_tool_with_feedback,
    handle_tool_error,
    inject_turn_warning,  
    handle_stop_max_turns,
    inject_tool_reminder,
    run_agent,
)

class BinarySearchEnvironment(Environment):
    def __init__(
            self, range_min:int=0, range_max:int=7, 
            space_size: int=8, answer: int=0,
            _turns = 0, _correct = False
        ):
        self.range_min:  int = range_min
        self.range_max:  int = range_max
        self.answer:     int = answer
        self.space_size: int = space_size

        # managed during runtime
        self._turns: int = _turns
        self._correct: bool = _correct

        assert abs(range_min - range_max)+1 == space_size, f"[{range_min},{range_max}] is not {space_size}"
        assert (answer >= range_min) & (answer <= range_max)
    
    async def serialize(self):
        return {k: v for k, v in self.__dict__.items()}
    
    @staticmethod
    async def deserialize(data: dict) -> 'BinarySearchEnvironment':
        return BinarySearchEnvironment(**data)
    
    def get_tools(self) -> List[Tool]:
        return [
            Tool(function=ToolFunction(
                name="guess_answer",
                description = "Guess the hidden number. You'll be told if your guess is too high or too low.",
                parameters = ToolFunctionParameter(
                    properties={"number": {"type": "number", "description": "Your guess"}},
                ),
                required = ["number"],
            ))
        ]
    
    def requires_confirmation(self, tool_call: ToolCall) -> bool: return False

    async def exec_tool(self, tool_call: ToolCall) -> ToolResult:
        try:
            if tool_call.name == "guess_answer":
                guess = int(tool_call.args["number"])
                self._correct = (guess == self.answer)
                if self._correct:
                    return ToolResult(
                        call_id = tool_call.id, ok=True,
                        content = f"CONGRATS!!!! {guess} is correct!",
                        stop_reason=StopReason.TASK_COMPLETED
                    )
                else:
                    hint = "too high" if guess > self.answer else "too low"
                    return ToolResult(
                        call_id = tool_call.id, ok=True,
                        content = f"Wrong! {guess} is {hint}. Try again!"
                    )
            else:
                return ToolResult(
                    call_id=tool_call.id, ok=False,
                    content=f"{tool_call.name} is not a valid tool",
                )
        except Exception as e:
            return ToolResult(
                call_id=tool_call.id, ok=False,
                content="",
                error=str(e)
            )


async def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Calculator Agent')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--checkpoint-dir', default='/tmp/agent-checkpoints', help='Checkpoint directory')
    parser.add_argument('--model', default='gpt-4o-mini', help='Model to use')
    parser.add_argument('--no-checkpoint', action='store_true', help='Disable checkpointing')
    args = parser.parse_args()
    
    # Create checkpoint store (might not use it)
    environment_registry = {
        "BinarySearchEnvironment": BinarySearchEnvironment,
    }
    checkpoint_store = FileCheckpointStore(
        environment_registry=environment_registry,
        directory=args.checkpoint_dir
    ) if not args.no_checkpoint else None
    
    # Create run config (same for both paths)
    run_config = RunConfig(
        on_chunk=stdout_handler,
        confirm_tool=confirm_tool_with_feedback, #type: ignore
        handle_tool_error=handle_tool_error,
        handle_stop=handle_stop_max_turns,
        handle_no_tool=inject_tool_reminder,
        on_step_start=inject_turn_warning,  # Use on_step_start for turn warnings
        checkpoint_store=checkpoint_store
    )
        
    # Create the initial environment
    environment = BinarySearchEnvironment()

    # Create the initial actor state
    sys_msg = Message(
        role="system",
        content=f"You are helpful tool use agent. Your job is to guess a number in the range {environment.range_min} and {environment.range_max} inclusive.",
        tool_calls=[],
        tool_call_id=None
    )
    user_msg = Message(
        role="user",
        content="I'll take a backset while you do this task. Have fun!",
        tool_calls=[],
        tool_call_id=None
    )
    
    trajectory = Trajectory(messages=[sys_msg, user_msg])
    endpoint = Endpoint(
        provider="openai",
        model=args.model,
        api_base="https://api.openai.com/v1"
    )
    actor = Actor(trajectory=trajectory, endpoint=endpoint)
    environment = BinarySearchEnvironment()
    
    # Create the initial agent state
    initial_state = AgentState(
        actor=actor,
        environment=environment,
        turn_idx=0,
        max_turns=10,
    )
    
    states = await run_agent(initial_state, run_config)
    
    print("\n" + "="*80)
    print("📊 Conversation Summary")
    print("="*80)
    
    final_state = states[-1]
    print(f"✅ Total turns: {final_state.turn_idx}")
    print(f"💬 Total messages: {len(final_state.actor.trajectory.messages)}")
    
    # Count tool calls
    tool_calls = sum(len(msg.tool_calls) for msg in final_state.actor.trajectory.messages)
    print(f"🔧 Total tool calls: {tool_calls}")

    # (optionally) save trajectory to disk
    Trajectory.save_jsonl([final_state.actor.trajectory], "./trajectory.jsonl") 
    if final_state.stop:
        print(f"🛑 Stopped due to: {final_state.stop.value}")
    
    if checkpoint_store:
        print(f"\n💾 Session checkpoints saved in: {args.checkpoint_dir}")


if __name__ == "__main__":
    asyncio.run(main())
