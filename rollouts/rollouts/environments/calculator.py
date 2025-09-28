import asyncio
from typing import List
from dataclasses import dataclass

from ..dtypes import (
    Tool, ToolFunction, ToolFunctionParameter,
    ToolCall, ToolResult, StopReason, AgentState, RunConfig, Environment
)

@dataclass
class CalculatorEnvironment:
    """Calculator environment with numeric operations."""
    current_value: float = 0.0
    
    async def serialize(self) -> dict:
        return {"current_value": self.current_value}

    @staticmethod
    async def deserialize(data: dict) -> 'CalculatorEnvironment':
        return CalculatorEnvironment(current_value=data["current_value"])

    def requires_confirmation(self, tool_call: ToolCall) -> bool:
        """Calculator tools don't require confirmation by default."""
        return False
    
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
        # e.g. only confirm “divide” calls:
        return tool_call.name == "divide"

    async def exec_tool(self, tool_call: ToolCall, current_state: 'AgentState',
                       run_config: 'RunConfig', checkpoint_store = None) -> ToolResult:
        """Execute tool call, mutating environment state"""
        try:
            if tool_call.name == "add":
                value = tool_call.args["value"]
                self.current_value += value
                return ToolResult(
                    call_id=tool_call.id,
                    ok=True,
                    content=f"Added {value}. Current value: {self.current_value}"
                )
            
            elif tool_call.name == "subtract":
                value = tool_call.args["value"]
                self.current_value -= value
                return ToolResult(
                    call_id=tool_call.id,
                    ok=True,
                    content=f"Subtracted {value}. Current value: {self.current_value}"
                )
            
            elif tool_call.name == "multiply":
                value = tool_call.args["value"]
                self.current_value *= value
                return ToolResult(
                    call_id=tool_call.id,
                    ok=True,
                    content=f"Multiplied by {value}. Current value: {self.current_value}"
                )
            
            elif tool_call.name == "divide":
                value = tool_call.args["value"]
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
                summary = tool_call.args["summary"]
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

# ── Entry point ───────────────────────────────────────────────────────────────

async def main():
    """Simple demo main function"""
    print("Use the simple_calculator.py example in examples/ instead!")

if __name__ == "__main__":
    asyncio.run(main())
