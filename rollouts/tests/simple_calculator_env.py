"""
Simple calculator environment for Gemini testing
"""

from typing import List
from rollouts.dtypes import (
    Tool, ToolFunction, ToolFunctionParameter, 
    ToolCall, ToolResult, Environment, AgentState, RunConfig
)


class SimpleCalculatorEnvironment(Environment):
    """Very simple calculator with just basic arithmetic functions"""
    
    def __init__(self):
        self.result = 0
    
    async def serialize(self) -> dict:
        return {"result": self.result}
    
    @staticmethod
    async def deserialize(data: dict) -> 'SimpleCalculatorEnvironment':
        env = SimpleCalculatorEnvironment()
        env.result = data.get("result", 0)
        return env
    
    def get_tools(self) -> List[Tool]:
        return [
            Tool(
                type="function",
                function=ToolFunction(
                    name="calculate",
                    description="Calculate a simple arithmetic expression like '2 + 3' or '10 * 5'",
                    parameters=ToolFunctionParameter(
                        type="object",
                        properties={
                            "expression": {
                                "type": "string", 
                                "description": "The arithmetic expression to evaluate, e.g. '15 + 27'"
                            }
                        }
                    ),
                    required=["expression"]
                )
            )
        ]
    
    async def exec_tool(self, tool_call: ToolCall, current_state: 'AgentState',
                       run_config: 'RunConfig', checkpoint_store=None) -> ToolResult:
        """Execute the calculator tool"""
        
        if tool_call.name == "calculate":
            try:
                expression = tool_call.args.get("expression", "")
                
                # Simple and safe expression evaluation
                # Only allow basic arithmetic operations
                allowed_chars = set("0123456789+-*/(). ")
                if not all(c in allowed_chars for c in expression):
                    return ToolResult(
                        call_id=tool_call.id,
                        ok=False,
                        error=f"Expression contains invalid characters: {expression}"
                    )
                
                # Evaluate the expression safely
                result = eval(expression)  # Note: This is normally unsafe, but OK for testing
                self.result = result
                
                return ToolResult(
                    call_id=tool_call.id,
                    ok=True,
                    content=f"Result: {expression} = {result}"
                )
                
            except Exception as e:
                return ToolResult(
                    call_id=tool_call.id,
                    ok=False,
                    error=f"Error calculating '{expression}': {str(e)}"
                )
        
        return ToolResult(
            call_id=tool_call.id,
            ok=False,
            error=f"Unknown tool: {tool_call.name}"
        )