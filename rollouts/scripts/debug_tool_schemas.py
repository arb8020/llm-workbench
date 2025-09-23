#!/usr/bin/env python3
"""Debug tool schemas"""
import json
from rollouts import CalculatorEnvironment
from rollouts.providers import _tool_to_openai

def main():
    env = CalculatorEnvironment()
    tools = env.get_tools()

    print("=== ORIGINAL TOOLS ===")
    for tool in tools:
        print(f"Tool: {tool.function.name}")
        print(f"  Required: {tool.function.required}")
        print(f"  Properties: {list(tool.function.parameters.properties.keys())}")
        print()

    print("=== CONVERTED TO OPENAI FORMAT ===")
    openai_tools = [_tool_to_openai(t) for t in tools]
    for tool in openai_tools:
        print(f"Tool: {tool['function']['name']}")
        print(f"  Required: {tool['function']['parameters']['required']}")
        print(f"  Properties: {list(tool['function']['parameters']['properties'].keys())}")
        print(f"  Full schema: {json.dumps(tool, indent=2)}")
        print()

if __name__ == "__main__":
    main()