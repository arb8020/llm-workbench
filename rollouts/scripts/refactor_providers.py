#!/usr/bin/env python3
"""
Script to copy function implementations from agents.py to providers.py.

This ensures exact copying without manual errors.
"""

import ast
import re
from typing import Dict, List, Optional


def extract_function_from_file(file_path: str, function_name: str) -> Optional[str]:
    """Extract a complete function definition from a Python file."""
    with open(file_path, 'r') as f:
        content = f.read()

    # Parse the AST to find the function
    tree = ast.parse(content)

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            # Get the line numbers
            start_line = node.lineno - 1  # 0-indexed
            end_line = node.end_lineno if hasattr(node, 'end_lineno') else None

            lines = content.split('\n')

            if end_line:
                # Modern Python with end_lineno
                function_lines = lines[start_line:end_line]
            else:
                # Fall back to heuristic: find next function or class
                function_lines = []
                current_line = start_line

                # Get the function definition line
                function_lines.append(lines[current_line])
                current_line += 1

                # Get the rest based on indentation
                base_indent = len(lines[start_line]) - len(lines[start_line].lstrip())

                while current_line < len(lines):
                    line = lines[current_line]

                    # Empty line or comment - include it
                    if not line.strip() or line.strip().startswith('#'):
                        function_lines.append(line)
                        current_line += 1
                        continue

                    # Check indentation
                    line_indent = len(line) - len(line.lstrip())

                    # If we hit something at the same or less indentation, we're done
                    if line.strip() and line_indent <= base_indent:
                        break

                    function_lines.append(line)
                    current_line += 1

            return '\n'.join(function_lines)

    return None


def replace_function_in_file(file_path: str, function_name: str, new_implementation: str):
    """Replace a function stub with the actual implementation."""
    with open(file_path, 'r') as f:
        content = f.read()

    # Find the function stub and replace it
    pattern = rf'(def {re.escape(function_name)}\([^)]*\)[^:]*:.*?)(\n    # TODO: Implement\n    pass)'

    def replacement(match):
        signature = match.group(1)
        # Extract just the function body (without the def line)
        impl_lines = new_implementation.split('\n')
        body_lines = []
        found_def = False

        for line in impl_lines:
            if line.strip().startswith('def ') and function_name in line:
                found_def = True
                continue
            if found_def:
                body_lines.append(line)

        body = '\n'.join(body_lines)
        return signature + '\n' + body

    new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)

    with open(file_path, 'w') as f:
        f.write(new_content)

    return new_content != content  # Return True if something was changed


def count_functions_in_file(file_path: str, function_names: List[str]) -> Dict[str, bool]:
    """Check which functions exist in a file."""
    with open(file_path, 'r') as f:
        content = f.read()

    tree = ast.parse(content)
    found_functions = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            found_functions.add(node.name)

    return {name: name in found_functions for name in function_names}


def normalize_function_body(func_code: str) -> str:
    """Normalize function code for comparison (remove whitespace differences)."""
    # Remove the function signature line
    lines = func_code.split('\n')
    body_lines = []
    found_def = False

    for line in lines:
        if line.strip().startswith('def ') and not found_def:
            found_def = True
            continue
        if found_def:
            body_lines.append(line.rstrip())  # Remove trailing whitespace

    # Remove empty lines at start and end
    while body_lines and not body_lines[0].strip():
        body_lines.pop(0)
    while body_lines and not body_lines[-1].strip():
        body_lines.pop()

    return '\n'.join(body_lines)


def verify_exact_copy(agents_file: str, providers_file: str, function_names: List[str]):
    """Verify that the exact function implementations were copied."""
    print("\n🔍 Verifying exact code copy...")

    all_exact = True

    for func_name in function_names:
        # Extract from both files
        agents_impl = extract_function_from_file(agents_file, func_name)
        providers_impl = extract_function_from_file(providers_file, func_name)

        if agents_impl is None:
            print(f"  ❌ {func_name}: Not found in agents.py")
            all_exact = False
            continue

        if providers_impl is None:
            print(f"  ❌ {func_name}: Not found in providers.py")
            all_exact = False
            continue

        # Normalize and compare function bodies
        agents_body = normalize_function_body(agents_impl)
        providers_body = normalize_function_body(providers_impl)

        if agents_body == providers_body:
            print(f"  ✅ {func_name}: Exact match")
        else:
            print(f"  ❌ {func_name}: Code differs!")
            agents_lines = agents_body.split('\n')
            providers_lines = providers_body.split('\n')
            print(f"    Agents lines: {len(agents_lines)}")
            print(f"    Providers lines: {len(providers_lines)}")
            all_exact = False

    return all_exact


def verify_refactor(agents_file: str, providers_file: str, function_names: List[str]):
    """Verify the refactor was successful."""
    print("\n🔍 Verifying refactor...")

    # Basic existence check
    agents_functions = count_functions_in_file(agents_file, function_names)
    providers_functions = count_functions_in_file(providers_file, function_names)

    print(f"\n📊 Function existence check:")
    for func_name in function_names:
        agents_has = agents_functions.get(func_name, False)
        providers_has = providers_functions.get(func_name, False)

        status = "✅" if providers_has else "❌"
        print(f"  {status} {func_name}: agents.py={agents_has}, providers.py={providers_has}")

    all_copied = all(providers_functions.values())

    # Exact code comparison
    exact_match = verify_exact_copy(agents_file, providers_file, function_names)

    print(f"\n🎯 Results:")
    print(f"  Functions copied: {'✅' if all_copied else '❌'}")
    print(f"  Exact code match: {'✅' if exact_match else '❌'}")

    return all_copied and exact_match


def main():
    """Main refactor script."""
    agents_file = "rollouts/agents.py"
    providers_file = "rollouts/providers.py"

    # Functions to copy from agents.py to providers.py
    functions_to_copy = [
        # Utility functions
        "add_cache_control_to_last_content",
        "verbose",

        # Message/Tool conversion functions
        "_message_to_openai",
        "_message_to_anthropic",
        "_tool_to_openai",
        "_tool_to_anthropic",
        "_parse_usage",
        "_parse_completion",
        "_apply_inline_thinking_template",

        # Stream aggregation functions
        "aggregate_stream",
        "aggregate_anthropic_stream",

        # Provider rollout functions
        "rollout_openai",
        "rollout_moonshot",
        "rollout_vllm",
        "rollout_anthropic",
    ]

    print("🚀 Starting refactor: copying functions from agents.py to providers.py")

    copied = 0
    failed = 0

    for func_name in functions_to_copy:
        print(f"📋 Processing {func_name}...")

        # Extract from agents.py
        implementation = extract_function_from_file(agents_file, func_name)

        if implementation is None:
            print(f"  ❌ Could not find {func_name} in {agents_file}")
            failed += 1
            continue

        # Replace in providers.py
        success = replace_function_in_file(providers_file, func_name, implementation)

        if success:
            print(f"  ✅ Copied {func_name}")
            copied += 1
        else:
            print(f"  ⚠️  Could not replace {func_name} stub in {providers_file}")
            failed += 1

    print(f"\n🎉 Refactor complete!")
    print(f"✅ Successfully copied: {copied} functions")
    print(f"❌ Failed: {failed} functions")

    # Verify the refactor
    verification_passed = verify_refactor(agents_file, providers_file, functions_to_copy)

    if failed == 0 and verification_passed:
        print("\n🔥 All functions copied successfully!")
        print("Next steps:")
        print("1. Run: git diff rollouts/providers.py  # to see what changed")
        print("2. Test: python examples/simple_calculator.py")
        print("3. Test: python examples/search_calculator_demo.py")
        print("4. If tests pass, remove functions from agents.py")
    else:
        print(f"\n⚠️  Issues detected. Check the output above.")


if __name__ == "__main__":
    main()