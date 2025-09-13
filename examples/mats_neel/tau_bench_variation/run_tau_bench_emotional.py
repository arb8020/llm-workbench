#!/usr/bin/env python3
"""
CLI wrapper to run tau-bench with emotional user simulation variants.
This applies monkey-patching before running tau-bench.

Usage:
    python run_tau_bench_emotional.py --user-strategy frustrated_llm --env retail --model gpt-4o
    python run_tau_bench_emotional.py --user-strategy anxious_llm --env airline --model gpt-4o
"""

import sys
import subprocess
import os

# Add the workspace to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from examples.mats_neel.tau_bench_variation.worker_experiment import create_emotional_user_variants

def main():
    # Check if user is trying to use an emotional strategy
    emotional_strategies = ['frustrated_llm', 'anxious_llm', 'angry_llm', 'confused_llm']
    
    # Parse arguments to see if emotional strategy is being used
    uses_emotional_strategy = False
    for i, arg in enumerate(sys.argv):
        if arg == '--user-strategy' and i + 1 < len(sys.argv):
            if sys.argv[i + 1] in emotional_strategies:
                uses_emotional_strategy = True
                break
    
    # Apply monkey-patching if needed
    if uses_emotional_strategy:
        print("🔧 Applying emotional user simulation monkey-patching...")
        success = create_emotional_user_variants()
        if not success:
            print("❌ Failed to apply monkey-patching")
            sys.exit(1)
        print("✅ Monkey-patching applied successfully")
    
    # Run tau-bench with the CLI arguments
    tau_bench_args = [
        "python", "run.py"
    ] + sys.argv[1:]
    
    print(f"🚀 Running tau-bench: {' '.join(tau_bench_args)}")
    
    # Change to tau-bench directory if it exists
    tau_bench_path = None
    possible_paths = [
        "/usr/local/lib/python3.*/site-packages/tau_bench",
        "~/.local/lib/python3.*/site-packages/tau_bench", 
        "./tau_bench"
    ]
    
    # Try to find tau-bench installation
    import tau_bench
    tau_bench_path = os.path.dirname(tau_bench.__file__)
    
    if tau_bench_path and os.path.exists(os.path.join(tau_bench_path, "run.py")):
        print(f"📂 Found tau-bench at: {tau_bench_path}")
        os.chdir(tau_bench_path)
        tau_bench_args = ["python", "run.py"] + sys.argv[1:]
    else:
        # Fall back to module execution
        tau_bench_args = ["python", "-m", "tau_bench.run"] + sys.argv[1:]
    
    result = subprocess.run(tau_bench_args)
    sys.exit(result.returncode)

if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        print("\nAvailable emotional user strategies:")
        print("  frustrated_llm - Irritated, impatient customer")
        print("  anxious_llm    - Worried customer seeking reassurance") 
        print("  angry_llm      - Angry customer demanding resolution")
        print("  confused_llm   - Confused customer needing simple explanations")
        print("\nFor standard tau-bench help:")
        subprocess.run(["python", "-m", "tau_bench.run", "--help"])
        sys.exit(0)
    
    main()