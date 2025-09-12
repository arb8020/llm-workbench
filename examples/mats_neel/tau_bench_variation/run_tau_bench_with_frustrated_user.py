#!/usr/bin/env python3
"""
Wrapper script to run tau-bench with frustrated user simulation.
This ensures monkey-patching happens before tau-bench runs.
"""

import sys
import subprocess
from examples.mats_neel.tau_bench_variation.worker_experiment import create_emotional_user_variants

def main():
    # Apply monkey-patching first
    print("🔧 Applying frustrated user monkey-patching...")
    success = create_emotional_user_variants()
    if not success:
        print("❌ Failed to apply monkey-patching")
        sys.exit(1)
    print("✅ Monkey-patching applied successfully")
    
    # Run tau-bench with the rest of the arguments
    tau_bench_args = [
        "python", "-m", "tau_bench.run"
    ] + sys.argv[1:]
    
    print(f"🚀 Running tau-bench: {' '.join(tau_bench_args)}")
    result = subprocess.run(tau_bench_args)
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()