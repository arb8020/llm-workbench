"""Minimal hello world test script for bifrost testing.

This script requires minimal compute and is safe for frequent testing.
"""

import sys
import platform
from datetime import datetime

def main():
    print("🌍 Hello from remote GPU!")
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    # Check if we're on a GPU-enabled system (without requiring GPU libraries)
    try:
        import os
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')
        print(f"CUDA_VISIBLE_DEVICES: {cuda_visible}")
    except Exception as e:
        print(f"CUDA check failed: {e}")
    
    print("✅ Hello world completed successfully!")

if __name__ == "__main__":
    main()