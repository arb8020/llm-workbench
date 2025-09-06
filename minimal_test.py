#!/usr/bin/env python3
"""
Minimal test script that only tests datasets import without heavy dependencies
"""

def test_minimal():
    print("Testing minimal datasets import...")
    try:
        from datasets import load_dataset
        print("✅ datasets import successful")
        return True
    except Exception as e:
        print(f"❌ datasets import failed: {e}")
        return False

if __name__ == "__main__":
    test_minimal()