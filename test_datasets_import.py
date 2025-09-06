#!/usr/bin/env python3
"""
Smoke test to verify datasets + pyarrow compatibility.
This reproduces the exact import chain that's failing in the remote environment.
"""

def test_import():
    try:
        print("Testing datasets import...")
        from datasets import load_dataset
        print("✅ datasets import successful")
        
        # Test basic functionality
        print("Testing basic dataset loading...")
        # Use a tiny dataset for quick testing
        dataset = load_dataset("squad", split="train[:1]")
        print("✅ Basic dataset loading successful")
        
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        return False

if __name__ == "__main__":
    success = test_import()
    if success:
        print("\n🎉 All tests passed! Configuration is working.")
    else:
        print("\n💥 Tests failed. Need to adjust versions.")
    
    # Also show current versions
    try:
        import datasets
        import pyarrow
        print(f"\nCurrent versions:")
        print(f"  datasets: {datasets.__version__}")
        print(f"  pyarrow: {pyarrow.__version__}")
    except:
        print("\nCouldn't determine current versions")