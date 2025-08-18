#!/usr/bin/env python3
"""Integration test for the broker/bifrost three-operation model.

This test validates that the push/exec/deploy operations work correctly
with real GPU infrastructure.
"""

import sys
import json
import shutil
from pathlib import Path

# Add package directories to Python path for testing
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "bifrost"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from broker import GPUClient
from broker.types import CloudType
from bifrost import BifrostClient, JobStatus


def create_test_script() -> Path:
    """Create a simple test script for validation."""
    test_file = Path(__file__).parent / "test_data" / "three_operation_test.py"
    test_file.parent.mkdir(exist_ok=True)
    
    script_content = '''#!/usr/bin/env python3
import json
from datetime import datetime
from pathlib import Path

print("🧪 Three-operation integration test")
print(f"Start time: {datetime.now()}")

# Create output directory
output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

# Test data
test_data = {
    "test_type": "three_operation_integration",
    "timestamp": datetime.now().isoformat(),
    "operations_tested": ["push", "exec", "deploy"],
    "status": "success"
}

output_file = output_dir / "integration_test.json"
output_file.write_text(json.dumps(test_data, indent=2))

print("✅ Integration test completed successfully")
print(f"Created output: {output_file}")
'''
    
    test_file.write_text(script_content)
    return test_file


def test_three_operations_with_gpu():
    """Test push, exec, deploy operations with real GPU instance."""
    print("🧪 Testing Three-Operation Model Integration")
    print("=" * 60)
    
    # Find existing running instance
    broker_client = GPUClient()
    instances = broker_client.list_instances()
    running_instances = [inst for inst in instances 
                        if inst.status.name == 'RUNNING' and inst.public_ip and inst.ssh_port]
    
    if not running_instances:
        print("❌ No running instances available for testing")
        print("Please start an instance first with:")
        print("  python scripts/test_integration_phase2_working.py --keep-alive")
        return False
    
    instance = running_instances[0]
    ssh_conn = instance.ssh_connection_string()
    print(f"✅ Using instance: {instance.id} - {ssh_conn}")
    
    # Create test data
    test_script = create_test_script()
    print(f"✅ Created test script: {test_script}")
    
    try:
        # Initialize Bifrost client
        bifrost = BifrostClient(ssh_conn)
        
        # Test 1: Push operation
        print("\n1️⃣ Testing PUSH operation...")
        worktree_path = bifrost.push("integration_test")
        assert worktree_path.endswith("integration_test"), "Custom target_dir not working"
        print(f"✅ Push successful: {worktree_path}")
        
        # Test 2: Exec operation
        print("\n2️⃣ Testing EXEC operation...")
        result = bifrost.exec(
            f"python {test_script.relative_to(Path.cwd())}", 
            worktree=worktree_path
        )
        assert "Integration test completed successfully" in result, "Exec operation failed"
        print("✅ Exec successful")
        
        # Test 3: Deploy operation (convenience)
        print("\n3️⃣ Testing DEPLOY operation...")
        result = bifrost.deploy("echo 'Deploy integration test successful'")
        assert "Deploy integration test successful" in result, "Deploy operation failed"
        print("✅ Deploy successful")
        
        # Test 4: Environment variables
        print("\n4️⃣ Testing environment variables...")
        env_vars = {"TEST_TYPE": "integration", "VERSION": "1.0"}
        result = bifrost.exec("echo $TEST_TYPE-$VERSION", env=env_vars, worktree=worktree_path)
        assert "integration-1.0" in result, "Environment variables not working"
        print("✅ Environment variables successful")
        
        # Test 5: File transfer
        print("\n5️⃣ Testing file transfer...")
        local_dir = Path(__file__).parent / "test_outputs"
        local_dir.mkdir(exist_ok=True)
        
        copy_result = bifrost.copy_files(f"{worktree_path}/outputs", str(local_dir), recursive=True)
        assert copy_result.success, f"File copy failed: {copy_result.error_message}"
        
        # Validate copied files
        test_file = local_dir / "integration_test.json"
        assert test_file.exists(), "Test output file not copied"
        
        test_data = json.loads(test_file.read_text())
        assert test_data["test_type"] == "three_operation_integration", "Invalid test data"
        print("✅ File transfer successful")
        
        print("\n🎉 ALL THREE-OPERATION TESTS PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup
        try:
            bifrost.close()
        except:
            pass


def cleanup_test_files():
    """Clean up test artifacts."""
    test_dirs = [
        Path(__file__).parent / "test_data",
        Path(__file__).parent / "test_outputs"
    ]
    
    for test_dir in test_dirs:
        if test_dir.exists():
            shutil.rmtree(test_dir)
            print(f"✅ Cleaned up {test_dir}")


if __name__ == "__main__":
    try:
        success = test_three_operations_with_gpu()
        if success:
            print("\n🎉 THREE-OPERATION INTEGRATION TEST PASSED")
        else:
            print("\n💥 THREE-OPERATION INTEGRATION TEST FAILED")
            sys.exit(1)
    finally:
        cleanup_test_files()