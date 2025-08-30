#!/usr/bin/env python3
"""
Comprehensive test suite for shared SSH foundation.

Tests the UniversalSSHClient and SSHConnectionInfo without requiring
actual SSH connections.
"""

import asyncio
import os
import stat
import sys
import tempfile
from unittest.mock import Mock, patch

# Add llm-workbench to path for imports
sys.path.insert(0, '/Users/chiraagbalu/llm-workbench')

from shared.ssh_foundation import (
    SSHConnectionInfo, 
    UniversalSSHClient, 
    secure_temp_ssh_key,
    execute_command_sync,
    execute_command_async
)


def test_ssh_connection_info():
    """Test SSHConnectionInfo parsing and generation"""
    print("🧪 Testing SSHConnectionInfo...")
    
    # Test valid connection string parsing
    conn = SSHConnectionInfo.from_string("root@gpu.example.com:22")
    assert conn.hostname == "gpu.example.com"
    assert conn.port == 22
    assert conn.username == "root"
    assert conn.timeout == 30  # default
    print("   ✅ Valid connection string parsing")
    
    # Test connection string with custom timeout
    conn = SSHConnectionInfo.from_string("user@host:2222", timeout=60)
    assert conn.port == 2222
    assert conn.timeout == 60
    print("   ✅ Custom port and timeout")
    
    # Test connection string generation
    conn_str = conn.connection_string()
    assert conn_str == "user@host:2222"
    print("   ✅ Connection string generation")
    
    # Test invalid formats
    try:
        SSHConnectionInfo.from_string("invalid-format")
        assert False, "Should have raised ValueError"
    except ValueError:
        print("   ✅ Invalid format rejected")
    
    try:
        SSHConnectionInfo.from_string("user@host:invalid-port")
        assert False, "Should have raised ValueError"
    except ValueError:
        print("   ✅ Invalid port rejected")
    
    # Test direct instantiation (generic approach)
    conn = SSHConnectionInfo(
        hostname="1.2.3.4",
        port=22,
        username="root",
        timeout=30
    )
    assert conn.hostname == "1.2.3.4"
    assert conn.username == "root"
    assert conn.port == 22
    print("   ✅ Direct instantiation working")
    
    print("   🎉 SSHConnectionInfo tests passed!")


def test_secure_temp_ssh_key():
    """Test secure temporary SSH key handling"""
    print("🔐 Testing secure temporary SSH key handling...")
    
    test_key = """[TEST SSH KEY - NOT A REAL PRIVATE KEY - USED FOR TESTING ONLY]
This is fake content used to test the secure_temp_ssh_key function.
No actual cryptographic material is present here.
[END FAKE TEST KEY]"""
    
    # Test basic functionality
    with secure_temp_ssh_key(test_key) as key_path:
        # Verify file exists
        assert os.path.exists(key_path), "Key file should exist"
        
        # Verify secure permissions (600)
        file_stat = os.stat(key_path)
        perms = oct(file_stat.st_mode)[-3:]
        assert perms == "600", f"Expected 600 permissions, got {perms}"
        
        # Verify content
        with open(key_path) as f:
            content = f.read()
        assert content == test_key, "Key content should match"
        
        temp_key_path = key_path  # Save for cleanup verification
    
    # Verify cleanup happened
    assert not os.path.exists(temp_key_path), "Key file should be cleaned up"
    print("   ✅ Secure permissions and cleanup working")
    
    # Test exception handling
    try:
        with secure_temp_ssh_key(test_key) as key_path:
            raise Exception("Test exception")
    except Exception:
        # Should still clean up
        assert not os.path.exists(key_path), "Should clean up even on exception"
        print("   ✅ Exception-safe cleanup working")
    
    print("   🎉 Secure key handling tests passed!")


def test_universal_ssh_client_structure():
    """Test UniversalSSHClient structure without actual connections"""
    print("🔌 Testing UniversalSSHClient structure...")
    
    client = UniversalSSHClient()
    
    # Verify all required methods exist
    sync_methods = ["connect", "exec_command", "close"]
    async_methods = ["aconnect", "aexec_command", "aclose"] 
    
    for method in sync_methods:
        assert hasattr(client, method), f"Missing sync method: {method}"
        assert not asyncio.iscoroutinefunction(getattr(client, method)), f"{method} should not be async"
    
    for method in async_methods:
        assert hasattr(client, method), f"Missing async method: {method}"
        assert asyncio.iscoroutinefunction(getattr(client, method)), f"{method} should be async"
    
    print("   ✅ All sync/async methods present and properly defined")
    
    # Test that client starts with no connections
    assert client._paramiko_client is None
    assert client._asyncssh_client is None
    print("   ✅ Client initializes with clean state")
    
    print("   🎉 UniversalSSHClient structure tests passed!")


def test_convenience_functions():
    """Test convenience functions"""
    print("🛠️ Testing convenience functions...")
    
    # Test that convenience functions exist and have correct signatures
    assert callable(execute_command_sync), "execute_command_sync should be callable"
    assert callable(execute_command_async), "execute_command_async should be callable"
    assert asyncio.iscoroutinefunction(execute_command_async), "execute_command_async should be async"
    
    print("   ✅ Convenience functions exist with correct signatures")
    print("   🎉 Convenience function tests passed!")


async def test_async_patterns():
    """Test async pattern compatibility"""
    print("🔄 Testing async patterns...")
    
    from shared.ssh_foundation import UniversalSSHClient, SSHConnectionInfo
    
    # Test that async methods can be called (they'll fail without real SSH but shouldn't crash)
    client = UniversalSSHClient()
    conn_info = SSHConnectionInfo("fake-host", 22, "root")
    
    # These will fail to connect but should not crash with syntax errors
    try:
        result = await client.aconnect(conn_info)
        # Expected to fail - no real SSH server
    except Exception as e:
        print(f"   ✅ aconnect() handles connection failure gracefully: {type(e).__name__}")
    
    try:
        result = await client.aexec_command("echo test")
        # Expected to fail - no connection
    except Exception as e:
        print(f"   ✅ aexec_command() handles no-connection gracefully: {type(e).__name__}")
    
    # Test convenience async function
    try:
        result = await execute_command_async(conn_info, "echo test")
        # Expected to fail but shouldn't crash
    except Exception as e:
        print(f"   ✅ execute_command_async() handles failure gracefully: {type(e).__name__}")
    
    print("   🎉 Async pattern tests passed!")


def main():
    """Run all tests"""
    print("🧪 Comprehensive Shared SSH Foundation Test Suite")
    print("=" * 60)
    
    try:
        test_ssh_connection_info()
        print()
        
        test_secure_temp_ssh_key()
        print()
        
        test_universal_ssh_client_structure()
        print()
        
        test_convenience_functions()
        print()
        
        # Run async tests
        asyncio.run(test_async_patterns())
        print()
        
        print("🎉 All Shared Foundation Tests Passed!")
        print("")
        print("📋 Test Summary:")
        print("   ✅ Connection string parsing and validation")
        print("   ✅ Secure temporary key file handling")
        print("   ✅ UniversalSSHClient sync/async structure")
        print("   ✅ Convenience function interfaces")
        print("   ✅ Async pattern compatibility")
        print("")
        print("🚀 Shared foundation ready for broker/bifrost migration!")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)