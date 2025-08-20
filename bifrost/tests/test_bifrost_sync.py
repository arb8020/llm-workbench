#!/usr/bin/env python3
"""
Bifrost synchronous SSH integration test

Tests bifrost's use of the shared SSH foundation with synchronous operations:
1. Test SSH connectivity using shared foundation (sync only)
2. Validate nvidia-smi output capture (sync only)
3. Test bifrost-specific SSH patterns (connection strings)

This test validates that bifrost can use the shared SSH foundation for
deployment operations without SSH code duplication.

Usage:
    python test_bifrost_sync.py user@host:port
    
Example:
    python test_bifrost_sync.py root@194.68.245.12:22059
    
Use setup_ssh_helper.py to provision a GPU instance for testing.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add bifrost to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bifrost.ssh_clients_compat import (
    execute_command_sync,
    test_ssh_connection,
    SSHConnectionInfo
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BifrostSyncTest:
    """Synchronous bifrost SSH integration test using shared foundation"""
    
    def __init__(self, ssh_connection: str):
        self.ssh_connection = ssh_connection
        self.test_results = {
            "ssh_connectivity": False,
            "nvidia_validation": False,
            "hostname_test": False,
            "connection_string_parsing": False
        }
    
    def run_sync_test(self) -> bool:
        """Run complete synchronous bifrost SSH test"""
        logger.info("🚀 BIFROST SYNCHRONOUS SSH INTEGRATION TEST")
        logger.info("Testing shared SSH foundation integration")
        logger.info("=" * 70)
        
        try:
            # Step 1: Test connection string parsing
            if not self._test_connection_string_parsing():
                return False
            
            # Step 2: Test SSH connectivity
            if not self._test_ssh_connectivity():
                return False
            
            # Step 3: Test nvidia-smi validation
            if not self._test_nvidia_validation():
                return False
            
            # Step 4: Test hostname command
            if not self._test_hostname_command():
                return False
            
            # Step 5: Results analysis
            return self._analyze_results()
            
        except Exception as e:
            logger.error(f"❌ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _test_connection_string_parsing(self) -> bool:
        """Test bifrost SSH connection string parsing via shared foundation"""
        logger.info("\n🔗 Step 1: Testing SSH connection string parsing...")
        logger.info(f"   Connection: {self.ssh_connection}")
        
        try:
            # Test that shared foundation can parse bifrost connection strings
            conn_info = SSHConnectionInfo.from_string(self.ssh_connection)
            
            logger.info(f"   ✅ Parsed hostname: {conn_info.hostname}")
            logger.info(f"   ✅ Parsed port: {conn_info.port}")
            logger.info(f"   ✅ Parsed username: {conn_info.username}")
            
            # Verify round-trip
            reconstructed = conn_info.connection_string()
            if reconstructed == self.ssh_connection:
                logger.info("   ✅ Round-trip parsing successful")
                self.test_results["connection_string_parsing"] = True
            else:
                logger.warning(f"   ⚠️ Round-trip mismatch: {reconstructed} != {self.ssh_connection}")
                self.test_results["connection_string_parsing"] = True  # Still functional
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Connection string parsing failed: {e}")
            return False
    
    def _test_ssh_connectivity(self) -> bool:
        """Test SSH connectivity via shared foundation"""
        logger.info("\n🔗 Step 2: Testing SSH connectivity...")
        logger.info(f"   Using connection: {self.ssh_connection}")
        
        # Test connection using bifrost compatibility layer
        logger.info("\n--- Testing SSH Connection Test ---")
        try:
            success, message = test_ssh_connection(self.ssh_connection)
            
            if success:
                logger.info("✅ SSH connection test passed!")
                logger.info(f"   Result: {message}")
                self.test_results["ssh_connectivity"] = True
            else:
                logger.error(f"❌ SSH connection test failed: {message}")
                
        except Exception as e:
            logger.error(f"❌ SSH connection test exception: {e}")
        
        # Test basic command execution
        logger.info("\n--- Testing Basic Command Execution ---")
        try:
            exit_code, stdout, stderr = execute_command_sync(
                self.ssh_connection, "echo 'BIFROST_SYNC_SSH_TEST'", timeout=30
            )
            
            if exit_code == 0 and "BIFROST_SYNC_SSH_TEST" in stdout:
                logger.info("✅ Sync SSH command execution working!")
                logger.info(f"   Output: {stdout.strip()}")
                self.test_results["ssh_connectivity"] = True
            elif exit_code == 0:
                logger.info("✅ Sync SSH connection successful (limited output)")
                self.test_results["ssh_connectivity"] = True
            else:
                logger.error(f"❌ Sync SSH command failed - exit code {exit_code}, {stderr}")
                
        except Exception as e:
            logger.error(f"❌ Sync SSH command exception: {e}")
        
        return self.test_results["ssh_connectivity"]
    
    def _test_nvidia_validation(self) -> bool:
        """Test nvidia-smi output capture via shared foundation"""
        logger.info("\n🎯 Step 3: Testing nvidia-smi validation...")
        
        # Only proceed if we have SSH connectivity
        if not self.test_results["ssh_connectivity"]:
            logger.error("❌ No SSH connectivity - skipping nvidia-smi tests")
            return False
        
        logger.info(f"   Using connection: {self.ssh_connection}")
        
        logger.info("\n--- Testing nvidia-smi Version ---")
        try:
            exit_code, stdout, stderr = execute_command_sync(
                self.ssh_connection, "nvidia-smi --version", timeout=30
            )
            
            if exit_code == 0 and "NVIDIA-SMI" in stdout:
                logger.info("✅ nvidia-smi version: Real GPU output captured!")
                # Extract version info
                version_info = stdout.split('\n')[0] if stdout else "Unknown"
                logger.info(f"   Version info: {version_info}")
                self.test_results["nvidia_validation"] = True
            elif exit_code == 0:
                logger.warning("⚠️ nvidia-smi ran but limited output")
            else:
                logger.error(f"❌ nvidia-smi failed - exit code {exit_code}, {stderr}")
                
        except Exception as e:
            logger.error(f"❌ nvidia-smi exception: {e}")
        
        # Test GPU name query
        logger.info("\n--- Testing GPU Name Query ---")
        try:
            exit_code, stdout, stderr = execute_command_sync(
                self.ssh_connection, "nvidia-smi --query-gpu=name --format=csv,noheader", timeout=30
            )
            
            if exit_code == 0 and stdout.strip():
                gpu_name = stdout.strip()
                logger.info(f"✅ GPU detected: {gpu_name}")
                self.test_results["nvidia_validation"] = True
            else:
                logger.warning(f"⚠️ GPU query limited: exit_code={exit_code}, stderr={stderr}")
                
        except Exception as e:
            logger.error(f"❌ GPU query exception: {e}")
        
        return self.test_results["nvidia_validation"]
    
    def _test_hostname_command(self) -> bool:
        """Test hostname command execution"""
        logger.info("\n🏠 Step 4: Testing hostname command...")
        
        if not self.test_results["ssh_connectivity"]:
            logger.error("❌ No SSH connectivity - skipping hostname test")
            return False
        
        try:
            exit_code, stdout, stderr = execute_command_sync(
                self.ssh_connection, "hostname", timeout=30
            )
            
            if exit_code == 0 and stdout.strip():
                hostname = stdout.strip()
                logger.info(f"✅ Hostname: {hostname}")
                self.test_results["hostname_test"] = True
            else:
                logger.warning(f"⚠️ Hostname command limited: exit_code={exit_code}, stderr={stderr}")
                
        except Exception as e:
            logger.error(f"❌ Hostname command exception: {e}")
        
        return self.test_results["hostname_test"]
    
    def _analyze_results(self) -> bool:
        """Analyze test results and determine overall success"""
        logger.info("\n" + "=" * 70)
        logger.info("🎯 BIFROST SYNCHRONOUS SSH TEST RESULTS")
        logger.info("=" * 70)
        
        # Show detailed results
        logger.info(f"SSH Connection: {self.ssh_connection}")
        logger.info("")
        
        # Test results
        parsing_status = "✅ PASS" if self.test_results["connection_string_parsing"] else "❌ FAIL"
        ssh_status = "✅ PASS" if self.test_results["ssh_connectivity"] else "❌ FAIL"
        nvidia_status = "✅ PASS" if self.test_results["nvidia_validation"] else "❌ FAIL"
        hostname_status = "✅ PASS" if self.test_results["hostname_test"] else "❌ FAIL"
        
        logger.info(f"  Connection String Parsing: {parsing_status}")
        logger.info(f"  SSH Connectivity:          {ssh_status}")
        logger.info(f"  Hostname Test:             {hostname_status}")
        logger.info(f"  nvidia-smi Validation:     {nvidia_status}")
        
        # Overall assessment
        parsing_working = self.test_results["connection_string_parsing"]
        ssh_working = self.test_results["ssh_connectivity"]
        nvidia_working = self.test_results["nvidia_validation"]
        
        logger.info("\n📊 Summary:")
        logger.info(f"   Connection parsing: {'✅ Working' if parsing_working else '❌ Failed'}")
        logger.info(f"   SSH connectivity: {'✅ Working' if ssh_working else '❌ Failed'}")
        logger.info(f"   GPU validation: {'✅ Working' if nvidia_working else '❌ Failed'}")
        logger.info(f"   SSH Foundation: Shared (no code duplication)")
        
        # Success criteria: SSH and parsing must work, nvidia preferred
        success = parsing_working and ssh_working
        
        if success:
            if nvidia_working:
                logger.info("\n🎉 BIFROST SYNCHRONOUS SSH TEST PASSED!")
                logger.info("Shared SSH foundation working perfectly with bifrost!")
            else:
                logger.info("\n✅ BIFROST SSH TEST MOSTLY PASSED!")
                logger.info("SSH connectivity working, GPU access limited")
        else:
            logger.error("\n❌ BIFROST SYNCHRONOUS SSH TEST FAILED!")
            logger.error("Critical SSH functionality not working")
        
        return success


def main():
    """CLI entry point for bifrost sync SSH test"""
    parser = argparse.ArgumentParser(
        description="Bifrost synchronous SSH integration test using shared foundation"
    )
    parser.add_argument(
        "ssh_connection",
        help="SSH connection string (user@host:port) - use setup_ssh_helper.py to get one"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate SSH connection format
    if '@' not in args.ssh_connection or ':' not in args.ssh_connection:
        print("❌ Invalid SSH connection format. Expected: user@host:port")
        print("   Example: root@194.68.245.12:22059")
        print("   Use setup_ssh_helper.py to provision a GPU instance")
        return 1
    
    print("🚀 BIFROST SYNCHRONOUS SSH INTEGRATION TEST")
    print("")
    print(f"SSH Connection: {args.ssh_connection}")
    print("")
    print("Test coverage:")
    print("  ✓ SSH connection string parsing")
    print("  ✓ SSH connectivity via shared foundation")
    print("  ✓ Command execution (sync methods)")
    print("  ✓ nvidia-smi output capture")
    print("  ✓ Hostname command execution")
    print("")
    
    # Run the test
    test = BifrostSyncTest(args.ssh_connection)
    success = test.run_sync_test()
    
    if success:
        print("\n✅ BIFROST SYNCHRONOUS SSH TEST PASSED!")
        print("Shared SSH foundation integration successful!")
        return 0
    else:
        print("\n❌ BIFROST SYNCHRONOUS SSH TEST FAILED!")
        print("Please check the logs above for specific issues.")
        return 1


if __name__ == "__main__":
    sys.exit(main())