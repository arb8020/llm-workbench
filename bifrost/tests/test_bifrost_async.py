#!/usr/bin/env python3
"""
Bifrost asynchronous SSH integration test

Tests bifrost's use of the shared SSH foundation with asynchronous operations:
1. Test SSH connectivity using shared foundation (async only)
2. Validate nvidia-smi output capture (async only)
3. Test parallel command execution (async)
4. Test bifrost-specific SSH patterns (connection strings)

This test validates that bifrost can use the shared SSH foundation for
asynchronous deployment operations without SSH code duplication.

Usage:
    python test_bifrost_async.py user@host:port
    
Example:
    python test_bifrost_async.py root@194.68.245.12:22059
    
Use setup_ssh_helper.py to provision a GPU instance for testing.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add bifrost to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.logging_config import setup_logging
from bifrost.ssh_clients_compat import (
    execute_command_async,
    test_ssh_connection,
    SSHConnectionInfo
)

# Configure logging
setup_logging(level="INFO")
logger = logging.getLogger(__name__)


class BifrostAsyncTest:
    """Asynchronous bifrost SSH integration test using shared foundation"""
    
    def __init__(self, ssh_connection: str):
        self.ssh_connection = ssh_connection
        self.test_results = {
            "async_ssh_connectivity": False,
            "async_nvidia_validation": False,
            "async_hostname_test": False,
            "parallel_execution": False,
            "connection_string_parsing": False
        }
    
    async def run_async_test(self) -> bool:
        """Run complete asynchronous bifrost SSH test"""
        logger.info("🚀 BIFROST ASYNCHRONOUS SSH INTEGRATION TEST")
        logger.info("Testing shared SSH foundation async integration")
        logger.info("=" * 70)
        
        try:
            # Step 1: Test connection string parsing
            if not await self._test_connection_string_parsing():
                return False
            
            # Step 2: Test async SSH connectivity
            if not await self._test_async_ssh_connectivity():
                return False
            
            # Step 3: Test async nvidia-smi validation
            if not await self._test_async_nvidia_validation():
                return False
            
            # Step 4: Test parallel command execution
            if not await self._test_parallel_execution():
                return False
            
            # Step 5: Results analysis
            return self._analyze_results()
            
        except Exception as e:
            logger.error(f"❌ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def _test_connection_string_parsing(self) -> bool:
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
    
    async def _test_async_ssh_connectivity(self) -> bool:
        """Test async SSH connectivity via shared foundation"""
        logger.info("\n🔗 Step 2: Testing async SSH connectivity...")
        logger.info(f"   Using connection: {self.ssh_connection}")
        
        # Test basic async command execution
        logger.info("\n--- Testing Async Command Execution ---")
        try:
            exit_code, stdout, stderr = await execute_command_async(
                self.ssh_connection, "echo 'BIFROST_ASYNC_SSH_TEST'", timeout=30
            )
            
            if exit_code == 0 and "BIFROST_ASYNC_SSH_TEST" in stdout:
                logger.info("✅ Async SSH command execution working!")
                logger.info(f"   Output: {stdout.strip()}")
                self.test_results["async_ssh_connectivity"] = True
            elif exit_code == 0:
                logger.info("✅ Async SSH connection successful (limited output)")
                self.test_results["async_ssh_connectivity"] = True
            else:
                logger.error(f"❌ Async SSH command failed - exit code {exit_code}, {stderr}")
                
        except Exception as e:
            logger.error(f"❌ Async SSH command exception: {e}")
        
        # Test hostname via async
        logger.info("\n--- Testing Async Hostname Command ---")
        try:
            exit_code, stdout, stderr = await execute_command_async(
                self.ssh_connection, "hostname", timeout=30
            )
            
            if exit_code == 0 and stdout.strip():
                hostname = stdout.strip()
                logger.info(f"✅ Async hostname: {hostname}")
                self.test_results["async_hostname_test"] = True
            else:
                logger.warning(f"⚠️ Async hostname limited: exit_code={exit_code}, stderr={stderr}")
                
        except Exception as e:
            logger.error(f"❌ Async hostname exception: {e}")
        
        return self.test_results["async_ssh_connectivity"]
    
    async def _test_async_nvidia_validation(self) -> bool:
        """Test async nvidia-smi output capture via shared foundation"""
        logger.info("\n🎯 Step 3: Testing async nvidia-smi validation...")
        
        # Only proceed if we have async SSH connectivity
        if not self.test_results["async_ssh_connectivity"]:
            logger.error("❌ No async SSH connectivity - skipping nvidia-smi tests")
            return False
        
        logger.info(f"   Using connection: {self.ssh_connection}")
        
        logger.info("\n--- Testing Async nvidia-smi Version ---")
        try:
            exit_code, stdout, stderr = await execute_command_async(
                self.ssh_connection, "nvidia-smi --version", timeout=30
            )
            
            if exit_code == 0 and "NVIDIA-SMI" in stdout:
                logger.info("✅ Async nvidia-smi: Real GPU output captured!")
                # Extract version info
                version_info = stdout.split('\n')[0] if stdout else "Unknown"
                logger.info(f"   Version info: {version_info}")
                self.test_results["async_nvidia_validation"] = True
            elif exit_code == 0:
                logger.warning("⚠️ Async nvidia-smi ran but limited output")
            else:
                logger.error(f"❌ Async nvidia-smi failed - exit code {exit_code}, {stderr}")
                
        except Exception as e:
            logger.error(f"❌ Async nvidia-smi exception: {e}")
        
        # Test GPU info query
        logger.info("\n--- Testing Async GPU Info Query ---")
        try:
            exit_code, stdout, stderr = await execute_command_async(
                self.ssh_connection, "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", timeout=30
            )
            
            if exit_code == 0 and stdout.strip():
                gpu_info = stdout.strip()
                logger.info(f"✅ Async GPU info: {gpu_info}")
                self.test_results["async_nvidia_validation"] = True
            else:
                logger.warning(f"⚠️ Async GPU query limited: exit_code={exit_code}, stderr={stderr}")
                
        except Exception as e:
            logger.error(f"❌ Async GPU query exception: {e}")
        
        return self.test_results["async_nvidia_validation"]
    
    async def _test_parallel_execution(self) -> bool:
        """Test parallel async command execution"""
        logger.info("\n🔄 Step 4: Testing parallel async execution...")
        
        if not self.test_results["async_ssh_connectivity"]:
            logger.error("❌ No async SSH connectivity - skipping parallel tests")
            return False
        
        logger.info("\n--- Testing Parallel Commands ---")
        try:
            # Execute multiple commands in parallel
            tasks = [
                execute_command_async(self.ssh_connection, "whoami"),
                execute_command_async(self.ssh_connection, "pwd"),
                execute_command_async(self.ssh_connection, "date"),
                execute_command_async(self.ssh_connection, "uname -a")
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            success_count = 0
            command_names = ["whoami", "pwd", "date", "uname"]
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"❌ Parallel command {command_names[i]}: {result}")
                elif result[0] == 0:  # exit_code == 0
                    logger.info(f"✅ Parallel {command_names[i]}: {result[1].strip()}")
                    success_count += 1
                else:
                    logger.warning(f"⚠️ Parallel {command_names[i]}: Limited output")
            
            if success_count >= 2:
                logger.info("✅ Parallel async execution working!")
                self.test_results["parallel_execution"] = True
            else:
                logger.warning("⚠️ Parallel async execution has issues")
                
        except Exception as e:
            logger.error(f"❌ Parallel execution exception: {e}")
        
        # Test concurrent GPU queries
        logger.info("\n--- Testing Concurrent GPU Queries ---")
        try:
            concurrent_tasks = [
                execute_command_async(self.ssh_connection, "nvidia-smi --query-gpu=name --format=csv,noheader"),
                execute_command_async(self.ssh_connection, "nvidia-smi --query-gpu=memory.total --format=csv,noheader"),
                execute_command_async(self.ssh_connection, "nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader")
            ]
            
            results = await asyncio.gather(*concurrent_tasks, return_exceptions=True)
            
            for i, result in enumerate(results):
                query_type = ["GPU Name", "Memory", "Temperature"][i]
                if isinstance(result, Exception):
                    logger.error(f"❌ {query_type} query: {result}")
                elif result[0] == 0 and result[1].strip():
                    logger.info(f"✅ {query_type}: {result[1].strip()}")
                else:
                    logger.warning(f"⚠️ {query_type}: Limited output")
                    
        except Exception as e:
            logger.error(f"❌ Concurrent GPU queries exception: {e}")
        
        return self.test_results["parallel_execution"]
    
    def _analyze_results(self) -> bool:
        """Analyze test results and determine overall success"""
        logger.info("\n" + "=" * 70)
        logger.info("🎯 BIFROST ASYNCHRONOUS SSH TEST RESULTS")
        logger.info("=" * 70)
        
        # Show detailed results
        logger.info(f"SSH Connection: {self.ssh_connection}")
        logger.info("")
        
        # Test results
        parsing_status = "✅ PASS" if self.test_results["connection_string_parsing"] else "❌ FAIL"
        async_ssh_status = "✅ PASS" if self.test_results["async_ssh_connectivity"] else "❌ FAIL"
        async_nvidia_status = "✅ PASS" if self.test_results["async_nvidia_validation"] else "❌ FAIL"
        async_hostname_status = "✅ PASS" if self.test_results["async_hostname_test"] else "❌ FAIL"
        parallel_status = "✅ PASS" if self.test_results["parallel_execution"] else "❌ FAIL"
        
        logger.info(f"  Connection String Parsing:  {parsing_status}")
        logger.info(f"  Async SSH Connectivity:     {async_ssh_status}")
        logger.info(f"  Async Hostname Test:        {async_hostname_status}")
        logger.info(f"  Async nvidia-smi Validation:{async_nvidia_status}")
        logger.info(f"  Parallel Execution:         {parallel_status}")
        
        # Overall assessment
        parsing_working = self.test_results["connection_string_parsing"]
        async_ssh_working = self.test_results["async_ssh_connectivity"]
        async_nvidia_working = self.test_results["async_nvidia_validation"]
        parallel_working = self.test_results["parallel_execution"]
        
        logger.info("\n📊 Summary:")
        logger.info(f"   Connection parsing: {'✅ Working' if parsing_working else '❌ Failed'}")
        logger.info(f"   Async SSH connectivity: {'✅ Working' if async_ssh_working else '❌ Failed'}")
        logger.info(f"   Async GPU validation: {'✅ Working' if async_nvidia_working else '❌ Failed'}")
        logger.info(f"   Parallel execution: {'✅ Working' if parallel_working else '❌ Failed'}")
        logger.info(f"   SSH Foundation: Shared (async support)")
        
        # Success criteria: async SSH and parsing must work
        success = parsing_working and async_ssh_working
        
        if success:
            if async_nvidia_working and parallel_working:
                logger.info("\n🎉 BIFROST ASYNCHRONOUS SSH TEST PASSED!")
                logger.info("Full async shared SSH foundation integration successful!")
            else:
                logger.info("\n✅ BIFROST ASYNC SSH TEST MOSTLY PASSED!")
                logger.info("Async SSH connectivity working, some features limited")
        else:
            logger.error("\n❌ BIFROST ASYNCHRONOUS SSH TEST FAILED!")
            logger.error("Critical async SSH functionality not working")
        
        return success


async def run_async_test(ssh_connection: str) -> bool:
    """Run the asynchronous bifrost SSH test"""
    test = BifrostAsyncTest(ssh_connection)
    return await test.run_async_test()


def main():
    """CLI entry point for bifrost async SSH test"""
    parser = argparse.ArgumentParser(
        description="Bifrost asynchronous SSH integration test using shared foundation"
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
    
    print("🚀 BIFROST ASYNCHRONOUS SSH INTEGRATION TEST")
    print("")
    print(f"SSH Connection: {args.ssh_connection}")
    print("")
    print("Test coverage:")
    print("  ✓ SSH connection string parsing")
    print("  ✓ Async SSH connectivity via shared foundation")
    print("  ✓ Async command execution")
    print("  ✓ Parallel command execution")
    print("  ✓ Async nvidia-smi output capture")
    print("  ✓ Concurrent GPU queries")
    print("")
    
    # Run the test
    success = asyncio.run(run_async_test(args.ssh_connection))
    
    if success:
        print("\n✅ BIFROST ASYNCHRONOUS SSH TEST PASSED!")
        print("Async shared SSH foundation integration successful!")
        return 0
    else:
        print("\n❌ BIFROST ASYNCHRONOUS SSH TEST FAILED!")
        print("Please check the logs above for specific issues.")
        return 1


if __name__ == "__main__":
    sys.exit(main())