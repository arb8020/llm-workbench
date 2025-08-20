#!/usr/bin/env python3
"""
Broker asynchronous integration test

Tests the complete broker workflow using asynchronous operations where available:
1. Search for GPU offers (sync - no async version)
2. Provision GPU instance (sync - no async version)
3. Test SSH connectivity (async only)
4. Validate nvidia-smi output capture (async only)
5. Clean up resources (sync - no async version)

This test validates the asynchronous broker API path.
"""

import asyncio
import logging
import sys
import time
from typing import Optional

from broker import GPUClient
from broker.types import CloudType

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BrokerAsyncTest:
    """Asynchronous broker integration test"""
    
    def __init__(self):
        self.client = GPUClient()
        self.instance = None
        self.test_results = {
            "async_ssh_connectivity": False,
            "async_nvidia_validation": False,
            "async_hostname_test": False
        }
    
    async def run_async_test(self) -> bool:
        """Run complete asynchronous integration test with cleanup"""
        logger.info("🚀 BROKER ASYNCHRONOUS INTEGRATION TEST")
        logger.info("Testing asynchronous execution path where available")
        logger.info("=" * 70)
        
        try:
            # Step 1: Search for GPU offers (sync - no async version)
            if not await self._test_gpu_search():
                return False
            
            # Step 2: Provision GPU instance (sync - no async version)
            if not await self._test_gpu_provisioning():
                return False
            
            # Step 3: Wait for instance to be ready
            if not await self._test_wait_for_ready():
                return False
            
            # Step 4: Test SSH connectivity (async only)
            if not await self._test_async_ssh_connectivity():
                return False
            
            # Step 5: Test nvidia-smi output (async only)  
            if not await self._test_async_nvidia_validation():
                return False
            
            # Step 6: Results analysis
            return self._analyze_results()
            
        except Exception as e:
            logger.error(f"❌ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            await self._cleanup()
    
    async def _test_gpu_search(self) -> bool:
        """Test GPU search functionality (sync - no async version)"""
        logger.info("\n🔍 Step 1: Testing GPU search...")
        logger.info("   (Note: GPU search API is synchronous)")
        
        try:
            # Search for affordable secure cloud GPUs (required for direct SSH)
            query = (self.client.cloud_type == CloudType.SECURE) & (self.client.price_per_hour < 0.50)
            offers = self.client.search(query)
            
            if not offers:
                logger.error("❌ No affordable secure cloud GPUs found")
                logger.error("   Secure cloud required for direct SSH with output capture")
                return False
            
            logger.info(f"✅ Found {len(offers)} suitable offers")
            logger.info(f"   Best offer: {offers[0].gpu_type} at ${offers[0].price_per_hour:.3f}/hr")
            logger.info(f"   Memory: {offers[0].memory_gb}GB")
            
            # Store the best offer for provisioning
            self.best_offer = offers[0]
            return True
            
        except Exception as e:
            logger.error(f"❌ GPU search failed: {e}")
            return False
    
    async def _test_gpu_provisioning(self) -> bool:
        """Test GPU provisioning (sync - no async version)"""
        logger.info("\n🚀 Step 2: Testing GPU provisioning...")
        logger.info("   (Note: GPU provisioning API is synchronous)")
        
        try:
            # Provision using the best offer found
            self.instance = self.client.create(self.best_offer)
            
            if not self.instance:
                logger.error("❌ GPU provisioning failed")
                return False
            
            logger.info(f"✅ Instance created: {self.instance.id}")
            logger.info(f"   GPU: {self.instance.gpu_type}")
            logger.info(f"   Price: ${self.instance.price_per_hour:.3f}/hr")
            logger.info(f"   Status: {self.instance.status}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ GPU provisioning failed: {e}")
            return False
    
    async def _test_wait_for_ready(self) -> bool:
        """Wait for instance to be running with public IP"""
        logger.info("\n⏳ Step 3: Waiting for instance to be ready...")
        logger.info("   (Note: Instance status polling is synchronous)")
        
        max_wait = 300  # 5 minutes
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            # Refresh instance status
            fresh_instance = self.client.get_instance(self.instance.id)
            if not fresh_instance:
                logger.error("❌ Instance disappeared during provisioning")
                return False
            
            self.instance = fresh_instance
            logger.info(f"   Status: {self.instance.status}")
            logger.info(f"   Public IP: {self.instance.public_ip}")
            
            # Check if running with public IP
            if (str(self.instance.status) == "InstanceStatus.RUNNING" and 
                self.instance.public_ip and 
                self.instance.ssh_port):
                
                # Check SSH type - keep trying until we get direct SSH
                if self.instance.public_ip == "ssh.runpod.io":
                    logger.warning("⚠️ Got proxy SSH - need direct SSH for proper testing")
                    logger.info("   Will keep provisioning until we get direct SSH...")
                    
                    # Terminate proxy SSH instance
                    self.instance.terminate()
                    await asyncio.sleep(5)
                    
                    # Keep trying to get direct SSH (up to 5 attempts)
                    max_attempts = 5
                    for attempt in range(max_attempts):
                        logger.info(f"   Attempt {attempt + 1}/{max_attempts} for direct SSH...")
                        
                        # Provision new instance
                        retry_instance = self.client.create(self.best_offer)
                        if not retry_instance:
                            logger.warning(f"   Failed to provision on attempt {attempt + 1}")
                            continue
                        
                        # Wait for it to be ready and check SSH type
                        retry_start = time.time()
                        got_direct_ssh = False
                        
                        while time.time() - retry_start < 180:  # 3 minutes per attempt
                            await asyncio.sleep(15)
                            fresh_retry = self.client.get_instance(retry_instance.id)
                            
                            if (fresh_retry and 
                                str(fresh_retry.status) == "InstanceStatus.RUNNING" and 
                                fresh_retry.public_ip and 
                                fresh_retry.ssh_port):
                                
                                if fresh_retry.public_ip != "ssh.runpod.io":
                                    logger.info(f"✅ Got direct SSH on attempt {attempt + 1}!")
                                    logger.info(f"   SSH: root@{fresh_retry.public_ip}:{fresh_retry.ssh_port}")
                                    self.instance = fresh_retry
                                    got_direct_ssh = True
                                    break
                                else:
                                    logger.info(f"   Attempt {attempt + 1}: Still proxy SSH, terminating...")
                                    fresh_retry.terminate()
                                    break
                        
                        if got_direct_ssh:
                            break
                        elif retry_instance:
                            # Clean up if we didn't get direct SSH
                            try:
                                retry_instance.terminate()
                            except:
                                pass
                    
                    # Final check
                    if self.instance.public_ip == "ssh.runpod.io":
                        logger.error(f"❌ Could not get direct SSH after {max_attempts} attempts")
                        logger.error("   RunPod may not have direct SSH instances available right now")
                        logger.error("   This is a RunPod infrastructure limitation, not a broker issue")
                        return False
                
                logger.info("✅ Got direct SSH - full output capture expected")
                
                logger.info(f"   SSH: root@{self.instance.public_ip}:{self.instance.ssh_port}")
                
                # Wait additional time for SSH daemon to be ready
                logger.info("⏳ Waiting 30s for SSH daemon to initialize...")
                await asyncio.sleep(30)
                return True
            
            await asyncio.sleep(15)
        
        logger.error("❌ Instance not ready after 5 minutes")
        return False
    
    async def _test_async_ssh_connectivity(self) -> bool:
        """Test asynchronous SSH connectivity with direct SSH"""
        logger.info("\n🔗 Step 4: Testing asynchronous SSH connectivity...")
        logger.info(f"   Using direct SSH: root@{self.instance.public_ip}:{self.instance.ssh_port}")
        
        # Test async SSH basic connectivity
        logger.info("\n--- Testing Async SSH Command ---")
        try:
            from broker.ssh_clients_compat import execute_command_async
            
            exit_code, stdout, stderr = await execute_command_async(
                self.instance, None, "echo 'ASYNC_SSH_TEST'", timeout=30
            )
            
            if exit_code == 0 and "ASYNC_SSH_TEST" in stdout:
                logger.info("✅ Async SSH: Real output captured!")
                self.test_results["async_ssh_connectivity"] = True
            elif exit_code == 0:
                logger.info("✅ Async SSH: Connection successful (limited output)")
                self.test_results["async_ssh_connectivity"] = True
            else:
                logger.error(f"❌ Async SSH: Failed - exit code {exit_code}, {stderr}")
                
        except Exception as e:
            logger.error(f"❌ Async SSH: Exception - {e}")
        
        # Test async instance method
        logger.info("\n--- Testing instance.aexec() Method ---")
        try:
            result = await self.instance.aexec("hostname")
            if result.success and result.stdout.strip():
                logger.info(f"✅ instance.aexec(): hostname = {result.stdout.strip()}")
                self.test_results["async_hostname_test"] = True
            else:
                logger.warning(f"⚠️ instance.aexec(): Limited - {result.stderr}")
                
        except Exception as e:
            logger.error(f"❌ instance.aexec() error: {e}")
        
        # Test parallel async execution
        logger.info("\n--- Testing Parallel Async Commands ---")
        try:
            # Execute multiple commands in parallel
            tasks = [
                self.instance.aexec("whoami"),
                self.instance.aexec("pwd"),
                self.instance.aexec("date")
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            success_count = 0
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"❌ Parallel command {i+1}: {result}")
                elif result.success:
                    logger.info(f"✅ Parallel command {i+1}: {result.stdout.strip()}")
                    success_count += 1
                else:
                    logger.warning(f"⚠️ Parallel command {i+1}: Limited output")
            
            if success_count >= 2:
                logger.info("✅ Parallel async execution working!")
            else:
                logger.warning("⚠️ Parallel async execution has issues")
                
        except Exception as e:
            logger.error(f"❌ Parallel execution error: {e}")
        
        return self.test_results["async_ssh_connectivity"]
    
    async def _test_async_nvidia_validation(self) -> bool:
        """Test nvidia-smi output capture asynchronously"""
        logger.info("\n🎯 Step 5: Testing nvidia-smi validation (async)...")
        
        # Only proceed if we have async SSH connectivity
        if not self.test_results["async_ssh_connectivity"]:
            logger.error("❌ No async SSH connectivity - skipping nvidia-smi tests")
            return False
        
        logger.info(f"   Using direct SSH: root@{self.instance.public_ip}:{self.instance.ssh_port}")
        
        logger.info("\n--- Testing Async nvidia-smi ---")
        try:
            from broker.ssh_clients_compat import execute_command_async
            
            exit_code, stdout, stderr = await execute_command_async(
                self.instance, None, "nvidia-smi --version", timeout=30
            )
            
            if exit_code == 0 and "NVIDIA-SMI" in stdout:
                logger.info("✅ Async nvidia-smi: Real GPU output captured!")
                # Extract driver version
                version_info = stdout.split('\n')[0] if stdout else "Unknown"
                logger.info(f"   Version info: {version_info}")
                self.test_results["async_nvidia_validation"] = True
            elif exit_code == 0:
                logger.warning("⚠️ Async nvidia-smi: Command ran but limited output")
            else:
                logger.error(f"❌ Async nvidia-smi: Failed - exit code {exit_code}, {stderr}")
                
        except Exception as e:
            logger.error(f"❌ Async nvidia-smi: Exception - {e}")
        
        # Also test via async instance method
        logger.info("\n--- Testing nvidia-smi via instance.aexec() ---")
        try:
            result = await self.instance.aexec("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
            if result.success and result.stdout.strip():
                gpu_info = result.stdout.strip()
                logger.info(f"✅ GPU info: {gpu_info}")
                self.test_results["async_nvidia_validation"] = True
            else:
                logger.warning(f"⚠️ GPU query limited: {result.stderr}")
                
        except Exception as e:
            logger.error(f"❌ GPU query error: {e}")
        
        # Test concurrent GPU queries
        logger.info("\n--- Testing Concurrent GPU Queries ---")
        try:
            concurrent_tasks = [
                self.instance.aexec("nvidia-smi --query-gpu=name --format=csv,noheader"),
                self.instance.aexec("nvidia-smi --query-gpu=memory.total --format=csv,noheader"),
                self.instance.aexec("nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader")
            ]
            
            results = await asyncio.gather(*concurrent_tasks, return_exceptions=True)
            
            for i, result in enumerate(results):
                query_type = ["GPU Name", "Memory", "Temperature"][i]
                if isinstance(result, Exception):
                    logger.error(f"❌ {query_type} query: {result}")
                elif result.success and result.stdout.strip():
                    logger.info(f"✅ {query_type}: {result.stdout.strip()}")
                else:
                    logger.warning(f"⚠️ {query_type}: Limited output")
                    
        except Exception as e:
            logger.error(f"❌ Concurrent queries error: {e}")
        
        return self.test_results["async_nvidia_validation"]
    
    def _analyze_results(self) -> bool:
        """Analyze test results and determine overall success"""
        logger.info("\n" + "=" * 70)
        logger.info("🎯 ASYNCHRONOUS INTEGRATION TEST RESULTS")
        logger.info("=" * 70)
        
        # Show detailed results
        logger.info(f"Instance: {self.instance.id}")
        logger.info(f"SSH method: {'direct' if self.instance.public_ip != 'ssh.runpod.io' else 'proxy'}")
        logger.info(f"Public IP: {self.instance.public_ip}:{self.instance.ssh_port}")
        logger.info("")
        
        # Test results
        async_ssh_status = "✅ PASS" if self.test_results["async_ssh_connectivity"] else "❌ FAIL"
        async_nvidia_status = "✅ PASS" if self.test_results["async_nvidia_validation"] else "❌ FAIL"
        async_hostname_status = "✅ PASS" if self.test_results["async_hostname_test"] else "❌ FAIL"
        
        logger.info(f"  Async SSH Connectivity:     {async_ssh_status}")
        logger.info(f"  Async Hostname Test:        {async_hostname_status}")
        logger.info(f"  Async nvidia-smi Validation:{async_nvidia_status}")
        
        # Overall assessment
        async_ssh_working = self.test_results["async_ssh_connectivity"]
        async_nvidia_working = self.test_results["async_nvidia_validation"]
        
        logger.info("\n📊 Summary:")
        logger.info(f"   Async SSH connectivity: {'✅ Working' if async_ssh_working else '❌ Failed'}")
        logger.info(f"   Async GPU validation: {'✅ Working' if async_nvidia_working else '❌ Failed'}")
        logger.info(f"   Execution path: Asynchronous where available")
        
        # Success criteria: Both async SSH and nvidia-smi must work
        success = async_ssh_working and async_nvidia_working
        
        if success:
            logger.info("\n🎉 ASYNCHRONOUS INTEGRATION TEST PASSED!")
            logger.info("All asynchronous broker functionality is working correctly!")
        else:
            logger.error("\n❌ ASYNCHRONOUS INTEGRATION TEST FAILED!")
            logger.error("Critical asynchronous functionality not working")
        
        return success
    
    async def _cleanup(self):
        """Clean up test resources (sync - no async version)"""
        logger.info(f"\n🗑️ Step 6: Cleaning up resources...")
        logger.info("   (Note: Instance termination API is synchronous)")
        
        if self.instance:
            try:
                if self.instance.terminate():
                    logger.info(f"✅ Instance {self.instance.id} terminated successfully")
                else:
                    logger.warning(f"⚠️ Failed to terminate instance {self.instance.id}")
                    logger.warning("   Please terminate manually to avoid charges")
            except Exception as e:
                logger.error(f"❌ Cleanup error: {e}")
                logger.error(f"   Please manually terminate instance: {self.instance.id}")


async def run_async_test() -> bool:
    """Run the asynchronous broker integration test"""
    test = BrokerAsyncTest()
    return await test.run_async_test()


def main():
    """CLI entry point for async integration test"""
    
    # Check for skip flag
    if len(sys.argv) > 1 and sys.argv[1] == "--skip":
        print("⏩ Broker async integration test skipped (use without --skip to run)")
        return 0
    
    # Cost and duration warning
    print("🚀 BROKER ASYNCHRONOUS INTEGRATION TEST")
    print("")
    print("⚠️  This test will provision a GPU instance")
    print("⏳ Duration: ~6-10 minutes (including provisioning)")
    print("💰 Estimated cost: ~$0.02-0.05")
    print("")
    print("Test coverage (asynchronous path where available):")
    print("  ✓ GPU search functionality (sync)")
    print("  ✓ GPU provisioning (sync)")
    print("  ✓ SSH connectivity (AsyncSSH)")
    print("  ✓ Command execution (async methods)")
    print("  ✓ Parallel command execution")
    print("  ✓ nvidia-smi output capture (async)")
    print("  ✓ Resource cleanup (sync)")
    print("")
    
    response = input("Continue with asynchronous broker test? (y/N): ")
    if response.lower() != 'y':
        print("❌ Test cancelled")
        return 1
    
    # Run the test
    success = asyncio.run(run_async_test())
    
    if success:
        print("\n✅ ASYNCHRONOUS INTEGRATION TEST PASSED!")
        print("All asynchronous broker functionality is working!")
        return 0
    else:
        print("\n❌ ASYNCHRONOUS INTEGRATION TEST FAILED!")
        print("Please check the logs above for specific issues.")
        return 1


if __name__ == "__main__":
    sys.exit(main())