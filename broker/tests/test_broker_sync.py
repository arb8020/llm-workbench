#!/usr/bin/env python3
"""
Broker synchronous integration test

Tests the complete broker workflow using only synchronous operations:
1. Search for GPU offers
2. Provision GPU instance  
3. Test SSH connectivity (sync only)
4. Validate nvidia-smi output capture (sync only)
5. Clean up resources

This test validates the synchronous broker API path.
"""

import logging
import sys
import time
from typing import Optional

from broker import GPUClient
from broker.types import CloudType

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BrokerSyncTest:
    """Synchronous broker integration test"""
    
    def __init__(self):
        self.client = GPUClient()
        self.instance = None
        self.test_results = {
            "ssh_connectivity": False,
            "nvidia_validation": False,
            "hostname_test": False
        }
    
    def run_sync_test(self) -> bool:
        """Run complete synchronous integration test with cleanup"""
        logger.info("🚀 BROKER SYNCHRONOUS INTEGRATION TEST")
        logger.info("Testing synchronous execution path only")
        logger.info("=" * 70)
        
        try:
            # Step 1: Search for GPU offers
            if not self._test_gpu_search():
                return False
            
            # Step 2: Provision GPU instance
            if not self._test_gpu_provisioning():
                return False
            
            # Step 3: Wait for instance to be ready
            if not self._test_wait_for_ready():
                return False
            
            # Step 4: Test SSH connectivity (sync only)
            if not self._test_sync_ssh_connectivity():
                return False
            
            # Step 5: Test nvidia-smi output (sync only)  
            if not self._test_sync_nvidia_validation():
                return False
            
            # Step 6: Results analysis
            return self._analyze_results()
            
        except Exception as e:
            logger.error(f"❌ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            self._cleanup()
    
    def _test_gpu_search(self) -> bool:
        """Test GPU search functionality"""
        logger.info("\n🔍 Step 1: Testing GPU search...")
        
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
    
    def _test_gpu_provisioning(self) -> bool:
        """Test GPU provisioning"""
        logger.info("\n🚀 Step 2: Testing GPU provisioning...")
        
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
    
    def _test_wait_for_ready(self) -> bool:
        """Wait for instance to be running with public IP"""
        logger.info("\n⏳ Step 3: Waiting for instance to be ready...")
        
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
                    time.sleep(5)
                    
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
                            time.sleep(15)
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
                time.sleep(30)
                return True
            
            time.sleep(15)
        
        logger.error("❌ Instance not ready after 5 minutes")
        return False
    
    def _test_sync_ssh_connectivity(self) -> bool:
        """Test synchronous SSH connectivity with direct SSH"""
        logger.info("\n🔗 Step 4: Testing synchronous SSH connectivity...")
        logger.info(f"   Using direct SSH: root@{self.instance.public_ip}:{self.instance.ssh_port}")
        
        # Test basic connectivity with direct SSH
        logger.info("\n--- Testing Basic SSH Command ---")
        try:
            from broker.ssh_clients_compat import execute_command_sync
            
            exit_code, stdout, stderr = execute_command_sync(
                self.instance, None, "echo 'SYNC_SSH_TEST'", timeout=30
            )
            
            if exit_code == 0 and "SYNC_SSH_TEST" in stdout:
                logger.info("✅ Sync SSH: Real output captured!")
                logger.info(f"   Output: {stdout.strip()}")
                self.test_results["ssh_connectivity"] = True
            elif exit_code == 0:
                logger.info("✅ Sync SSH: Connection successful (limited output)")
                self.test_results["ssh_connectivity"] = True
            else:
                logger.error(f"❌ Sync SSH: Failed - exit code {exit_code}, {stderr}")
                
        except Exception as e:
            logger.error(f"❌ Sync SSH: Exception - {e}")
        
        # Test hostname command  
        logger.info("\n--- Testing Hostname Command ---")
        try:
            result = self.instance.exec("hostname")
            if result.success and result.stdout.strip():
                logger.info(f"✅ instance.exec(): hostname = {result.stdout.strip()}")
                self.test_results["hostname_test"] = True
            else:
                logger.warning(f"⚠️ instance.exec(): Limited - {result.stderr}")
                
        except Exception as e:
            logger.error(f"❌ instance.exec() error: {e}")
        
        return self.test_results["ssh_connectivity"]
    
    def _test_sync_nvidia_validation(self) -> bool:
        """Test nvidia-smi output capture synchronously"""
        logger.info("\n🎯 Step 5: Testing nvidia-smi validation (sync)...")
        
        # Only proceed if we have SSH connectivity
        if not self.test_results["ssh_connectivity"]:
            logger.error("❌ No SSH connectivity - skipping nvidia-smi tests")
            return False
        
        logger.info(f"   Using direct SSH: root@{self.instance.public_ip}:{self.instance.ssh_port}")
        
        logger.info("\n--- Testing Sync nvidia-smi ---")
        try:
            from broker.ssh_clients_compat import execute_command_sync
            
            exit_code, stdout, stderr = execute_command_sync(
                self.instance, None, "nvidia-smi --version", timeout=30
            )
            
            if exit_code == 0 and "NVIDIA-SMI" in stdout:
                logger.info("✅ Sync nvidia-smi: Real GPU output captured!")
                # Extract driver version
                version_info = stdout.split('\n')[0] if stdout else "Unknown"
                logger.info(f"   Version info: {version_info}")
                self.test_results["nvidia_validation"] = True
            elif exit_code == 0:
                logger.warning("⚠️ Sync nvidia-smi: Command ran but limited output")
            else:
                logger.error(f"❌ Sync nvidia-smi: Failed - exit code {exit_code}, {stderr}")
                
        except Exception as e:
            logger.error(f"❌ Sync nvidia-smi: Exception - {e}")
        
        # Also test via instance method
        logger.info("\n--- Testing nvidia-smi via instance.exec() ---")
        try:
            result = self.instance.exec("nvidia-smi --query-gpu=name --format=csv,noheader")
            if result.success and result.stdout.strip():
                gpu_name = result.stdout.strip()
                logger.info(f"✅ GPU detected: {gpu_name}")
                self.test_results["nvidia_validation"] = True
            else:
                logger.warning(f"⚠️ GPU query limited: {result.stderr}")
                
        except Exception as e:
            logger.error(f"❌ GPU query error: {e}")
        
        return self.test_results["nvidia_validation"]
    
    def _analyze_results(self) -> bool:
        """Analyze test results and determine overall success"""
        logger.info("\n" + "=" * 70)
        logger.info("🎯 SYNCHRONOUS INTEGRATION TEST RESULTS")
        logger.info("=" * 70)
        
        # Show detailed results
        logger.info(f"Instance: {self.instance.id}")
        logger.info(f"SSH method: {'direct' if self.instance.public_ip != 'ssh.runpod.io' else 'proxy'}")
        logger.info(f"Public IP: {self.instance.public_ip}:{self.instance.ssh_port}")
        logger.info("")
        
        # Test results
        ssh_status = "✅ PASS" if self.test_results["ssh_connectivity"] else "❌ FAIL"
        nvidia_status = "✅ PASS" if self.test_results["nvidia_validation"] else "❌ FAIL"
        hostname_status = "✅ PASS" if self.test_results["hostname_test"] else "❌ FAIL"
        
        logger.info(f"  SSH Connectivity:     {ssh_status}")
        logger.info(f"  Hostname Test:        {hostname_status}")
        logger.info(f"  nvidia-smi Validation:{nvidia_status}")
        
        # Overall assessment
        ssh_working = self.test_results["ssh_connectivity"]
        nvidia_working = self.test_results["nvidia_validation"]
        
        logger.info("\n📊 Summary:")
        logger.info(f"   SSH connectivity: {'✅ Working' if ssh_working else '❌ Failed'}")
        logger.info(f"   GPU validation: {'✅ Working' if nvidia_working else '❌ Failed'}")
        logger.info(f"   Execution path: Synchronous only")
        
        # Success criteria: Both SSH and nvidia-smi must work
        success = ssh_working and nvidia_working
        
        if success:
            logger.info("\n🎉 SYNCHRONOUS INTEGRATION TEST PASSED!")
            logger.info("All synchronous broker functionality is working correctly!")
        else:
            logger.error("\n❌ SYNCHRONOUS INTEGRATION TEST FAILED!")
            logger.error("Critical synchronous functionality not working")
        
        return success
    
    def _cleanup(self):
        """Clean up test resources"""
        logger.info(f"\n🗑️ Step 6: Cleaning up resources...")
        
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


def run_sync_test() -> bool:
    """Run the synchronous broker integration test"""
    test = BrokerSyncTest()
    return test.run_sync_test()


def main():
    """CLI entry point for sync integration test"""
    
    # Check for skip flag
    if len(sys.argv) > 1 and sys.argv[1] == "--skip":
        print("⏩ Broker sync integration test skipped (use without --skip to run)")
        return 0
    
    # Cost and duration warning
    print("🚀 BROKER SYNCHRONOUS INTEGRATION TEST")
    print("")
    print("⚠️  This test will provision a GPU instance")
    print("⏳ Duration: ~6-10 minutes (including provisioning)")
    print("💰 Estimated cost: ~$0.02-0.05")
    print("")
    print("Test coverage (synchronous path only):")
    print("  ✓ GPU search functionality")
    print("  ✓ GPU provisioning")
    print("  ✓ SSH connectivity (Paramiko)")
    print("  ✓ Command execution (sync methods)")
    print("  ✓ nvidia-smi output capture")
    print("  ✓ Resource cleanup")
    print("")
    
    response = input("Continue with synchronous broker test? (y/N): ")
    if response.lower() != 'y':
        print("❌ Test cancelled")
        return 1
    
    # Run the test
    success = run_sync_test()
    
    if success:
        print("\n✅ SYNCHRONOUS INTEGRATION TEST PASSED!")
        print("All synchronous broker functionality is working!")
        return 0
    else:
        print("\n❌ SYNCHRONOUS INTEGRATION TEST FAILED!")
        print("Please check the logs above for specific issues.")
        return 1


if __name__ == "__main__":
    sys.exit(main())