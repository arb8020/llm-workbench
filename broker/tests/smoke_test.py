#!/usr/bin/env python3
"""
Smoke test for GPU broker minimal - provision, wait for SSH, test connectivity
"""

import sys
import os
import asyncio
import logging
import time

# Add the src directory to Python path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def wait_for_public_ip(instance_id: str, max_wait_seconds: int = 300):
    """Wait for instance to have a public IP"""
    import gpu_broker_minimal as gpus
    
    logger.info(f"⏳ Waiting for public IP on instance {instance_id}...")
    start_time = time.time()
    
    while time.time() - start_time < max_wait_seconds:
        fresh_instance = gpus.get_instance(instance_id)
        if not fresh_instance:
            logger.error(f"❌ Instance {instance_id} not found")
            return None
            
        logger.info(f"   Status: {fresh_instance.status}")
        logger.info(f"   Public IP: {fresh_instance.public_ip}")
        logger.info(f"   SSH Port: {fresh_instance.ssh_port}")
        
        # Check if we have a public IP
        if fresh_instance.public_ip and fresh_instance.public_ip != "":
            logger.info(f"✅ Public IP available: {fresh_instance.public_ip}")
            return fresh_instance
            
        logger.info(f"   Waiting for public IP... ({time.time() - start_time:.0f}s elapsed)")
        await asyncio.sleep(15)
    
    logger.error(f"❌ Public IP not available after {max_wait_seconds}s")
    return None

async def test_ssh_connectivity(instance, ssh_method_hint=None):
    """Test actual SSH connectivity and output capture"""
    from gpu_broker_minimal.ssh_clients import execute_command_sync, execute_command_async, SSHMethod
    
    logger.info(f"🔗 Testing SSH connectivity to {instance.public_ip}:{instance.ssh_port}")
    
    # Determine SSH method based on public IP
    if instance.public_ip == "ssh.runpod.io":
        ssh_method = "proxy"
        expected_behavior = "limited output (PTY limitations)"
    else:
        ssh_method = "direct"  
        expected_behavior = "real output capture"
        
    logger.info(f"   SSH method: {ssh_method}")
    logger.info(f"   Expected: {expected_behavior}")
    
    # Wait a bit for SSH daemon to be ready
    logger.info("⏳ Waiting 30s for SSH daemon to initialize...")
    await asyncio.sleep(30)
    
    results = {}
    
    # Test simple command first
    logger.info("\n--- Testing Simple Command (hostname) ---")
    try:
        if ssh_method == "direct":
            success, stdout, stderr = execute_command_sync(
                instance, None, "hostname", SSHMethod.DIRECT, timeout=30
            )
        else:
            # For proxy SSH, we'll need to modify the client to support it
            logger.info("⚠️ Proxy SSH testing would require proxy support in client")
            success, stdout, stderr = False, "", "Proxy SSH not supported in minimal version"
        
        logger.info(f"   Success: {success}")
        logger.info(f"   Stdout: '{stdout.strip()}'")
        logger.info(f"   Stderr: '{stderr.strip()}'")
        
        if success and stdout.strip() and "Your SSH client doesn't support PTY" not in stdout:
            results["hostname"] = "real_output"
            logger.info("✅ Real output captured!")
        elif success:
            results["hostname"] = "limited"
            logger.info("⚠️ Limited output")
        else:
            results["hostname"] = "failed"
            logger.info("❌ Command failed")
            
    except Exception as e:
        logger.error(f"❌ Hostname test error: {e}")
        results["hostname"] = "error"
    
    # Test nvidia-smi command
    logger.info("\n--- Testing NVIDIA Command (nvidia-smi --version) ---")
    try:
        if ssh_method == "direct":
            success, stdout, stderr = execute_command_sync(
                instance, None, "nvidia-smi --version", SSHMethod.DIRECT, timeout=30
            )
        else:
            logger.info("⚠️ Proxy SSH testing skipped")
            success, stdout, stderr = False, "", "Proxy SSH not supported"
        
        logger.info(f"   Success: {success}")
        logger.info(f"   Stdout: '{stdout.strip()}'")
        logger.info(f"   Stderr: '{stderr.strip()}'")
        
        if success and "NVIDIA-SMI" in stdout:
            results["nvidia-smi"] = "real_output"
            logger.info("✅ Real GPU info captured!")
            # Extract driver version
            if "version" in stdout:
                driver_version = stdout.split("version")[1].split()[1] if len(stdout.split("version")) > 1 else "unknown"
                logger.info(f"   Driver: {driver_version}")
        elif success:
            results["nvidia-smi"] = "limited"
            logger.info("⚠️ Limited output")
        else:
            results["nvidia-smi"] = "failed"
            logger.info("❌ Command failed")
            
    except Exception as e:
        logger.error(f"❌ nvidia-smi test error: {e}")
        results["nvidia-smi"] = "error"
    
    # Test AsyncSSH if available
    logger.info("\n--- Testing AsyncSSH Client ---")
    try:
        if ssh_method == "direct":
            success, stdout, stderr = await execute_command_async(
                instance, None, "echo 'AsyncSSH test'", SSHMethod.DIRECT, timeout=30
            )
            
            logger.info(f"   Success: {success}")
            logger.info(f"   Stdout: '{stdout.strip()}'")
            
            if success and "AsyncSSH test" in stdout:
                results["asyncssh"] = "real_output"
                logger.info("✅ AsyncSSH real output!")
            elif success:
                results["asyncssh"] = "limited"
                logger.info("⚠️ AsyncSSH limited output")
            else:
                results["asyncssh"] = "failed"
                logger.info("❌ AsyncSSH failed")
        else:
            logger.info("⚠️ AsyncSSH proxy test skipped")
            results["asyncssh"] = "skipped"
            
    except Exception as e:
        logger.error(f"❌ AsyncSSH test error: {e}")
        results["asyncssh"] = "error"
    
    return results, ssh_method

async def smoke_test():
    """Main smoke test function"""
    
    logger.info("🔥 GPU BROKER MINIMAL - SMOKE TEST")
    logger.info("Testing full workflow: provision → wait for IP → test SSH")
    logger.info("=" * 70)
    
    instance = None
    
    try:
        # Import GPU broker
        import gpu_broker_minimal as gpus
        from gpu_broker_minimal.types import CloudType
        
        logger.info("✅ GPU broker imports successful")
        
        # Step 1: Search for cheap GPU
        logger.info("\n🔍 Step 1: Searching for available GPU...")
        query = gpus.price_per_hour < 0.50
        offers = gpus.search(query)
        
        if not offers:
            logger.error("❌ No affordable offers found")
            return False
            
        logger.info(f"✅ Found {len(offers)} offers")
        
        # Step 2: Try to provision (may need multiple attempts)
        logger.info(f"\n🚀 Step 2: Provisioning instance...")
        instance = None
        
        for i, offer in enumerate(offers[:3]):  # Try up to 3 offers
            logger.info(f"   Attempt {i+1}: {offer.gpu_type} at ${offer.price_per_hour:.3f}/hr")
            try:
                instance = gpus.create(offer)
                if instance:
                    logger.info(f"✅ Success with {offer.gpu_type}")
                    break
            except Exception as e:
                logger.warning(f"⚠️ Failed: {e}")
                continue
        
        if not instance:
            logger.error("❌ Failed to provision any instance")
            return False
            
        logger.info(f"✅ Instance created: {instance.id}")
        logger.info(f"   Initial status: {instance.status}")
        logger.info(f"   Initial public_ip: {instance.public_ip}")
        
        # Step 3: Wait for public IP
        logger.info(f"\n⏳ Step 3: Waiting for public IP...")
        ready_instance = await wait_for_public_ip(instance.id, max_wait_seconds=300)
        
        if not ready_instance:
            logger.error("❌ Instance never got a public IP")
            return False
            
        instance = ready_instance
        
        # Step 4: Test SSH connectivity  
        logger.info(f"\n🔗 Step 4: Testing SSH connectivity...")
        results, ssh_method = await test_ssh_connectivity(instance)
        
        # Step 5: Analyze results
        logger.info(f"\n" + "=" * 70)
        logger.info("🎯 SMOKE TEST RESULTS")
        logger.info("=" * 70)
        
        real_output_count = sum(1 for r in results.values() if r == "real_output")
        working_count = sum(1 for r in results.values() if r in ["real_output", "limited"])
        total_tests = len([r for r in results.values() if r != "skipped"])
        
        logger.info(f"Instance: {instance.id}")
        logger.info(f"SSH method: {ssh_method}")
        logger.info(f"Public IP: {instance.public_ip}:{instance.ssh_port}")
        logger.info(f"")
        
        for test_name, result in results.items():
            if result == "real_output":
                logger.info(f"  ✅ {test_name.upper()}: Real output captured")
            elif result == "limited":
                logger.info(f"  ⚠️ {test_name.upper()}: Limited output")
            elif result == "failed":
                logger.info(f"  ❌ {test_name.upper()}: Failed")
            elif result == "error":
                logger.info(f"  ❌ {test_name.upper()}: Error")
            elif result == "skipped":
                logger.info(f"  ⏩ {test_name.upper()}: Skipped")
        
        logger.info(f"\n📊 Summary:")
        logger.info(f"   Real output: {real_output_count}/{total_tests} tests")
        logger.info(f"   Working: {working_count}/{total_tests} tests")
        
        # Determine overall success
        if ssh_method == "direct":
            success = real_output_count >= 2  # Expect real output for direct SSH
            if success:
                logger.info(f"\n🎉 SMOKE TEST PASSED!")
                logger.info(f"Direct SSH with real output capture is working!")
            else:
                logger.error(f"\n❌ SMOKE TEST FAILED!")
                logger.error(f"Direct SSH should provide real output capture")
        else:
            success = working_count >= 1  # Proxy SSH is limited but should work somewhat
            if success:
                logger.info(f"\n⚠️ SMOKE TEST PARTIAL PASS")
                logger.info(f"Got proxy SSH (limited functionality as expected)")
            else:
                logger.error(f"\n❌ SMOKE TEST FAILED!")
                logger.error(f"Even proxy SSH is not working")
        
        return success
        
    except Exception as e:
        logger.error(f"❌ Smoke test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        if instance:
            logger.info(f"\n🗑️ Cleaning up instance {instance.id}...")
            try:
                if instance.terminate():
                    logger.info(f"✅ Instance terminated")
                else:
                    logger.warning(f"⚠️ Failed to terminate instance {instance.id}")
                    logger.warning("   Please terminate manually to avoid charges")
            except Exception as e:
                logger.error(f"❌ Cleanup error: {e}")

def main():
    """Run smoke test"""
    
    if len(sys.argv) > 1 and sys.argv[1] == "--skip":
        print("⏩ Smoke test skipped")
        return 0
    
    print("🔥 GPU Broker Minimal - Smoke Test")
    print("This will provision 1 GPU and test SSH connectivity")
    print("⏳ Duration: ~5-8 minutes")
    print("💰 Cost: ~$0.02-0.04")
    print("")
    
    response = input("Run smoke test? (y/N): ")
    if response.lower() != 'y':
        print("❌ Test cancelled")
        return 1
    
    success = asyncio.run(smoke_test())
    
    if success:
        print("\n✅ SMOKE TEST PASSED!")
        return 0
    else:
        print("\n❌ SMOKE TEST FAILED!")
        return 1

if __name__ == "__main__":
    sys.exit(main())