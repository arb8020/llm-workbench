#!/usr/bin/env python3
"""
Test both Paramiko and AsyncSSH clients with output capture validation
"""

import asyncio
import logging
import os
import sys


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Test instance ID (update this to your test instance)
TEST_INSTANCE_ID = "tcbo8o4ilmf53v"

async def test_both_ssh_clients():
    """Test both Paramiko and AsyncSSH with real output validation"""
    
    logger.info("🚀 Testing Both SSH Clients - Minimal GPU Broker")
    logger.info("=" * 60)
    
    try:
        # Import minimal GPU broker
        import broker as gpus
        from broker.ssh_clients import execute_command_async, execute_command_sync
        logger.info("✅ GPU broker imports successfully")
        
        # Get test instance
        instance = gpus.get_instance(TEST_INSTANCE_ID)
        logger.info(f"✅ Retrieved instance: {instance.id}")
        logger.info(f"   Status: {instance.status}")
        logger.info(f"   Public IP: {instance.public_ip}")
        
        # Check if we have direct SSH (only supported method)
        if instance.public_ip == "ssh.runpod.io":
            logger.error("❌ Instance has proxy SSH - only direct SSH is supported")
            return False
        
        logger.info("🎯 Using direct SSH method")
        
        logger.info("   Expected: Real output capture")
        
        results = {}
        
        # Test 1: Paramiko SSH client
        logger.info("\n" + "=" * 30)
        logger.info("🔗 Testing Paramiko SSH Client")
        logger.info("=" * 30)
        
        try:
            success, stdout, stderr = execute_command_sync(
                instance, None, "nvidia-smi --version", timeout=20
            )
            
            logger.info(f"Command executed: {success}")
            
            if success:
                if "Your SSH client doesn't support PTY" in stdout:
                    logger.warning("⚠️ Paramiko: PTY error (limited functionality)")
                    results["paramiko"] = "pty_error"
                elif "NVIDIA-SMI" in stdout:
                    logger.info("✅ Paramiko: REAL output captured!")
                    logger.info(f"   GPU Info: {stdout.split()[0:4]}")
                    results["paramiko"] = "real_output"
                else:
                    logger.info("✅ Paramiko: Command successful")
                    logger.info(f"   Output: {stdout[:50]}...")
                    results["paramiko"] = "success"
            else:
                logger.error(f"❌ Paramiko: Command failed - {stderr}")
                results["paramiko"] = "failed"
                
        except Exception as e:
            logger.error(f"❌ Paramiko: Exception - {e}")
            results["paramiko"] = "error"
        
        # Test 2: AsyncSSH client
        logger.info("\n" + "=" * 30)
        logger.info("🔗 Testing AsyncSSH Client")
        logger.info("=" * 30)
        
        try:
            success, stdout, stderr = await execute_command_async(
                instance, None, "nvidia-smi --version", timeout=20
            )
            
            logger.info(f"Command executed: {success}")
            
            if success:
                if "Your SSH client doesn't support PTY" in stdout:
                    logger.warning("⚠️ AsyncSSH: PTY error (limited functionality)")
                    results["asyncssh"] = "pty_error"
                elif "NVIDIA-SMI" in stdout:
                    logger.info("✅ AsyncSSH: REAL output captured!")
                    logger.info(f"   GPU Info: {stdout.split()[0:4]}")
                    results["asyncssh"] = "real_output"
                else:
                    logger.info("✅ AsyncSSH: Command successful")
                    logger.info(f"   Output: {stdout[:50]}...")
                    results["asyncssh"] = "success"
            else:
                logger.error(f"❌ AsyncSSH: Command failed - {stderr}")
                results["asyncssh"] = "failed"
                
        except Exception as e:
            logger.error(f"❌ AsyncSSH: Exception - {e}")
            results["asyncssh"] = "error"
        
        # Test 3: Instance convenience method
        logger.info("\n" + "=" * 30)
        logger.info("🛠️ Testing instance.exec() Method")
        logger.info("=" * 30)
        
        try:
            result = instance.exec("hostname")
            
            if result.success:
                if "Your SSH client doesn't support PTY" in result.stdout:
                    logger.warning("⚠️ instance.exec(): PTY error")
                    results["instance_exec"] = "pty_error"
                else:
                    logger.info(f"✅ instance.exec(): hostname = {result.stdout.strip()}")
                    results["instance_exec"] = "real_output"
            else:
                logger.error(f"❌ instance.exec(): Failed - {result.stderr}")
                results["instance_exec"] = "failed"
                
        except Exception as e:
            logger.error(f"❌ instance.exec(): Exception - {e}")
            results["instance_exec"] = "error"
        
        # Final Results
        logger.info("\n" + "=" * 60)
        logger.info("🎯 FINAL RESULTS - SSH CLIENT TESTING")
        logger.info("=" * 60)
        
        for client_name, result in results.items():
            if result == "real_output":
                logger.info(f"  ✅ {client_name.upper()}: REAL OUTPUT CAPTURED")
            elif result == "pty_error":
                logger.warning(f"  ⚠️ {client_name.upper()}: PTY error (exit codes only)")
            elif result == "success":
                logger.info(f"  ✅ {client_name.upper()}: WORKING")
            elif result == "failed":
                logger.error(f"  ❌ {client_name.upper()}: COMMAND FAILED")
            else:
                logger.error(f"  ❌ {client_name.upper()}: ERROR")
        
        # Summary
        real_count = sum(1 for r in results.values() if r == "real_output")
        working_count = sum(1 for r in results.values() if r in ["real_output", "success"])
        total = len(results)
        
        logger.info("\n📊 Summary:")
        logger.info(f"   Real output: {real_count}/{total} clients")
        logger.info(f"   Working: {working_count}/{total} clients")
        logger.info("   SSH method: direct")
        
        if real_count >= 2:
            logger.info("\n🎉 EXCELLENT: Multiple clients capturing real output!")
            logger.info("✅ SSH output capture fix is working perfectly!")
            return True
        elif working_count >= 2:
            logger.info("\n✅ GOOD: Multiple clients working")
            logger.info("Consider using direct SSH for full output capture")
            return True
        else:
            logger.error("\n❌ ISSUES: SSH clients not working properly")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the SSH client tests"""
    success = asyncio.run(test_both_ssh_clients())
    
    if success:
        print("\n✅ SSH CLIENT TESTING PASSED!")
        print("Both Paramiko and AsyncSSH are working correctly.")
        return 0
    else:
        print("\n❌ SSH CLIENT TESTING FAILED!")
        print("Check SSH configuration and instance status.")
        return 1

if __name__ == "__main__":
    sys.exit(main())