#!/usr/bin/env python3
"""
Helper script to set up SSH connection for bifrost testing.

This script helps users provision a GPU instance via broker and get the SSH connection
string that can be used for bifrost testing.
"""

import argparse
import sys
import time
from pathlib import Path

# Add paths for broker imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "broker"))

def provision_gpu_for_testing():
    """Provision a GPU instance and return SSH connection info."""
    try:
        from broker import GPUClient
        from broker.types import CloudType
        
        print("🚀 Provisioning GPU instance for bifrost SSH testing...")
        
        client = GPUClient()
        
        # Search for affordable secure cloud GPUs (required for direct SSH) - same as broker test
        print("🔍 Searching for suitable GPU offers...")
        query = (client.cloud_type == CloudType.SECURE) & (client.price_per_hour < 0.50)
        offers = client.search(query)
        
        if not offers:
            print("❌ No affordable secure cloud GPUs found")
            print("   Secure cloud required for direct SSH with output capture")
            return None, None
        
        print(f"✅ Found {len(offers)} suitable offers")
        print(f"   Best offer: {offers[0].gpu_type} at ${offers[0].price_per_hour:.3f}/hr")
        print(f"   Memory: {offers[0].memory_gb}GB")
        
        # Store the best offer for provisioning (same as broker test)
        best_offer = offers[0]
        
        # Provision instance
        print("🚀 Provisioning GPU instance...")
        instance = client.create(best_offer)
        
        if not instance:
            print("❌ GPU provisioning failed")
            return None, None
        
        print(f"✅ Instance created: {instance.id}")
        print(f"   GPU: {instance.gpu_type}")
        print(f"   Status: {instance.status}")
        
        # Wait for instance to be ready with direct SSH (using broker test logic)
        print("⏳ Waiting for instance to be ready with direct SSH...")
        max_wait = 300  # 5 minutes
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            # Refresh instance status
            fresh_instance = client.get_instance(instance.id)
            if not fresh_instance:
                print("❌ Instance disappeared during provisioning")
                return None, None
            
            instance = fresh_instance
            print(f"   Status: {instance.status}")
            print(f"   Public IP: {instance.public_ip}")
            
            # Check if running with public IP
            if (str(instance.status) == "InstanceStatus.RUNNING" and 
                instance.public_ip and 
                instance.ssh_port):
                
                # Check SSH type - keep trying until we get direct SSH (like broker test)
                if instance.public_ip == "ssh.runpod.io":
                    print("⚠️ Got proxy SSH - need direct SSH for proper testing")
                    print("   Will keep provisioning until we get direct SSH...")
                    
                    # Terminate proxy SSH instance
                    instance.terminate()
                    time.sleep(5)
                    
                    # Keep trying to get direct SSH (up to 5 attempts) - same as broker test
                    max_attempts = 5
                    for attempt in range(max_attempts):
                        print(f"   Attempt {attempt + 1}/{max_attempts} for direct SSH...")
                        
                        # Provision new instance
                        retry_instance = client.create(best_offer)
                        if not retry_instance:
                            print(f"   Failed to provision on attempt {attempt + 1}")
                            continue
                        
                        # Wait for it to be ready and check SSH type (same timing as broker test)
                        retry_start = time.time()
                        got_direct_ssh = False
                        
                        while time.time() - retry_start < 180:  # 3 minutes per attempt
                            time.sleep(15)
                            fresh_retry = client.get_instance(retry_instance.id)
                            
                            if (fresh_retry and 
                                str(fresh_retry.status) == "InstanceStatus.RUNNING" and 
                                fresh_retry.public_ip and 
                                fresh_retry.ssh_port):
                                
                                if fresh_retry.public_ip != "ssh.runpod.io":
                                    print(f"✅ Got direct SSH on attempt {attempt + 1}!")
                                    print(f"   SSH: root@{fresh_retry.public_ip}:{fresh_retry.ssh_port}")
                                    instance = fresh_retry
                                    got_direct_ssh = True
                                    break
                                else:
                                    print(f"   Attempt {attempt + 1}: Still proxy SSH, terminating...")
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
                    if not got_direct_ssh or instance.public_ip == "ssh.runpod.io":
                        print(f"❌ Could not get direct SSH after {max_attempts} attempts")
                        print("   RunPod may not have direct SSH instances available right now")
                        print("   This is a RunPod infrastructure limitation, not a bifrost issue")
                        return None, None
                
                print("✅ Got direct SSH - full output capture expected")
                print(f"   SSH: root@{instance.public_ip}:{instance.ssh_port}")
                
                # Wait additional time for SSH daemon to be ready (same as broker test)
                print("⏳ Waiting 30s for SSH daemon to initialize...")
                time.sleep(30)
                
                ssh_connection = f"root@{instance.public_ip}:{instance.ssh_port}"
                return instance, ssh_connection
            
            time.sleep(15)
        
        print("❌ Instance not ready after 5 minutes")
        if instance:
            instance.terminate()
        return None, None
        
    except Exception as e:
        print(f"❌ Error provisioning GPU: {e}")
        return None, None


def test_ssh_connection(ssh_connection: str):
    """Test that SSH connection works using shared foundation."""
    try:
        # Test using bifrost compatibility layer
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from bifrost.ssh_clients_compat import test_ssh_connection, execute_command_sync
        
        print(f"🧪 Testing SSH connection: {ssh_connection}")
        
        # Test connection
        success, message = test_ssh_connection(ssh_connection)
        if success:
            print("✅ SSH connection test passed!")
        else:
            print(f"❌ SSH connection test failed: {message}")
            return False
        
        # Test basic command
        print("🧪 Testing basic command execution...")
        exit_code, stdout, stderr = execute_command_sync(ssh_connection, "echo 'SSH_TEST_SUCCESS'")
        
        if exit_code == 0 and "SSH_TEST_SUCCESS" in stdout:
            print("✅ Basic command execution working!")
        else:
            print(f"❌ Command execution failed: exit_code={exit_code}, stderr={stderr}")
            return False
        
        # Test nvidia-smi
        print("🧪 Testing nvidia-smi...")
        exit_code, stdout, stderr = execute_command_sync(ssh_connection, "nvidia-smi --version")
        
        if exit_code == 0 and "NVIDIA-SMI" in stdout:
            print("✅ nvidia-smi working - GPU access confirmed!")
            print(f"   Version: {stdout.split()[0] if stdout else 'Unknown'}")
        else:
            print(f"⚠️ nvidia-smi test failed: exit_code={exit_code}, stderr={stderr}")
        
        return True
        
    except Exception as e:
        print(f"❌ SSH connection test error: {e}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Helper script to set up SSH connection for bifrost testing"
    )
    parser.add_argument(
        "--provision", 
        action="store_true",
        help="Provision a new GPU instance and set up SSH connection"
    )
    parser.add_argument(
        "--test", 
        type=str,
        help="Test an existing SSH connection string (user@host:port)"
    )
    parser.add_argument(
        "--setup-and-test",
        action="store_true", 
        help="Provision GPU and run full test suite"
    )
    
    args = parser.parse_args()
    
    if args.provision or args.setup_and_test:
        instance, ssh_connection = provision_gpu_for_testing()
        
        if not ssh_connection:
            print("❌ Failed to provision GPU with direct SSH")
            return 1
        
        print("\n" + "="*60)
        print("🎉 GPU INSTANCE READY FOR BIFROST TESTING!")
        print("="*60)
        print(f"SSH Connection: {ssh_connection}")
        print(f"Instance ID: {instance.id}")
        print("")
        print("Usage:")
        print(f"  python tests/test_bifrost_sync.py {ssh_connection}")
        print(f"  python tests/test_bifrost_async.py {ssh_connection}")
        print("")
        print("⚠️  Remember to terminate the instance when done:")
        print(f"  broker terminate {instance.id}")
        print("="*60)
        
        if args.setup_and_test:
            print("\n🧪 Running SSH connection tests...")
            if test_ssh_connection(ssh_connection):
                print("\n✅ SSH setup complete and tested!")
            else:
                print("\n❌ SSH setup complete but testing failed")
                return 1
    
    elif args.test:
        if test_ssh_connection(args.test):
            print("\n✅ SSH connection test passed!")
        else:
            print("\n❌ SSH connection test failed")
            return 1
    
    else:
        parser.print_help()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())