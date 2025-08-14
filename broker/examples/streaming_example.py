#!/usr/bin/env python3
"""
Example showing streaming vs non-streaming execution
"""

import sys
import os

# Add the broker package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import gpu_broker_minimal as gpus


def streaming_callback(line, is_stderr):
    """Callback function for real-time output"""
    prefix = "STDERR" if is_stderr else "STDOUT"
    print(f"[{prefix}] {line}", flush=True)


def main():
    print("🚀 GPU Broker Streaming Example")
    print("=" * 50)
    
    try:
        # Search for available GPUs
        print("\n1. Searching for available GPUs...")
        offers = gpus.search()
        if not offers:
            print("❌ No GPU offers found")
            return
        
        print(f"✅ Found {len(offers)} GPU offers")
        
        # Get cheapest GPU
        cheapest = min(offers, key=lambda x: x.price_per_hour)
        print(f"💰 Cheapest: {cheapest.gpu_type} at ${cheapest.price_per_hour:.3f}/hr")
        
        # For demo purposes, let's create an instance and test streaming
        # (Remove these lines if you don't want to actually provision)
        """
        print(f"\n2. Provisioning {cheapest.gpu_type}...")
        instance = gpus.create([cheapest])
        
        if not instance:
            print("❌ Failed to provision instance")
            return
        
        print(f"✅ Instance created: {instance.id}")
        
        try:
            # Wait for SSH to be ready
            print("\n3. Waiting for SSH to be ready...")
            if instance.wait_until_ssh_ready(timeout=180):
                print("✅ SSH is ready")
                
                # Create SSH client for streaming demo
                from gpu_broker_minimal.ssh_clients import ParamikoSSHClient, get_ssh_connection_info, SSHMethod
                
                hostname, port, username = get_ssh_connection_info(instance, SSHMethod.DIRECT)
                client = ParamikoSSHClient()
                
                if client.connect(hostname, port, username):
                    print("\n4. Comparing streaming vs non-streaming execution:")
                    
                    # Non-streaming execution
                    print("\n📦 Non-streaming (traditional):")
                    success, stdout, stderr = client.execute("echo 'Line 1'; sleep 1; echo 'Line 2'; sleep 1; echo 'Line 3'")
                    print("Result (all at once):")
                    print(stdout)
                    
                    # Streaming execution with callback
                    print("\n🌊 Streaming (real-time):")
                    success, stdout, stderr = client.execute_streaming(
                        "echo 'Line 1'; sleep 1; echo 'Line 2'; sleep 1; echo 'Line 3'",
                        output_callback=streaming_callback
                    )
                    print("Final accumulated result:")
                    print(f"Exit code: {0 if success else 1}")
                    
                    client.close()
                else:
                    print("❌ Failed to connect via SSH")
            else:
                print("❌ SSH not ready after timeout")
                
        finally:
            # Cleanup
            print(f"\n5. Cleaning up instance {instance.id}")
            instance.terminate()
            print("✅ Instance terminated")
        """
        
        print("\n💡 To test streaming with a real instance, uncomment the demo code above")
        print("   This will provision a GPU, test streaming, and clean up.")
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()