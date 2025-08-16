#!/usr/bin/env python3
"""
GPU Broker Minimal - Basic Workflow Example

This example shows the complete lifecycle:
1. Search for GPUs with custom criteria and sorting
2. Create an instance from the search results  
3. Wait for the instance to be ready with SSH
4. Run commands via SSH to verify GPU access
5. Clean up the instance

This is the fundamental workflow that most users will follow.
"""

import broker as gpus
import asyncio
import time

async def main():
    print("🚀 GPU Broker Minimal - Basic Workflow")
    print("=" * 50)
    
    instance = None
    
    try:
        # Step 1: Search for GPUs
        print("\n📋 Step 1: Search for GPUs")
        print("Looking for best value GPUs under $0.40/hr...")
        
        # Search with custom sorting - best memory per dollar
        offers = gpus.search(
            gpus.price_per_hour < 0.40,
            sort=lambda x: x.memory_gb / x.price_per_hour,  # GB per dollar
            reverse=True  # Highest value first
        )
        
        print(f"Found {len(offers)} offers")
        for i, offer in enumerate(offers[:3]):
            value = offer.memory_gb / offer.price_per_hour
            print(f"  {i+1}. {offer.gpu_type}: {offer.memory_gb}GB @ ${offer.price_per_hour:.3f}/hr = {value:.1f} GB/$")
        
        if not offers:
            print("❌ No offers found - try increasing price limit")
            return
        
        # Step 2: Create instance
        print("\n🔨 Step 2: Create GPU instance")
        print("Trying to provision from top offers...")
        
        # Create will try multiple offers automatically
        instance = gpus.create(
            offers[:5],  # Try top 5 offers
            image="runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04"
        )
        
        if not instance:
            print("❌ Failed to provision any instance")
            return
        
        print(f"✅ Instance created!")
        print(f"   ID: {instance.id}")
        print(f"   Status: {instance.status}")
        print(f"   GPU: {instance.gpu_type}")
        print(f"   Price: ${instance.price_per_hour:.3f}/hr")
        
        # Step 3: Wait for instance to be ready
        print("\n⏳ Step 3: Wait for instance to be ready")
        
        max_wait = 300  # 5 minutes
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            # Refresh instance data
            fresh_instance = gpus.get_instance(instance.id)
            if not fresh_instance:
                print("❌ Instance disappeared")
                return
            
            instance = fresh_instance
            print(f"   Status: {instance.status}")
            
            # Check if running and has connectivity info
            if (str(instance.status) == "InstanceStatus.RUNNING" and 
                instance.public_ip and 
                instance.ssh_port):
                print(f"✅ Instance ready!")
                print(f"   Public IP: {instance.public_ip}")
                print(f"   SSH Port: {instance.ssh_port}")
                break
            
            print(f"   Waiting... ({time.time() - start_time:.0f}s elapsed)")
            await asyncio.sleep(15)
        else:
            print("❌ Instance not ready after 5 minutes")
            return
        
        # Step 4: Test SSH connectivity
        print("\n🔗 Step 4: Test SSH connectivity")
        
        # Check SSH method
        if instance.public_ip == "ssh.runpod.io":
            print("⚠️  Instance has proxy SSH - minimal version only supports direct SSH")
            print("   This is expected behavior - proxy SSH has output capture limitations")
            print("   In production, you might retry provisioning to get direct SSH")
        else:
            print(f"✅ Direct SSH available: {instance.public_ip}:{instance.ssh_port}")
            
            # Wait for SSH daemon to be ready
            print("   Waiting 30s for SSH daemon to initialize...")
            await asyncio.sleep(30)
            
            # Test basic command
            print("   Testing hostname command...")
            result = instance.exec("hostname")
            
            if result.success and result.stdout.strip():
                print(f"   ✅ SSH working! Hostname: {result.stdout.strip()}")
                
                # Test GPU command
                print("   Testing GPU access...")
                gpu_result = instance.exec("nvidia-smi --version")
                
                if gpu_result.success and "NVIDIA-SMI" in gpu_result.stdout:
                    print("   ✅ GPU accessible!")
                    # Extract driver version
                    lines = gpu_result.stdout.split('\n')
                    for line in lines:
                        if "NVIDIA-SMI" in line and "version" in line:
                            print(f"   📊 {line.strip()}")
                            break
                else:
                    print(f"   ⚠️  GPU command issues: {gpu_result.stderr}")
            else:
                print(f"   ❌ SSH issues: {result.stderr}")
        
        print("\n🎉 Workflow completed successfully!")
        print("Instance is ready for your workloads.")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Step 5: Cleanup
        if instance:
            print(f"\n🗑️  Cleaning up instance {instance.id}...")
            try:
                if instance.terminate():
                    print("✅ Instance terminated")
                else:
                    print("⚠️  Failed to terminate - please check manually")
            except Exception as e:
                print(f"❌ Cleanup error: {e}")

def simple_sync_example():
    """Simpler synchronous example for basic users"""
    print("\n" + "="*50)
    print("🔥 Quick Sync Example")
    print("="*50)
    
    # Search and create in one step
    instance = gpus.create(
        gpus.gpu_type.contains("RTX") & (gpus.price_per_hour < 0.30),
        max_attempts=3
    )
    
    if instance:
        print(f"✅ Created: {instance.id}")
        print(f"💰 Price: ${instance.price_per_hour:.3f}/hr")
        
        # Do work here...
        
        # Cleanup
        instance.terminate()
        print("🗑️  Cleaned up")
    else:
        print("❌ No instances available")

if __name__ == "__main__":
    print("Choose example:")
    print("1. Full async workflow (recommended)")
    print("2. Quick sync example")
    
    choice = input("Enter 1 or 2: ").strip()
    
    if choice == "2":
        simple_sync_example()
    else:
        asyncio.run(main())