#!/usr/bin/env python3
"""
GPU Broker Minimal - Simple Example

The fastest way to get started:
1. Search for GPUs
2. Create an instance  
3. Use it
4. Clean up
"""

import gpu_broker_minimal as gpus
import time
from gpu_broker_minimal.types import CloudType

# Step 1: Find best value GPU under $0.40/hr in secure cloud
# Secure cloud has better availability and direct SSH
offers = gpus.search(
    (gpus.cloud_type == CloudType.SECURE) & (gpus.price_per_hour < 0.40),
    sort=lambda x: x.memory_gb / x.price_per_hour,  # Best GB per dollar
    reverse=True
)

print(f"Found {len(offers)} offers")
if not offers:
    print("No GPUs available - try higher price limit")
    exit(1)

# Step 2: Create instance (tries multiple offers automatically)
instance = gpus.create(offers[:3])  # Try top 3 offers
if not instance:
    print("Failed to provision - all offers unavailable")
    exit(1)

print(f"✅ Created: {instance.id}")
print(f"💰 Price: ${instance.price_per_hour:.3f}/hr")

# Step 3: Wait for SSH to be ready (blocks up to 5 minutes)
print("⏳ Waiting for instance and SSH to be ready...")
if instance.wait_until_ssh_ready():
    print("🚀 Instance and SSH ready!")
    
    # Step 4: Use it  
    result = instance.exec("nvidia-smi --version", timeout=60)
    if result.success and 'version' in result.stdout:
        version = result.stdout.split('version')[1].split()[1]
        print(f"🎯 GPU ready: Driver version {version}")
        
        # Check GPU temperature
        temp_result = instance.exec("nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits")
        if temp_result.success:
            print(f"📊 Current GPU temp: {temp_result.stdout.strip()}°C")
else:
    print("⚠️  SSH not ready within 5 minutes")

# Step 5: Clean up
instance.terminate()
print("🗑️  Cleaned up")