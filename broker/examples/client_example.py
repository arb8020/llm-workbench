#!/usr/bin/env python3
"""
GPU Broker Minimal - Client Interface Example

Shows the new client-based approach (recommended)
"""

from broker import GPUClient
from broker.types import CloudType

# Step 1: Create configured client
client = GPUClient(
    ssh_key_path="~/.ssh/id_ed25519",  # Your SSH key
    api_key=None  # Uses RUNPOD_API_KEY env var
)

print("✅ Client configured")
print(f"   SSH key: {client.get_ssh_key_path()}")

# Step 2: Search with client query interface  
offers = client.search(
    (client.cloud_type == CloudType.SECURE) & (client.price_per_hour < 0.40),
    sort=lambda x: x.memory_gb / x.price_per_hour,
    reverse=True
)

print(f"✅ Found {len(offers)} offers")
if not offers:
    print("No offers available")
    exit(1)

# Step 3: Create instance
instance = client.create(offers[:3])
if not instance:
    print("Failed to provision")
    exit(1)

print(f"✅ Created: {instance.id}")

# Step 4: Use client-configured SSH
if instance.wait_until_ssh_ready():
    print("🚀 SSH ready!")
    
    # Uses client's SSH key automatically
    result = instance.exec("nvidia-smi --version")
    if result.success:
        print(f"🎯 GPU ready: {result.stdout.split('version')[1].split()[1]}")

# Step 5: Clean up
instance.terminate()
print("🗑️ Cleaned up")

print("")
print("🎉 Client interface provides:")
print("  ✅ Explicit configuration (SSH keys, API keys)")
print("  ✅ No global state")  
print("  ✅ Multiple clients possible")
print("  ✅ Clear error messages")