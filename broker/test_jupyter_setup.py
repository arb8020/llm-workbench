#!/usr/bin/env python3
"""
Test script to verify Jupyter auto-start functionality
"""

import time
import requests
from broker.client import GPUClient
from broker.types import CloudType

def test_jupyter_provisioning():
    """Test provisioning instance with Jupyter auto-start"""
    
    print("🔍 Searching for cheap GPU offers...")
    
    # Create client
    client = GPUClient()
    
    # Search for cheapest available GPU
    offers = client.search(
        (client.cloud_type == CloudType.SECURE) & (client.price_per_hour < 1.0),
        sort=lambda x: x.price_per_hour,  # Cheapest first
        reverse=False
    )
    
    if not offers:
        print("❌ No suitable offers found")
        return None
        
    print(f"✅ Found {len(offers)} offers, cheapest: {offers[0].gpu_type} at ${offers[0].price_per_hour:.3f}/hr")
    
    # Create instance with Jupyter enabled
    print("🚀 Creating instance with Jupyter auto-start...")
    instance = client.create(
        offers[:1],  # Try the cheapest offer
        start_jupyter=True,
        jupyter_password="test123",
        exposed_ports=[8888],
        enable_http_proxy=True,
        name="jupyter-test"
    )
    
    if not instance:
        print("❌ Failed to create instance")
        return None
        
    print(f"✅ Instance created: {instance.id}")
    print(f"💰 Price: ${instance.price_per_hour:.3f}/hr")
    
    return instance

def test_jupyter_accessibility(instance, max_wait_minutes=10):
    """Test if Jupyter is accessible via proxy URL"""
    
    print(f"⏳ Waiting for instance to be ready...")
    if not instance.wait_until_ready(timeout=300):  # 5 minutes
        print("❌ Instance not ready within 5 minutes")
        return False
        
    print("✅ Instance is running")
    
    # Get proxy URL
    proxy_url = instance.get_proxy_url(8888)
    if not proxy_url:
        print("❌ Could not get proxy URL")
        return False
        
    print(f"🔗 Testing Jupyter at: {proxy_url}")
    
    # Test accessibility with retries
    for attempt in range(max_wait_minutes * 2):  # Check every 30 seconds
        try:
            print(f"   Attempt {attempt + 1}/{max_wait_minutes * 2}...")
            response = requests.get(proxy_url, timeout=10, allow_redirects=True)
            
            # Check if response looks like Jupyter
            response_text = response.text.lower()
            if any(keyword in response_text for keyword in ['jupyter', 'lab', 'notebook']):
                print(f"✅ Jupyter is accessible at {proxy_url}")
                print(f"   Response status: {response.status_code}")
                return True
            elif response.status_code == 200:
                print(f"⚠️  Got 200 but doesn't look like Jupyter: {response_text[:200]}...")
            else:
                print(f"   Status: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"   Connection failed: {e}")
            
        if attempt < max_wait_minutes * 2 - 1:
            time.sleep(30)  # Wait 30 seconds before retry
    
    print("❌ Jupyter not accessible after waiting")
    return False

def main():
    """Main test function"""
    print("🧪 Testing Broker + Jupyter Integration")
    print("=" * 50)
    
    # Step 1: Provision instance
    instance = test_jupyter_provisioning()
    if not instance:
        return False
        
    try:
        # Step 2: Test accessibility
        success = test_jupyter_accessibility(instance)
        
        if success:
            print("\n🎉 SUCCESS! Jupyter integration works!")
            print(f"🔗 Jupyter URL: {instance.get_proxy_url(8888)}")
            
            # Additional info for SSH tunnel testing
            print(f"\n📡 SSH Connection Info:")
            print(f"   Host: {instance.public_ip}")
            print(f"   Port: {instance.ssh_port}")
            print(f"   User: {instance.ssh_username}")
            if instance.public_ip and instance.ssh_port:
                print(f"   SSH Tunnel: ssh -p {instance.ssh_port} {instance.ssh_username}@{instance.public_ip} -L 8888:localhost:8888")
                print(f"   Local URL: http://localhost:8888/?token=test123")
        else:
            print("\n❌ Test failed - Jupyter not accessible")
            
        return success
        
    finally:
        # Step 3: Cleanup
        print(f"\n🗑️  Cleaning up instance {instance.id}...")
        if instance.terminate():
            print("✅ Instance terminated successfully")
        else:
            print("⚠️  Failed to terminate instance - please check manually")

if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)