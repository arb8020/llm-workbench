#!/usr/bin/env python3
"""
Cleanup utility to nuke all active GPU instances
"""

import sys
import logging
import requests
import os
from typing import List, Dict, Any

sys.path.insert(0, '../src')
sys.path.insert(0, 'src')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_api_key() -> str:
    """Get RunPod API key from environment or .env file"""
    api_key = os.getenv("RUNPOD_API_KEY")
    if api_key:
        return api_key
    
    # Try to read from .env file
    try:
        with open('.env', 'r') as f:
            for line in f:
                if line.startswith('RUNPOD_API_KEY='):
                    return line.split('=', 1)[1].strip()
    except FileNotFoundError:
        pass
    
    try:
        with open('../.env', 'r') as f:
            for line in f:
                if line.startswith('RUNPOD_API_KEY='):
                    return line.split('=', 1)[1].strip()
    except FileNotFoundError:
        pass
    
    raise ValueError("RUNPOD_API_KEY not found in environment or .env file")

def make_graphql_request(query: str, variables: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Make a GraphQL request to RunPod API"""
    api_key = get_api_key()
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {"query": query}
    if variables:
        payload["variables"] = variables
    
    response = requests.post(
        "https://api.runpod.io/graphql",
        json=payload,
        headers=headers,
        timeout=30
    )
    
    if response.status_code != 200:
        raise Exception(f"HTTP {response.status_code}: {response.text}")
    
    data = response.json()
    if "errors" in data:
        raise Exception(f"GraphQL errors: {data['errors']}")
    
    return data.get("data", {})

def list_all_pods() -> List[Dict[str, Any]]:
    """List all pods for the user"""
    query = """
    query {
        myself {
            pods {
                id
                name
                desiredStatus
                costPerHr
                gpuCount
                machine {
                    podHostId
                }
                createdAt
            }
        }
    }
    """
    
    try:
        data = make_graphql_request(query)
        return data.get("myself", {}).get("pods", [])
    except Exception as e:
        logger.error(f"Failed to list pods: {e}")
        return []

def terminate_pod(pod_id: str) -> bool:
    """Terminate a specific pod"""
    mutation = """
    mutation terminatePod($input: PodTerminateInput!) {
        podTerminate(input: $input)
    }
    """
    
    variables = {"input": {"podId": pod_id}}
    
    try:
        data = make_graphql_request(mutation, variables)
        return data.get("podTerminate") is not None
    except Exception as e:
        logger.error(f"Failed to terminate pod {pod_id}: {e}")
        return False

def cleanup_all_pods(force: bool = False) -> None:
    """Clean up all active pods"""
    logger.info("🔍 Scanning for active GPU instances...")
    
    pods = list_all_pods()
    
    if not pods:
        logger.info("✅ No pods found")
        return
    
    logger.info(f"Found {len(pods)} total pods:")
    
    active_pods = []
    for pod in pods:
        status = pod.get("desiredStatus", "UNKNOWN")
        cost = pod.get("costPerHr", 0)
        name = pod.get("name", "unnamed")
        created = pod.get("createdAt", "unknown")
        
        logger.info(f"  {pod['id']}: {name} - {status} - ${cost}/hr - {created}")
        
        if status in ["RUNNING", "PENDING"]:
            active_pods.append(pod)
    
    if not active_pods:
        logger.info("✅ No active pods to clean up")
        return
    
    logger.info(f"\n🚨 Found {len(active_pods)} ACTIVE pods to terminate")
    
    if not force:
        total_cost = sum(pod.get("costPerHr", 0) for pod in active_pods)
        logger.info(f"💰 Total hourly cost: ${total_cost:.2f}/hr")
        
        response = input(f"\nTerminate {len(active_pods)} active pods? [y/N]: ")
        if response.lower() != 'y':
            logger.info("❌ Cleanup cancelled")
            return
    
    logger.info("\n🗑️ Terminating active pods...")
    
    success_count = 0
    for pod in active_pods:
        pod_id = pod['id']
        name = pod.get('name', 'unnamed')
        
        logger.info(f"  Terminating {pod_id} ({name})...")
        
        if terminate_pod(pod_id):
            logger.info(f"    ✅ {pod_id} terminated")
            success_count += 1
        else:
            logger.error(f"    ❌ Failed to terminate {pod_id}")
    
    logger.info(f"\n🎯 Cleanup complete: {success_count}/{len(active_pods)} pods terminated")
    
    if success_count < len(active_pods):
        logger.warning("⚠️ Some pods failed to terminate - manual cleanup may be required")
        logger.info("💡 Try running the script again in a few seconds")

def main():
    """Main cleanup function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Cleanup RunPod GPU instances")
    parser.add_argument("--force", "-f", action="store_true", 
                       help="Force cleanup without confirmation")
    parser.add_argument("--list", "-l", action="store_true",
                       help="List pods without terminating")
    
    args = parser.parse_args()
    
    if args.list:
        logger.info("📋 Listing all pods:")
        pods = list_all_pods()
        for pod in pods:
            status = pod.get("desiredStatus", "UNKNOWN")
            cost = pod.get("costPerHr", 0)
            name = pod.get("name", "unnamed")
            logger.info(f"  {pod['id']}: {name} - {status} - ${cost}/hr")
        return
    
    logger.info("🧹 RunPod GPU Cleanup Utility")
    logger.info("This will terminate ALL active RunPod instances")
    
    cleanup_all_pods(force=args.force)

if __name__ == "__main__":
    main()