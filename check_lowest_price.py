#!/usr/bin/env python3
"""Check RunPod lowestPrice input schema"""

import requests
import json
import os
from dotenv import load_dotenv

# Load environment
load_dotenv()

def check_lowest_price_schema():
    """Check what parameters lowestPrice accepts"""
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("❌ RUNPOD_API_KEY not set")
        return
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Query to get schema info for lowestPrice field specifically
    schema_query = """
    query {
        __type(name: "GpuType") {
            fields {
                name
                args {
                    name
                    type {
                        name
                        kind
                        inputFields {
                            name
                            type {
                                name
                                kind
                            }
                        }
                    }
                }
            }
        }
    }
    """
    
    try:
        response = requests.post(
            "https://api.runpod.io/graphql", 
            json={"query": schema_query}, 
            headers=headers
        )
        response.raise_for_status()
        
        data = response.json()
        print("Schema response:")
        print(json.dumps(data, indent=2))
        
    except Exception as e:
        print(f"❌ Error: {e}")
        
        # Let's try a simpler approach - test different parameters directly
        print("\n🔍 Testing different lowestPrice parameters...")
        
        test_params = [
            "{ gpuCount: 1 }",
            "{ gpuCount: 1, minMemoryInGb: 8 }",
            "{ gpuCount: 1, containerDiskInGb: 50 }",
            "{ gpuCount: 1, minVcpuCount: 1 }",
            "{ gpuCount: 1, minMemoryInGb: 8, containerDiskInGb: 50 }"
        ]
        
        for params in test_params:
            test_query = f"""
            query {{
                gpuTypes(limit: 1) {{
                    id
                    displayName
                    lowestPrice(input: {params}) {{
                        minimumBidPrice
                        uninterruptablePrice
                    }}
                }}
            }}
            """
            
            try:
                response = requests.post(
                    "https://api.runpod.io/graphql",
                    json={"query": test_query},
                    headers=headers
                )
                if response.status_code == 200:
                    result = response.json()
                    if "errors" not in result:
                        print(f"✅ {params} - WORKS")
                    else:
                        print(f"❌ {params} - ERROR: {result['errors']}")
                else:
                    print(f"❌ {params} - HTTP {response.status_code}")
            except Exception as e:
                print(f"❌ {params} - Exception: {e}")

if __name__ == "__main__":
    check_lowest_price_schema()