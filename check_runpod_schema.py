#!/usr/bin/env python3
"""Check RunPod GraphQL schema for lowestPrice input parameters"""

import requests
import json
import os
from dotenv import load_dotenv

# Load environment
load_dotenv()

def get_schema_info():
    """Get GraphQL schema info from RunPod"""
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("❌ RUNPOD_API_KEY not set")
        return
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Introspection query to get schema info
    introspection_query = """
    query {
        __schema {
            types {
                name
                fields {
                    name
                    type {
                        name
                        kind
                        ofType {
                            name
                            kind
                        }
                    }
                    args {
                        name
                        type {
                            name
                            kind
                            ofType {
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
            json={"query": introspection_query}, 
            headers=headers
        )
        response.raise_for_status()
        
        schema = response.json()
        
        # Find GPU type related types
        for type_info in schema["data"]["__schema"]["types"]:
            if "gpu" in type_info["name"].lower() or "price" in type_info["name"].lower():
                print(f"\n=== {type_info['name']} ===")
                if type_info["fields"]:
                    for field in type_info["fields"]:
                        print(f"  {field['name']}")
                        if field["args"]:
                            print(f"    Args:")
                            for arg in field["args"]:
                                print(f"      - {arg['name']}: {arg['type']}")
        
    except Exception as e:
        print(f"❌ Error querying schema: {e}")

if __name__ == "__main__":
    get_schema_info()