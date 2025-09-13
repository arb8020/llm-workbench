#!/usr/bin/env python3
"""Idempotent deploy of nano-ndif to a remote GPU via broker + bifrost.

Behavior
- Reuse a running instance with the given name if found; otherwise provision one.
- Deploy current git repo to remote workspace and start nano_ndif server as a detached job.
- Prints connection info and logs hints.

Requirements
- RUNPOD_API_KEY in env (for broker RunPod provider)
"""

from __future__ import annotations

import os
import sys
import time
import argparse
from typing import Optional

# Local imports
from broker import api as broker_api
from broker.types import GPUInstance, InstanceStatus

from bifrost.deploy import GitDeployment
from dotenv import load_dotenv


def find_existing_instance(name: str) -> Optional[GPUInstance]:
    instances = broker_api.list_instances(provider="runpod")
    for inst in instances:
        if inst.name == name and inst.status == InstanceStatus.RUNNING:
            return inst
    return None


def ensure_instance(name: str, port: int, gpu_type: Optional[str], manufacturer: Optional[str]) -> GPUInstance:
    inst = find_existing_instance(name)
    if inst:
        print(f"✅ Using existing instance: {inst.id} ({inst.gpu_type})")
        return inst

    print("🔎 No existing instance found; provisioning a new one...")
    # Provision with minimal constraints; expose the service port
    inst = broker_api.create(
        gpu_type=gpu_type,
        manufacturer=manufacturer,
        exposed_ports=[port],
        enable_http_proxy=True,
        name=name,
    )
    if not inst:
        raise RuntimeError("Failed to provision instance")
    print(f"🆔 Created instance: {inst.id}. Waiting to be running...")

    # Wait for RUNNING
    if not inst.wait_until_ready(timeout=900):
        raise TimeoutError("Instance did not reach RUNNING state in time")

    # Refresh details (IP/ports)
    inst = broker_api.get_instance(inst.id)
    if not inst:
        raise RuntimeError("Failed to refresh instance details")
    print(f"✅ Instance ready: {inst.id} ({inst.public_ip}:{inst.ssh_port})")
    return inst


def deploy_and_run(inst: GPUInstance, port: int, model: str, skip_bootstrap: bool) -> str:
    # Build command: manage dependencies explicitly using uv extras
    base_cmd = (
        "pip install -U uv && "
        "uv sync --extra examples_gsm8k_nnsight_remote && "
        f"NANO_NDIF_PORT={port} NANO_NDIF_MODEL={model} uv run python -m nano_ndif.server"
    )

    # Make deployment
    print(f"🚀 Deploying to {inst.ssh_username}@{inst.public_ip}:{inst.ssh_port}")
    deployment = GitDeployment(inst.ssh_username, inst.public_ip, inst.ssh_port)

    # Control bifrost bootstrap via env
    if skip_bootstrap:
        os.environ["BIFROST_SKIP_BOOTSTRAP"] = "1"

    job_id = deployment.deploy_and_execute_detached(base_cmd)
    print(f"✅ Started nano-ndif server job: {job_id}")
    return job_id


def main():
    parser = argparse.ArgumentParser(description="Deploy nano-ndif to a remote GPU")
    parser.add_argument("--name", default="nano-ndif", help="Instance name to reuse or create")
    parser.add_argument("--port", type=int, default=8002, help="Service port to expose")
    parser.add_argument("--gpu-type", default=None, help="Filter GPU type (e.g., 'A10', 'RTX 4090')")
    parser.add_argument("--manufacturer", default=None, help="Filter GPU manufacturer (e.g., 'nvidia')")
    parser.add_argument("--model", default=os.environ.get("NANO_NDIF_MODEL", "willcb/Qwen3-0.6B"), help="Model ID")
    parser.add_argument("--skip-bootstrap", action="store_true", help="Skip bifrost bootstrap and manage deps in command")
    args = parser.parse_args()

    # Load .env and validate env for RunPod
    load_dotenv()
    if not os.environ.get("RUNPOD_API_KEY"):
        print("ERROR: RUNPOD_API_KEY env var not set (load from .env failed)", file=sys.stderr)
        sys.exit(1)

    # Ensure instance
    inst = ensure_instance(args.name, args.port, args.gpu_type, args.manufacturer)
    # Ensure SSH is ready (direct SSH)
    if not inst.wait_until_ssh_ready(timeout=900):
        print("ERROR: Instance did not become SSH-ready in time", file=sys.stderr)
        sys.exit(2)

    # Deploy and run
    job_id = deploy_and_run(inst, args.port, args.model, args.skip_bootstrap)

    # Print access info
    print("\n=== Access Info ===")
    print(f"SSH: {inst.ssh_connection_string()}")
    proxy_url = inst.get_proxy_url(args.port)
    direct_url = f"http://{inst.public_ip}:{args.port}" if inst.public_ip else None
    if proxy_url:
        print(f"HTTP (proxy): {proxy_url}")
    if direct_url:
        print(f"HTTP (direct): {direct_url}")

    print("\n=== Next steps ===")
    if proxy_url:
        print(f"Health: curl {proxy_url}/health")
        print(f"Models: curl {proxy_url}/v1/models")
    if direct_url:
        print(f"Health: curl {direct_url}/health")
        print(f"Models: curl {direct_url}/v1/models")
    print(f"Logs: python -m bifrost.bifrost.jobs_cli logs '{inst.ssh_connection_string()}' {job_id} -f")


if __name__ == "__main__":
    main()
