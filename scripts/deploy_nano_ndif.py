#!/usr/bin/env python3
"""Idempotent deploy of nano-ndif to a remote GPU via broker + bifrost.

Behavior
- Reuse a running instance with the given name if found; otherwise provision one.
- Deploy current git repo to remote workspace and start examples.gsm8k_remote_nnsight server as a detached job.
- Prints connection info and logs hints.

Requirements
- RUNPOD_API_KEY in env (for broker RunPod provider)
"""

from __future__ import annotations

import os
import sys
import time
import argparse
import subprocess
from typing import Optional

# Local imports
from broker import api as broker_api
from broker.types import GPUInstance, InstanceStatus
from broker.client import GPUClient

from bifrost.deploy import GitDeployment
from bifrost.client import BifrostClient
from dotenv import load_dotenv


def find_existing_instance(name: str) -> Optional[GPUInstance]:
    instances = broker_api.list_instances(provider="runpod")
    for inst in instances:
        if inst.name == name and inst.status == InstanceStatus.RUNNING:
            return inst
    return None


def ensure_instance(
    name: str,
    port: int,
    gpu_type: Optional[str],
    manufacturer: Optional[str],
    min_vram: int,
    max_price: float,
    cloud_type: Optional[str] = None,
    max_attempts: int = 3,
) -> GPUInstance:
    inst = find_existing_instance(name)
    if inst:
        print(f"✅ Using existing instance: {inst.id} ({inst.gpu_type})")
        # Print specs if available
        try:
            vram = inst.raw_data.get("machine", {}).get("gpuType", {}).get("memoryInGb") if inst.raw_data else None
            manuf = inst.raw_data.get("machine", {}).get("gpuType", {}).get("manufacturer") if inst.raw_data else None
            price = inst.price_per_hour
            parts = [f"gpu={inst.gpu_type}"]
            if vram: parts.append(f"vram={vram}GB")
            if manuf: parts.append(f"mfg={manuf}")
            if price: parts.append(f"price=${price:.3f}/hr")
            print("   Specs: " + ", ".join(parts))
        except Exception:
            pass
        return inst

    print("🔎 No existing instance found; provisioning a new one with constraints…")
    # Use flexible GPUClient query for specs
    gc = GPUClient()
    query = (gc.vram_gb >= min_vram) & (gc.price_per_hour <= max_price)
    if gpu_type:
        try:
            query = query & (gc.gpu_type == gpu_type)
        except Exception:
            pass
    if manufacturer:
        try:
            query = query & (gc.manufacturer == manufacturer)
        except Exception:
            pass

    inst = gc.create(
        query=query,
        exposed_ports=[port],
        enable_http_proxy=True,
        name=name,
        cloud_type=cloud_type or "secure",
        sort=lambda x: x.price_per_hour,
        reverse=False,
        max_attempts=max_attempts,
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
    # Print specs if available
    try:
        vram = inst.raw_data.get("machine", {}).get("gpuType", {}).get("memoryInGb") if inst.raw_data else None
        manuf = inst.raw_data.get("machine", {}).get("gpuType", {}).get("manufacturer") if inst.raw_data else None
        price = inst.price_per_hour
        parts = [f"gpu={inst.gpu_type}", f"count={inst.gpu_count}"]
        if vram: parts.append(f"vram={vram}GB")
        if manuf: parts.append(f"mfg={manuf}")
        if price: parts.append(f"price=${price:.3f}/hr")
        print("   Specs: " + ", ".join(parts))
    except Exception:
        pass
    return inst


def deploy_and_run(inst: GPUInstance, port: int, model: str, skip_bootstrap: bool, device_map: str) -> str:
    """Old-style deploy: push with BifrostClient and run tmux session manually."""
    ssh = inst.ssh_connection_string()
    bc = BifrostClient(ssh)

    # Control bootstrap via env: use uv extras through push
    uv_extra = "examples_gsm8k_remote_nnsight"
    if skip_bootstrap:
        os.environ["BIFROST_SKIP_BOOTSTRAP"] = "1"
    else:
        os.environ.pop("BIFROST_SKIP_BOOTSTRAP", None)

    print("📦 Deploying code to remote workspace with extras…")
    workspace = bc.push(uv_extra=uv_extra)
    print(f"✅ Workspace ready: {workspace}")

    # Kill existing tmux session and free port
    bc.exec("tmux has-session -t nnsight-server 2>/dev/null && tmux kill-session -t nnsight-server || true")
    bc.exec(f"pids=$(lsof -ti:{port} 2>/dev/null || true); if [ -n \"$pids\" ]; then echo '🔪 Killing PIDs on port {port}: ' $pids; kill -9 $pids || true; fi")

    # Start server in tmux using start_server.sh, tee logs to ~/nnsight_server.log
    run_cmd = (
        "cd ~/.bifrost/workspace && "
        f"bash examples/gsm8k_remote_nnsight/server/start_server.sh --host 0.0.0.0 --port {port} --model {model} --device-map {device_map}"
    )
    tmux_cmd = f"tmux new-session -d -s nnsight-server '{run_cmd}'"
    bc.exec(tmux_cmd)
    print("✅ Launched tmux session 'nnsight-server'")
    print("📝 Debugging tips:")
    print(f"   - tmux capture: python -m bifrost.bifrost.exec '{ssh}' \"tmux capture-pane -pt nnsight-server -S -5000\"")
    print(f"   - Server log: ~/nnsight_server.log")
    return "nnsight-server"


def main():
    parser = argparse.ArgumentParser(description="Deploy nano-ndif to a remote GPU")
    parser.add_argument("--name", default="nano-ndif", help="Instance name to reuse or create")
    parser.add_argument("--port", type=int, default=8000, help="Service port to expose")
    parser.add_argument("--gpu-type", default=None, help="Filter GPU type (e.g., 'A10', 'RTX 4090')")
    parser.add_argument("--manufacturer", default=None, help="Filter GPU manufacturer (e.g., 'nvidia')")
    parser.add_argument("--min-vram", type=int, default=12, help="Minimum VRAM in GB (default: 12)")
    parser.add_argument("--max-price", type=float, default=0.40, help="Maximum price per hour (default: 0.40)")
    parser.add_argument("--cloud-type", default="secure", help="Cloud type (default: secure)")
    parser.add_argument("--max-attempts", type=int, default=3, help="Max provisioning attempts over top offers (default: 3)")
    parser.add_argument("--model", default="willcb/Qwen3-0.6B", help="Model ID")
    parser.add_argument("--device-map", default="auto", help="Device map for model loading (auto/cpu/cuda/…)")
    parser.add_argument("--runpod-api-key", default=None, help="RunPod API key (overrides .env)")
    parser.add_argument("--skip-bootstrap", action="store_true", help="Skip bifrost bootstrap and manage deps in command")
    parser.add_argument("--wait-health", action="store_true", help="Wait for /health to become ready after deploy")
    parser.add_argument("--health-timeout", type=int, default=900, help="Max seconds to wait for /health (default: 900)")
    parser.add_argument("--run-smoke", action="store_true", help="Run smoke test after health is ready")
    args = parser.parse_args()

    # Load .env and validate env for RunPod
    load_dotenv()
    if args.runpod_api_key:
        os.environ["RUNPOD_API_KEY"] = args.runpod_api_key
    if not os.environ.get("RUNPOD_API_KEY"):
        print("ERROR: RUNPOD_API_KEY env var not set (load from .env failed)", file=sys.stderr)
        sys.exit(1)

    # Ensure instance
    inst = ensure_instance(
        args.name,
        args.port,
        args.gpu_type,
        args.manufacturer,
        args.min_vram,
        args.max_price,
        args.cloud_type,
        args.max_attempts,
    )
    # Ensure SSH is ready (direct SSH)
    if not inst.wait_until_ssh_ready(timeout=900):
        print("ERROR: Instance did not become SSH-ready in time", file=sys.stderr)
        sys.exit(2)

    # Deploy and run
    session_name = deploy_and_run(inst, args.port, args.model, args.skip_bootstrap, args.device_map)

    # Print access info
    print("\n=== Access Info ===")
    print(f"SSH: {inst.ssh_connection_string()}")
    if inst.public_ip and inst.ssh_port and inst.ssh_username:
        print(f"SSH (cli): ssh -p {inst.ssh_port} {inst.ssh_username}@{inst.public_ip}")
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
    print("Logs:")
    print(f"  tmux capture: python -m bifrost.bifrost.exec '{inst.ssh_connection_string()}' \"tmux capture-pane -pt {session_name} -S -5000\"")
    print(f"  server log: python -m bifrost.bifrost.exec '{inst.ssh_connection_string()}' \"tail -n 200 ~/nnsight_server.log\"")

    # Optionally wait for health and run smoke test
    if args.wait_health:
        print("\n=== Waiting for health ===")
        base_urls = []
        if proxy_url:
            base_urls.append(proxy_url)
        if direct_url:
            base_urls.append(direct_url)
        if not base_urls:
            print("No reachable base URL to poll; skipping health wait.")
        else:
            start = time.time()
            ok = False
            while time.time() - start < args.health_timeout:
                for base in base_urls:
                    try:
                        import urllib.request
                        with urllib.request.urlopen(f"{base}/health", timeout=5) as resp:
                            body = resp.read().decode("utf-8", errors="ignore")
                            if resp.status == 200 and '"ok":true' in body:
                                print(f"✅ Health ready at {base}")
                                ok = True
                                ready_base = base
                                break
                    except Exception:
                        pass
                if ok:
                    break
                time.sleep(3)
            if not ok:
                print("❌ Health not ready before timeout", file=sys.stderr)
                sys.exit(3)

            if args.run_smoke:
                print("\n=== Running smoke test ===")
                code = subprocess.call([sys.executable, 'scripts/smoke_nano_ndif.py', '--base-url', ready_base])
                if code != 0:
                    print("❌ Smoke test failed", file=sys.stderr)
                    sys.exit(code)
                print("🎉 Smoke test passed")


if __name__ == "__main__":
    main()
