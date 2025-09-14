#!/usr/bin/env python3
"""Deploy the gsm8k_remote_nnsight server to a remote GPU.

This is the example-local version of the deploy script.
It provisions a GPU, pushes code with the proper extras, and starts the server
in a tmux session named 'nnsight-server'.

Usage:
  python examples/gsm8k_remote_nnsight/deploy.py --name nano-ndif --port 8000 --model willcb/Qwen3-0.6B
"""

from __future__ import annotations

import os
import sys
import time
import argparse
from typing import Optional

from broker.client import GPUClient
from broker.types import InstanceStatus
from bifrost.client import BifrostClient
from broker import api as broker_api
from dotenv import load_dotenv


def find_existing_instance(name: str):
    instances = broker_api.list_instances(provider="runpod")
    for inst in instances:
        if inst.name == name and inst.status == InstanceStatus.RUNNING:
            return inst
    return None


def ensure_instance(name: str, port: int, gpu_type: Optional[str], manufacturer: Optional[str],
                    min_vram: int, max_price: float, cloud_type: Optional[str], max_attempts: int):
    inst = find_existing_instance(name)
    if inst:
        print(f"✅ Using existing instance: {inst.id} ({inst.gpu_type})")
        return inst

    print("🔎 Provisioning new instance with constraints…")
    gc = GPUClient()
    query = (gc.vram_gb >= min_vram) & (gc.price_per_hour <= max_price)
    if gpu_type:
        query = query & (gc.gpu_type == gpu_type)
    if manufacturer:
        query = query & (gc.manufacturer == manufacturer)

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
    print(f"🆔 Created instance: {inst.id}. Waiting to be running…")
    if not inst.wait_until_ready(timeout=900):
        raise TimeoutError("Instance did not reach RUNNING state in time")
    inst = broker_api.get_instance(inst.id)
    if not inst:
        raise RuntimeError("Failed to refresh instance details")
    print(f"✅ Instance ready: {inst.id} ({inst.public_ip}:{inst.ssh_port})")
    return inst


def deploy_and_run(inst, port: int, model: str, device_map: str, skip_bootstrap: bool) -> str:
    ssh = inst.ssh_connection_string()
    bc = BifrostClient(ssh)

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

    run_cmd = (
        "cd ~/.bifrost/workspace && "
        f"bash examples/gsm8k_remote_nnsight/start_server.sh --host 0.0.0.0 --port {port} --model {model} --device-map {device_map}"
    )
    tmux_cmd = f"tmux new-session -d -s nnsight-server '{run_cmd}'"
    bc.exec(tmux_cmd)
    print("✅ Launched tmux session 'nnsight-server'")
    return "nnsight-server"


def main():
    ap = argparse.ArgumentParser(description="Deploy gsm8k_remote_nnsight server to remote GPU")
    ap.add_argument("--name", default="nano-ndif")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--gpu-type", default=None)
    ap.add_argument("--manufacturer", default=None)
    ap.add_argument("--min-vram", type=int, default=12)
    ap.add_argument("--max-price", type=float, default=0.40)
    ap.add_argument("--cloud-type", default="secure")
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--model", default="willcb/Qwen3-0.6B")
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--runpod-api-key", default=None)
    ap.add_argument("--skip-bootstrap", action="store_true")
    ap.add_argument("--wait-health", action="store_true")
    ap.add_argument("--health-timeout", type=int, default=900)
    ap.add_argument("--run-smoke", action="store_true")
    args = ap.parse_args()

    load_dotenv()
    if args.runpod_api_key:
        os.environ["RUNPOD_API_KEY"] = args.runpod_api_key
    if not os.environ.get("RUNPOD_API_KEY"):
        print("ERROR: RUNPOD_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    inst = ensure_instance(
        args.name, args.port, args.gpu_type, args.manufacturer,
        args.min_vram, args.max_price, args.cloud_type, args.max_attempts
    )
    if not inst.wait_until_ssh_ready(timeout=900):
        print("ERROR: SSH not ready", file=sys.stderr)
        sys.exit(2)

    session_name = deploy_and_run(inst, args.port, args.model, args.device_map, args.skip_bootstrap)

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

    if args.wait_health and proxy_url:
        print("\n=== Waiting for health ===")
        start = time.time()
        ok = False
        ready_base = None
        for _ in range(args.health_timeout // 3):
            for base in [proxy_url, direct_url]:
                if not base:
                    continue
                try:
                    import urllib.request
                    with urllib.request.urlopen(f"{base}/health", timeout=5) as resp:
                        if resp.status == 200 and json.loads(resp.read().decode()).get("ok"):
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
        if args.run_smoke and ready_base:
            print("\n=== Running smoke test ===")
            import subprocess, sys as _sys
            code = subprocess.call([_sys.executable, 'examples/gsm8k_remote_nnsight/smoke.py', '--base-url', ready_base])
            if code != 0:
                print("❌ Smoke test failed", file=sys.stderr)
                sys.exit(code)
            print("🎉 Smoke test passed")


if __name__ == "__main__":
    main()

