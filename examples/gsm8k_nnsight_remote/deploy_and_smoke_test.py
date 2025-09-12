#!/usr/bin/env python3
"""
Push-button deploy + smoke test for single-pass NNsight server.

What it does:
- Provisions a GPU via broker with HTTP proxy on port 8001
- Pushes this repo via bifrost and installs extras
- Starts server in tmux session
- Waits for health, then:
  - POST /models/load for chosen model and savepoints
  - POST /v1/chat/completions to generate + capture activations
  - Verifies activation .pt files exist on remote
- Leaves GPU running for SSH debugging (prints connection info)
"""

import json
import os
import sys
import time
from typing import Dict, Any, List, Optional
import argparse

from broker.client import GPUClient
from broker.types import InstanceStatus
from bifrost.client import BifrostClient
import requests


def get_or_create_gpu(min_vram: int, max_price: float, name: str, gpu_id: Optional[str], reuse_existing: bool) -> Any:
    gpu_client = GPUClient()

    # If a specific GPU id is provided, try to reuse it
    if gpu_id:
        gpu = gpu_client.get_instance(gpu_id)
        if not gpu:
            raise RuntimeError(f"GPU with id {gpu_id} not found")
        # Ensure SSH is ready
        if not gpu.wait_until_ssh_ready(timeout=300):
            raise RuntimeError(f"SSH not ready for GPU {gpu_id}")
        return gpu

    # Optionally reuse an existing running instance with the same name
    if reuse_existing:
        try:
            candidates = [
                g for g in gpu_client.list_instances()
                if getattr(g, 'name', None) == name and getattr(g, 'status', None) == InstanceStatus.RUNNING
            ]
        except Exception:
            candidates = []
        if candidates:
            # Pick the first running one
            gpu = candidates[0]
            if not gpu.wait_until_ssh_ready(timeout=180):
                raise RuntimeError(f"Existing GPU {gpu.id} not SSH-ready")
            return gpu

    # Otherwise create a fresh instance
    query = (gpu_client.vram_gb >= min_vram) & (gpu_client.price_per_hour <= max_price)
    gpu = gpu_client.create(
        query=query,
        exposed_ports=[8001],
        enable_http_proxy=True,
        name=name,
        cloud_type="secure",
        sort=lambda x: x.price_per_hour,
        reverse=False,
    )
    if not gpu.wait_until_ssh_ready(timeout=300):
        raise RuntimeError("SSH did not become ready in time")
    return gpu


def start_server(bc: BifrostClient, skip_sync: bool = False, frozen_sync: bool = False) -> None:
    # Install deps from extras and run uvicorn in tmux
    # Control dependency bootstrap speed via env flags consumed by bifrost
    if skip_sync:
        os.environ["BIFROST_SKIP_BOOTSTRAP"] = "1"
    elif frozen_sync:
        os.environ["BIFROST_BOOTSTRAP_FROZEN"] = "1"

    workspace = bc.push(uv_extra="examples_gsm8k_nnsight_remote")
    # Kill existing session if present, then start a clean one
    bc.exec("tmux has-session -t nnsight-singlepass 2>/dev/null && tmux kill-session -t nnsight-singlepass || true")
    if skip_sync:
        run_cmd = (
            "cd ~/.bifrost/workspace && "
            "if [ -x .venv/bin/python ]; then "
            ". .venv/bin/activate; echo 'Using existing venv:'; which python; python -V; "
            "python examples/gsm8k_nnsight_remote/server_singlepass.py --host 0.0.0.0 --port 8001; "
            "else echo '❌ Missing .venv. Run once without --skip-sync or with --frozen-sync to create it.' >&2; exit 42; fi"
        )
    else:
        run_cmd = (
            "cd ~/.bifrost/workspace && "
            "uv run python examples/gsm8k_nnsight_remote/server_singlepass.py --host 0.0.0.0 --port 8001"
        )
    tmux_cmd = f"tmux new-session -d -s nnsight-singlepass '{run_cmd} 2>&1 | tee ~/nnsight_singlepass.log'"
    bc.exec(tmux_cmd)

def wait_for_remote_health(bc: BifrostClient, timeout_s: int = 240) -> None:
    deadline = time.time() + timeout_s
    last = None
    while time.time() < deadline:
        try:
            out = bc.exec("curl -s -o /dev/null -w '%{http_code}' http://localhost:8001/health")
            code = out.strip()
            if code == "200":
                return
            last = code
        except Exception as e:
            last = str(e)
        time.sleep(2)
    logs = bc.exec("tail -n 200 ~/nnsight_singlepass.log || true")
    raise RuntimeError(f"Remote /health not ready (last={last}). Recent logs:\n{logs}")


def wait_for_health(url: str, timeout_s: int = 180) -> None:
    deadline = time.time() + timeout_s
    last_err = None
    while time.time() < deadline:
        try:
            r = requests.get(f"{url}/health", timeout=5)
            if r.status_code == 200:
                return
            last_err = f"{r.status_code} {r.text}"
        except Exception as e:
            last_err = str(e)
        time.sleep(2)
    raise RuntimeError(f"Server health never ready: {last_err}")


def _remote_post_json(bc: BifrostClient, remote_url: str, payload: Dict[str, Any], timeout_s: int = 180) -> Dict[str, Any]:
    data = json.dumps(payload)
    # Use curl on the remote host to bypass any external proxy method restrictions
    cmd = (
        f"curl -s -X POST {remote_url} "
        f"-H 'Content-Type: application/json' "
        f"--max-time {timeout_s} "
        f"-d '{data}'"
    )
    out = bc.exec(cmd)
    try:
        return json.loads(out)
    except Exception as e:
        raise RuntimeError(f"Remote POST failed or returned non-JSON. Output: {out[:300]}…")


def run_smoke(bc: BifrostClient, proxy_url: str, model_id: str) -> Dict[str, Any]:
    # Use proxy only for health; POST via remote curl to avoid 405s from provider proxy
    remote_base = "http://localhost:8001"

    # Load model with 1-2 safe savepoints
    sp = [
        {"name": "logits", "selector": "output.logits"},
        {"name": "layer0_in", "selector": "model.layers[0].input_layernorm.output"},
    ]
    load_payload = {"model_id": model_id, "device_map": "auto", "savepoints": sp}
    load_resp = _remote_post_json(bc, f"{remote_base}/models/load", load_payload, timeout_s=300)
    if not load_resp.get("ok"):
        raise RuntimeError(f"/models/load reported failure: {load_resp}")

    # Chat completion
    chat_req = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Say hello and count to three."},
        ],
        "max_tokens": 32,
        "temperature": 0.1,
        "store_activations": True,
    }
    chat_resp = _remote_post_json(bc, f"{remote_base}/v1/chat/completions", chat_req, timeout_s=600)
    return chat_resp


def verify_files(bc: BifrostClient, resp: Dict[str, Any]) -> List[str]:
    files = []
    try:
        msg = resp["choices"][0]["message"]
        act = msg.get("activation_files", {})
        for _, fpath in act.items():
            out = bc.exec(f"test -f '{fpath}' && echo OK || echo MISSING")
            if "OK" in out:
                files.append(fpath)
    except Exception:
        pass
    return files


def main():
    p = argparse.ArgumentParser(description="Deploy single-pass NNsight server and smoke test it")
    p.add_argument("--model", default="willcb/Qwen3-0.6B")
    p.add_argument("--min-vram", type=int, default=12)
    p.add_argument("--max-price", type=float, default=0.60)
    p.add_argument("--gpu-id", default=None, help="Reuse an existing GPU by id")
    p.add_argument("--reuse", action="store_true", help="Reuse running instance named 'nnsight-singlepass-server' if found")
    p.add_argument("--name", default="nnsight-singlepass-server", help="Name to assign or search for when reusing")
    p.add_argument("--skip-sync", action="store_true", help="Skip uv sync on reuse (fastest; assumes env already set up)")
    p.add_argument("--frozen-sync", action="store_true", help="Run uv sync --frozen (use lock only; faster and reproducible)")
    args = p.parse_args()

    print("🚀 Getting GPU (reuse or create)…")
    gpu = get_or_create_gpu(args.min_vram, args.max_price, args.name, args.gpu_id, args.reuse)
    ssh = gpu.ssh_connection_string()
    proxy_url = gpu.get_proxy_url(8001) or f"http://{gpu.public_ip}:8001"
    print(f"✅ GPU: {gpu.id}\nSSH: {ssh}\nURL: {proxy_url}")

    bc = BifrostClient(ssh)
    print("📦 Deploying code + starting server (tmux)…")
    start_server(bc, skip_sync=args.skip_sync, frozen_sync=args.frozen_sync)

    print("⏳ Waiting for remote /health…")
    try:
        wait_for_remote_health(bc)
        print("✅ Remote health ready")
    except Exception as e:
        print(str(e))
        raise

    print("⏳ Waiting for external /health…")
    try:
        wait_for_health(proxy_url)
        print("✅ External health ready")
    except Exception as e:
        print("⚠️ External health check failed (proxy may block). Continuing.")

    print("💬 Running smoke test…")
    resp = run_smoke(bc, proxy_url, args.model)
    print("📝 Chat response id:", resp.get("id"))
    print("🧪 Checking activation files on remote…")
    present = verify_files(bc, resp)
    if present:
        print("🎉 Found activation files:")
        for pth in present:
            print("   ", pth)
    else:
        print("⚠️ No activation files found via server response. Inspect logs:")
        print("   bifrost exec", ssh, "'cat ~/nnsight_singlepass.log' ")

    print("🔒 Leaving GPU running for debugging.")
    print("   Attach tmux:  bifrost exec", ssh, "'tmux attach -t nnsight-singlepass'")
    print("   Tail logs:    bifrost exec", ssh, "'tail -n 200 -f ~/nnsight_singlepass.log'")
    print("   Health:       curl -s", f"{proxy_url}/health")
    print("   Terminate:    broker terminate", gpu.id)


if __name__ == "__main__":
    main()
