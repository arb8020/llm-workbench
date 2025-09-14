#!/usr/bin/env python3
"""Simple CLI to test an NNsight server.

Features
- Checks /health and /v1/models
- Configures /v1/interventions
- Sends a chat completion
- Prints example curl commands for manual testing
- Optionally downloads activation artifacts via SSH (using BifrostClient)

Usage
  python examples/gsm8k_remote_nnsight/test_client.py --base-url http://localhost:8000
  python examples/gsm8k_remote_nnsight/test_client.py --base-url https://<pod>-8000.proxy.runpod.net \
      --ssh root@<ip>:<port>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

try:
    from bifrost.client import BifrostClient  # optional for --ssh
except Exception:
    BifrostClient = None  # type: ignore


def req(method: str, url: str, **kwargs) -> requests.Response:
    return requests.request(method, url, timeout=120, **kwargs)


def main() -> None:
    ap = argparse.ArgumentParser(description="NNsight server test client")
    ap.add_argument("--base-url", required=True, help="Server base URL, e.g., http://localhost:8000")
    ap.add_argument("--ssh", help="SSH conn string (user@host:port) to download activations")
    ap.add_argument("--layers", type=int, nargs="*", default=[8, 12, 16], help="Layer indices for capture")
    ap.add_argument("--hook-points", nargs="*", default=["input_layernorm.output", "post_attention_layernorm.output"], help="Hook points")
    ap.add_argument("--sample-hidden", type=int, default=64, help="Hidden dim slice size")
    ap.add_argument("--save-format", default="pt", choices=["pt", "npy"], help="Activation save format")
    ap.add_argument("--out", default=None, help="Local output dir for results and activations")
    args = ap.parse_args()

    base = args.base_url.rstrip("/")
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out or f"examples/gsm8k_remote_nnsight/test_results/{ts}")
    (out_dir / "activations").mkdir(parents=True, exist_ok=True)

    print(f"🔗 Testing server at {base}")

    # 1) Health
    r = req("GET", f"{base}/health")
    print(f"GET /health → {r.status_code}")
    print(r.text)
    r.raise_for_status()

    # 2) Models
    r = req("GET", f"{base}/v1/models")
    print(f"GET /v1/models → {r.status_code}")
    print(r.text)
    r.raise_for_status()
    models = r.json().get("data", [])
    if not models:
        print("❌ No models reported")
        sys.exit(2)
    model_id = models[0]["id"]
    print(f"Model: {model_id}")

    # 3) Configure interventions
    iv = {
        "enabled": True,
        "layers": args.layers,
        "hook_points": args.hook_points,
        "mode": "trace",
        "save_dir": "./activations",
        "per_request_subdir": True,
        "sample_hidden": args.sample_hidden,
        "save_format": args.save_format,
    }
    print("\n📝 Example curl to configure interventions:")
    print("curl -X POST", f"{base}/v1/interventions", "-H 'Content-Type: application/json'",
          "-d", json.dumps(iv))
    r = req("POST", f"{base}/v1/interventions", json=iv)
    print(f"POST /v1/interventions → {r.status_code}")
    print(r.text)
    r.raise_for_status()

    # 4) Chat completion
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": "You are a helpful math tutor."},
            {"role": "user", "content": "What is 12 + 35? Show your work and finish with Answer: [number]"},
        ],
        "max_tokens": 64,
        "temperature": 0.2,
    }
    print("\n📝 Example curl for chat:")
    print("curl -X POST", f"{base}/v1/chat/completions", "-H 'Content-Type: application/json'",
          "-d", json.dumps(payload))
    r = req("POST", f"{base}/v1/chat/completions", json=payload)
    print(f"POST /v1/chat/completions → {r.status_code}")
    r.raise_for_status()
    chat = r.json()
    print(json.dumps(chat, indent=2)[:2000])

    ndif = chat.get("ndif") or {}
    if not ndif:
        print("⚠️ No ndif breadcrumb in response (capture may be disabled)")
    else:
        req_id = ndif.get("request_id", "unknown")
        idx = ndif.get("index")
        print(f"📦 ndif: artifacts={len(ndif.get('artifacts', {}))}, request_id={req_id}")
        if idx:
            print(f"Index file: {idx}")

        # Optional download via SSH
        if idx and args.ssh and BifrostClient is not None:
            try:
                remote_dir = "~/.bifrost/workspace/" + str(Path(idx).parent.as_posix())
                local_dir = out_dir / "activations" / req_id
                print(f"⬇️  Downloading activations from {remote_dir} → {local_dir}")
                bc = BifrostClient(args.ssh)
                res = bc.download_files(remote_dir, str(local_dir), recursive=True)
                if res.success:
                    print(f"✅ Downloaded {res.files_copied} files ({res.total_bytes} bytes)")
                else:
                    print(f"❌ Download failed: {res.error_message}")
            except Exception as e:
                print(f"❌ SSH download error: {e}")
        elif idx and args.ssh and BifrostClient is None:
            print("⚠️ Bifrost not available; cannot download activations via SSH")

    # Save response for reference
    (out_dir / "responses").mkdir(parents=True, exist_ok=True)
    with open(out_dir / "responses" / "chat.json", "w") as f:
        json.dump(chat, f, indent=2)
    print(f"📝 Saved chat response to {out_dir/'responses'/'chat.json'}")

    print("\n🎉 Test complete")


if __name__ == "__main__":
    main()
