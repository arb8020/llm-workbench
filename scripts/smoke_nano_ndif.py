#!/usr/bin/env python3
"""Integration-style smoke test for nano-ndif server.

Usage:
  python scripts/smoke_nano_ndif.py --base-url http://localhost:8000
  python scripts/smoke_nano_ndif.py --base-url https://<pod>-8000.proxy.runpod.net
"""

from __future__ import annotations

import sys
import time
import json
import argparse
from typing import Any, Dict

import requests


def req(method: str, url: str, **kwargs) -> requests.Response:
    r = requests.request(method, url, timeout=120, **kwargs)
    return r


def expect_status(resp: requests.Response, code: int) -> None:
    if resp.status_code != code:
        print(f"❌ {resp.request.method} {resp.request.url} => {resp.status_code}\n{resp.text}")
        sys.exit(2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000", help="Server base URL")
    args = ap.parse_args()

    base = args.base_url.rstrip("/")
    print(f"🔗 Testing nano-ndif at {base}")

    # 1) Health
    r = req("GET", f"{base}/health")
    expect_status(r, 200)
    health = r.json()
    assert isinstance(health.get("ok"), bool), "health.ok missing"
    print("✅ health")

    # 2) Models
    r = req("GET", f"{base}/v1/models")
    expect_status(r, 200)
    models = r.json()
    assert models.get("data"), "models list empty"
    model_id = models["data"][0]["id"]
    print(f"✅ models → {model_id}")

    # 3) Baseline chat
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": "What is 2+2?"}],
        "max_tokens": 16,
        "temperature": 0.1,
    }
    r = req("POST", f"{base}/v1/chat/completions", json=payload)
    expect_status(r, 200)
    chat1 = r.json()
    ctext = chat1["choices"][0]["message"]["content"]
    print(f"✅ chat (baseline) → {ctext[:60]!r}")

    # 4) Configure interventions
    layers = [8, 12, 16]
    hook_points = ["input_layernorm.output", "post_attention_layernorm.output"]
    sample_hidden = 64
    iv = {
        "enabled": True,
        "layers": layers,
        "hook_points": hook_points,
        "mode": "trace",
        "save_dir": "./activations",
        "per_request_subdir": True,
        "sample_hidden": sample_hidden,
        "save_format": "pt",
    }
    r = req("POST", f"{base}/v1/interventions", json=iv)
    expect_status(r, 200)
    ivr = r.json()
    assert ivr.get("ok") is True, "interventions not acknowledged"
    print("✅ interventions configured")

    # 5) Chat with capture enabled
    payload2 = {
        "model": model_id,
        "messages": [{"role": "user", "content": "List 3 prime numbers."}],
        # Ensure multiple token generation
        "max_tokens": 24,
        "temperature": 0.1,
    }
    r = req("POST", f"{base}/v1/chat/completions", json=payload2)
    expect_status(r, 200)
    chat2 = r.json()
    ndif = chat2.get("ndif")
    assert ndif and isinstance(ndif.get("artifacts"), dict) and ndif["artifacts"], "ndif artifacts missing"
    # Verify completion generated more than one token
    usage = chat2.get("usage", {})
    assert usage and usage.get("completion_tokens", 0) >= 2, "expected multiple generated tokens"
    # Verify artifact count and shapes
    expected_count = len(layers) * len(hook_points)
    assert len(ndif["artifacts"]) == expected_count, f"expected {expected_count} artifacts"
    for key, meta in ndif["artifacts"].items():
        shape = meta.get("shape")
        saved_shape = meta.get("saved_shape")
        dtype = meta.get("dtype")
        assert isinstance(shape, list) and len(shape) == 3, f"bad shape for {key}: {shape}"
        assert isinstance(saved_shape, list) and len(saved_shape) == 3, f"bad saved_shape for {key}: {saved_shape}"
        assert saved_shape[0] == shape[0] and saved_shape[1] == shape[1], "batch/seq mismatch"
        assert saved_shape[2] == min(shape[2], sample_hidden), "hidden slice mismatch"
        assert isinstance(dtype, str) and dtype, "missing dtype"
    print(f"✅ chat (captured) → {len(ndif['artifacts'])} tensors with correct shapes; request_id={ndif.get('request_id')}")

    print("🎉 All smoke tests passed")


if __name__ == "__main__":
    main()
