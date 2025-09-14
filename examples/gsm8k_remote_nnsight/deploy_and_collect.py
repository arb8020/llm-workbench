#!/usr/bin/env python3
"""GSM8K remote NNsight demo: deploy server, run samples, collect activations + trajectories.

Usage:
  python examples/gsm8k_remote_nnsight/deploy_and_collect.py --samples 3
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from broker.client import GPUClient
from bifrost.client import BifrostClient


def _prepare_messages(sample: Dict[str, Any]) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are an expert at solving math word problems. "
                "Think step by step and finish with: Answer: [number]"
            ),
        },
        {
            "role": "user",
            "content": f"Solve the following math problem step by step:\n\n{sample['question']}",
        },
    ]


def _extract_answer(text: str) -> str:
    import re
    m = re.findall(r"Answer:\s*([^\n]+)", text, flags=re.IGNORECASE)
    if not m:
        return ""
    ans = m[-1].strip().replace("$", "").replace(",", "")
    m2 = re.search(r"-?\d+(?:\.\d+)?", ans)
    return m2.group() if m2 else ans


def _eq(a: str, b: str) -> bool:
    a = (a or "").strip().replace(",", "").replace("$", "")
    b = (b or "").strip().replace(",", "").replace("$", "")
    if a == b:
        return True
    try:
        return abs(float(a) - float(b)) < 1e-6
    except Exception:
        return False


def _wait_health(urls: List[str], timeout: int = 600) -> Optional[str]:
    start = time.time()
    while time.time() - start < timeout:
        for base in urls:
            try:
                r = requests.get(base.rstrip("/") + "/health", timeout=5)
                if r.status_code == 200 and r.json().get("ok"):
                    return base.rstrip("/")
            except Exception:
                pass
        time.sleep(3)
    return None


def main(samples: int = 3, layers: List[int] | None = None, out_dir: Optional[str] = None):
    layers = layers or [8, 12, 16]
    port = 8000
    model_id = "willcb/Qwen3-0.6B"

    # 1) Provision GPU
    print("🚀 Provisioning GPU instance (>=12GB VRAM, <=$0.40/hr)…")
    gc = GPUClient()
    query = (gc.vram_gb >= 12) & (gc.price_per_hour <= 0.40)
    inst = gc.create(query=query, exposed_ports=[port], enable_http_proxy=True, name="nnsight-gsm8k-demo")
    assert inst and inst.id
    print(f"✅ Instance: {inst.id}")
    if not inst.wait_until_ssh_ready(timeout=600):
        print("❌ SSH not ready in time")
        sys.exit(2)
    ssh = inst.ssh_connection_string()
    print(f"🔑 SSH: {ssh}")

    # 2) Deploy code with extras
    print("📦 Deploying code with examples_gsm8k_remote_nnsight extras…")
    bc = BifrostClient(ssh)
    workspace = bc.push(uv_extra="examples_gsm8k_remote_nnsight")
    print(f"✅ Workspace ready: {workspace}")

    # 3) Start server in tmux
    print("🌟 Starting NNsight server in tmux session 'nnsight-server'…")
    cmd = (
        f"bash examples/gsm8k_remote_nnsight/server/start_server.sh "
        f"--host 0.0.0.0 --port {port} --model {model_id} --device-map auto"
    )
    bc.exec(f"tmux kill-session -t nnsight-server 2>/dev/null || true")
    bc.exec("which tmux || (apt-get update -y && apt-get install -y tmux) || true")
    bc.exec(f"tmux new-session -d -s nnsight-server 'cd ~/.bifrost/workspace && {cmd} 2>&1 | tee ~/nnsight_server.log'")
    print("✅ Server starting; waiting for health…")

    # 4) Wait for health
    base_urls = []
    proxy = inst.get_proxy_url(port)
    if proxy:
        base_urls.append(proxy)
    if inst.public_ip:
        base_urls.append(f"http://{inst.public_ip}:{port}")
    ready = _wait_health(base_urls, timeout=900)
    if not ready:
        print("❌ Server did not become ready in time", file=sys.stderr)
        sys.exit(3)
    print(f"✅ Health OK: {ready}")

    # 5) Configure interventions
    iv = {
        "enabled": True,
        "layers": layers,
        "hook_points": [
            "input_layernorm.output",
            "post_attention_layernorm.output",
        ],
        "mode": "trace",
        "save_dir": "./activations",
        "per_request_subdir": True,
        "sample_hidden": 64,
        "save_format": "pt",
    }
    r = requests.post(ready + "/v1/interventions", json=iv, timeout=60)
    if r.status_code != 200 or not r.json().get("ok"):
        print("❌ Failed to configure interventions:", r.text)
        sys.exit(4)
    print("✅ Interventions configured")

    # 6) Load GSM8K samples
    print(f"📚 Loading {samples} GSM8K samples…")
    try:
        from datasets import load_dataset
        ds = load_dataset("gsm8k", "main")
        test = ds["test"]
        rows = [test[i] for i in range(min(samples, len(test)))]
    except Exception:
        print("⚠️ datasets unavailable; using tiny fallback prompts")
        rows = [
            {"question": "What is 2 + 3?", "answer": "5"},
            {"question": "A bag has 4 apples and 3 oranges. How many fruits?", "answer": "7"},
        ][:samples]

    # 7) Run samples and sync activations
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    exp_name = f"gsm8k_remote_nnsight_nsamples_{len(rows)}_{timestamp}"
    out_root = Path(out_dir or f"examples/gsm8k_remote_nnsight/results/{exp_name}")
    (out_root / "samples").mkdir(parents=True, exist_ok=True)
    (out_root / "trajectories").mkdir(parents=True, exist_ok=True)
    (out_root / "activations").mkdir(parents=True, exist_ok=True)

    correct = 0
    results: List[Dict[str, Any]] = []

    for i, row in enumerate(rows, start=1):
        sample_id = row.get("id") or f"gsm8k_{i:04d}"
        print(f"➡️  Sample {i}/{len(rows)}: {sample_id}")
        msgs = _prepare_messages(row)
        payload = {
            "model": model_id,
            "messages": msgs,
            "max_tokens": 128,
            "temperature": 0.2,
        }
        resp = requests.post(ready + "/v1/chat/completions", json=payload, timeout=180)
        if resp.status_code != 200:
            print("❌ Chat error:", resp.text)
            continue
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        ndif = data.get("ndif") or {}

        # Build simple trajectory file
        traj_path = out_root / "trajectories" / f"{sample_id}.jsonl"
        with open(traj_path, "w") as f:
            for m in msgs + [{"role": "assistant", "content": text}]:
                f.write(json.dumps(m) + "\n")

        # Save sample result
        pred = _extract_answer(text)
        gold = row.get("answer", "")
        ok = _eq(pred, gold)
        correct += 1 if ok else 0
        sample_out = {
            "sample_id": sample_id,
            "question": row.get("question"),
            "gold_answer": gold,
            "model_answer": text,
            "extracted_answer": pred,
            "correct": ok,
            "ndif": ndif,
        }
        with open(out_root / "samples" / f"{sample_id}.json", "w") as f:
            json.dump(sample_out, f, indent=2)

        # Sync activation folder for this request (if present)
        index_path = ndif.get("index")
        if index_path:
            # Server runs under ~/.bifrost/workspace
            # Index like: activations/run-.../<req_id>/metadata.json
            remote_dir = f"~/.bifrost/workspace/{Path(index_path).parent.as_posix()}"
            local_dir = out_root / "activations" / sample_id
            print(f"   ⬇️  Downloading activations → {local_dir}")
            bc.download_files(remote_dir, str(local_dir), recursive=True)
        else:
            print("   ⚠️ No ndif index in response; skipping activation download")

        results.append(sample_out)

    # 8) Write simple report
    acc = correct / max(1, len(rows))
    report = {
        "eval_name": exp_name,
        "num_samples": len(rows),
        "accuracy": acc,
    }
    with open(out_root / "report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n🎉 Done.")
    print(f"📁 Local results: {out_root}")
    print("🔎 Remote logs: ~/.bifrost/workspace → ~/nnsight_server.log")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Deploy NNsight server and collect activations on GSM8K")
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--layers", type=int, nargs="*", default=None)
    ap.add_argument("--out", type=str, default=None, help="Output dir (default under examples/gsm8k_remote_nnsight/results)")
    args = ap.parse_args()
    main(samples=args.samples, layers=args.layers, out_dir=args.out)
