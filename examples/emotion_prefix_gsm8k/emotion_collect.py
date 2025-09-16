#!/usr/bin/env python3
"""Emotion prompt GSM8K runner using the existing NNsight server demo path.

Provision/reuse the NNsight server from examples/gsm8k_remote_nnsight,
configure capture, and evaluate GSM8K across emotional variants using
simple HTTP calls (no rollouts yet). Saves a YAML config for hygiene.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from broker.client import GPUClient
from broker import api as broker_api
from broker.types import InstanceStatus
from bifrost.client import BifrostClient

from datasets import load_dataset

from examples.emotion_prefix_gsm8k.prompt_variants import PROMPT_VARIANTS


def _ensure_instance(name: str, port: int, min_vram: int, max_price: float):
    # Reuse running instance by name if present
    try:
        insts = broker_api.list_instances(provider="runpod")
        for inst in insts:
            if inst.name == name and inst.status == InstanceStatus.RUNNING:
                print(f"✅ Using existing instance: {inst.id} ({inst.gpu_type})")
                if not inst.wait_until_ssh_ready(timeout=900):
                    raise RuntimeError("SSH not ready on existing instance")
                return inst
    except Exception:
        pass
    # Otherwise create
    print("🔎 Provisioning new instance…")
    gc = GPUClient()
    query = (gc.vram_gb >= min_vram) & (gc.price_per_hour <= max_price)
    inst = gc.create(query=query, exposed_ports=[port], enable_http_proxy=True, name=name)
    if not inst:
        raise RuntimeError("Failed to provision instance")
    if not inst.wait_until_ssh_ready(timeout=900):
        raise RuntimeError("SSH not ready in time")
    return inst


def _wait_health(urls: List[str], timeout: int = 900) -> Optional[str]:
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


def _prepare_messages(question: str) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are an expert at solving math word problems. "
                "Think step by step and finish with: Answer: [number]"
            ),
        },
        {"role": "user", "content": f"Solve the following math problem step by step:\n\n{question}"},
    ]


def _apply_variant(messages: List[Dict[str, str]], variant: str) -> List[Dict[str, str]]:
    # Lightweight variant applier mirroring prompt_variants on dict messages
    from rollouts.dtypes import Message
    # Convert to Message to reuse transforms without rollouts runtime
    msg_objs = [Message(role=m["role"], content=m["content"]) for m in messages]
    transform = PROMPT_VARIANTS.get(variant)
    if transform:
        msg_objs = transform(msg_objs)
    return [{"role": m.role, "content": m.content or ""} for m in msg_objs]


def main():
    ap = argparse.ArgumentParser(description="Emotion GSM8K via NNsight server (non-rollouts)")
    ap.add_argument("--name", default="nnsight-gsm8k-demo")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--min-vram", type=int, default=12)
    ap.add_argument("--max-price", type=float, default=0.40)
    ap.add_argument("--model", default="willcb/Qwen3-0.6B")
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--variants", type=str, default="control,frustration,impatience,anxiety,collaborative,patience,calm")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--layers", type=int, nargs="*", default=[8, 12, 16], help="Layer indices (server doesn't support 'all' here)")
    args = ap.parse_args()

    out_root = Path(args.out or f"examples/emotion_prefix_gsm8k/results/emotion_collect_nsamples_{args.samples}_{time.strftime('%Y%m%d_%H%M%S')}")
    (out_root / "trajectories").mkdir(parents=True, exist_ok=True)

    # 1) Ensure instance
    inst = _ensure_instance(args.name, args.port, args.min_vram, args.max_price)
    ssh = inst.ssh_connection_string()
    bc = BifrostClient(ssh)

    # 2) Push code and start server in tmux (reuse gsm8k_remote_nnsight launcher)
    workspace = bc.push(uv_extra="examples_gsm8k_remote_nnsight")
    bc.exec("which tmux || (apt-get update -y && apt-get install -y tmux) || true")
    bc.exec("tmux kill-session -t nnsight-server 2>/dev/null || true")
    run_cmd = (
        "cd ~/.bifrost/workspace && "
        f"bash examples/gsm8k_remote_nnsight/server/start_server.sh --host 0.0.0.0 --port {args.port} --model {args.model} --device-map {args.device_map}"
    )
    bc.exec(f"tmux new-session -d -s nnsight-server '{run_cmd}'")

    # 3) Wait health
    base_urls = []
    proxy = inst.get_proxy_url(args.port)
    if proxy: base_urls.append(proxy)
    if inst.public_ip: base_urls.append(f"http://{inst.public_ip}:{args.port}")
    ready = _wait_health(base_urls, timeout=900)
    if not ready:
        print("❌ Server did not become healthy in time", file=sys.stderr)
        sys.exit(2)
    print(f"✅ Health OK: {ready}")
    # Write experiment YAML for hygiene
    try:
        import yaml  # type: ignore
        exp = {
            "run": {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "name": args.name,
            },
            "server": {
                "instance_id": inst.id,
                "ssh": ssh,
                "base_url": ready,
                "port": args.port,
                "model": args.model,
            },
            "dataset": {
                "name": "gsm8k",
                "split": "test",
                "nsamples": args.samples,
                "seed": args.seed,
            },
            "generation": {
                "max_tokens": 256,
                "temperature": 0.7,
            },
            "variants": [v.strip() for v in args.variants.split(',') if v.strip()],
            "interventions": {
                "layers": args.layers,
                "hook_points": ["input_layernorm.output", "post_attention_layernorm.output"],
                "save_dir_base": "./activations",
                "per_request_subdir": True,
            },
        }
        (out_root / "experiment.yaml").write_text(yaml.safe_dump(exp, sort_keys=False), encoding="utf-8")
    except Exception:
        pass

    # 4) Configure capture (subset of layers; two hook points)
    iv = {
        "enabled": True,
        "layers": args.layers,
        "hook_points": ["input_layernorm.output", "post_attention_layernorm.output"],
        "mode": "trace",
        "save_dir": "./activations",
        "per_request_subdir": True,
        "sample_hidden": None,
        "save_format": "pt",
    }
    r = requests.post(ready + "/v1/interventions", json=iv, timeout=60)
    if r.status_code != 200 or not r.json().get("ok"):
        print("❌ Failed to configure interventions:", r.text)
        sys.exit(3)
    print("✅ Interventions configured")

    # 5) Load dataset
    ds = load_dataset("gsm8k", "main")
    test = ds["test"]
    import random
    rng = random.Random(args.seed)
    idxs = list(range(len(test)))
    rng.shuffle(idxs)
    rows = [test[i] for i in idxs[:args.samples]]

    variants = [v.strip() for v in args.variants.split(',') if v.strip()]
    manifest: Dict[str, Any] = {}
    correct = 0
    total = 0

    for vi, variant in enumerate(variants, start=1):
        for i, row in enumerate(rows, start=1):
            sample_id = row.get("id") or f"gsm8k_{idxs[i-1]+1:04d}"
            print(f"➡️  {variant} | {i}/{len(rows)}: {sample_id}")
            msgs = _prepare_messages(row["question"])
            msgs = _apply_variant(msgs, variant)
            payload = {
                "model": args.model,
                "messages": msgs,
                "max_tokens": 256,
                "temperature": 0.7,
            }
            resp = requests.post(ready + "/v1/chat/completions", json=payload, timeout=180)
            if resp.status_code != 200:
                print("❌ Chat error:", resp.text)
                continue
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            ndif = data.get("ndif") or {}

            # Save simple trajectory
            tdir = out_root / variant / sample_id
            tdir.mkdir(parents=True, exist_ok=True)
            with open(tdir / "trajectory.jsonl", "w") as f:
                for m in msgs + [{"role": "assistant", "content": text}]:
                    f.write(json.dumps(m) + "\n")
            # Save sample.json with breadcrumb only (no downloads)
            gold = row.get("answer", "").strip()
            import re
            m = re.findall(r"Answer:\s*([^\n]+)", text, flags=re.IGNORECASE)
            pred_raw = m[-1].strip() if m else ""
            m2 = re.search(r"-?\d+(?:\.\d+)?", pred_raw)
            pred = m2.group() if m2 else pred_raw
            ok = 1 if (pred.replace(",", "").replace("$", "").strip() == gold.replace(",", "").replace("$", "").strip()) else 0
            total += 1
            correct += ok
            with open(tdir / "sample.json", "w") as f:
                json.dump({
                    "sample_id": sample_id,
                    "variant": variant,
                    "question": row.get("question"),
                    "gold_answer": gold,
                    "model_answer": text,
                    "extracted_answer": pred,
                    "correct": bool(ok),
                    "ndif": ndif,
                }, f, indent=2)
            # Breadcrumb manifest
            manifest[f"{variant}/{sample_id}"] = ndif

    (out_root / "activations_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    acc = correct / max(1, total)
    (out_root / "report.json").write_text(json.dumps({"num_samples": total, "accuracy": acc}, indent=2), encoding="utf-8")
    print(f"🎉 Done. Accuracy={acc:.3f}")
    print(f"📁 Results: {out_root}")


if __name__ == "__main__":
    main()
