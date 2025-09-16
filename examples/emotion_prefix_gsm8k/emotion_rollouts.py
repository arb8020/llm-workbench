#!/usr/bin/env python3
"""Emotion prompt GSM8K runner using rollouts against the existing NNsight server.

Ad-hoc HTTP for health/interventions/structure; rollouts for chat completions.
Saves rollouts-style outputs and breadcrumbs for remote activations.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from broker.client import GPUClient
from broker import api as broker_api
from broker.types import InstanceStatus
from bifrost.client import BifrostClient

from datasets import load_dataset

from rollouts.dtypes import Message, Endpoint, RunConfig
from rollouts.evaluation import evaluate_sample, EvalSample
from rollouts.agents import stdout_handler

from examples.emotion_prefix_gsm8k.prompt_variants import PROMPT_VARIANTS


def _ensure_instance(name: str, port: int, min_vram: int, max_price: float):
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
    print("🔎 Provisioning new instance…")
    gc = GPUClient()
    query = (gc.vram_gb >= min_vram) & (gc.price_per_hour <= max_price)
    inst = gc.create(query=query, exposed_ports=[port], enable_http_proxy=True, name=name)
    if not inst or not inst.wait_until_ssh_ready(timeout=900):
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


def _structure(base_url: str) -> Dict[str, Any]:
    r = requests.get(base_url.rstrip("/") + "/v1/model/structure", timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"structure failed: {r.status_code} {r.text}")
    return r.json()


def _prepare_messages(sample: Dict[str, Any]) -> List[Message]:
    return [
        Message(role="system", content=(
            "You are an expert at solving math word problems. "
            "Follow these instructions:\n\n"
            "1. Read the problem carefully\n"
            "2. Think through the solution step by step\n"
            "3. Show your reasoning clearly\n"
            "4. Provide your final answer in this exact format: Answer: [number]\n\n"
            "Important: Your final line must be 'Answer: [your numeric answer]' with nothing else on that line."
        )),
        Message(role="user", content=f"Solve the following math problem step by step:\n\n{sample['question']}")
    ]


def _extract_answer(text: str) -> str:
    import re
    m = re.findall(r"Answer:\s*([^\n]+)", text, flags=re.IGNORECASE)
    ans = m[-1].strip() if m else ""
    m2 = re.search(r"-?\d+(?:\.\d+)?", ans)
    return m2.group() if m2 else ans


def _eq(a: str, b: str) -> bool:
    a = (a or "").replace(",", "").replace("$", "").strip()
    b = (b or "").replace(",", "").replace("$", "").strip()
    if a == b:
        return True
    try:
        return abs(float(a) - float(b)) < 1e-6
    except Exception:
        return False


def _correctness_reward(sample: Dict[str, Any]):
    def fn(traj):
        texts = [m.content or "" for m in traj.messages if m.role == "assistant"]
        pred = _extract_answer(" ".join(texts))
        gold = str(sample.get("answer", "")).strip()
        return 1.0 if (pred and _eq(pred, gold)) else 0.0
    return fn


def _format_reward(traj) -> float:
    import re
    texts = [m.content or "" for m in traj.messages if m.role == "assistant"]
    return 1.0 if re.search(r"Answer:\s*[^\n]+", " ".join(texts), re.IGNORECASE) else 0.0


def _efficiency_reward(traj) -> float:
    total = sum(len(m.content or "") for m in traj.messages)
    if total < 500: return 1.0
    if total > 2000: return 0.0
    return 1.0 - (total - 500) / 1500


def _select_gsm8k(nsamples: int, seed: int) -> List[Dict[str, Any]]:
    ds = load_dataset("gsm8k", "main")
    test = ds["test"]
    import random
    rng = random.Random(seed)
    idxs = list(range(len(test)))
    rng.shuffle(idxs)
    rows: List[Dict[str, Any]] = []
    for i in idxs[:nsamples]:
        row = test[i]
        ans = row["answer"]
        gold = ans.split("####")[-1].strip() if "####" in ans else ans
        rows.append({"question": row["question"], "answer": gold, "sample_id": f"gsm8k_{i+1:04d}"})
    return rows


def _save_rollouts_outputs(exp_dir: Path, variant: str, sample_id: str, res: EvalSample, breadcrumb: Dict[str, Any]) -> None:
    vdir = exp_dir / variant / sample_id
    vdir.mkdir(parents=True, exist_ok=True)
    # Trajectory
    with open(vdir / "trajectory.jsonl", "w") as f:
        for m in res.trajectory.messages:
            f.write(m.to_json() + "\n")
    # Agent states
    from dataclasses import asdict
    with open(vdir / "agent_state.json", "w") as f:
        json.dump([asdict(s) for s in res.agent_states], f, indent=2, default=str)
    # Sample summary
    with open(vdir / "sample.json", "w") as f:
        json.dump({
            "sample_id": sample_id,
            "variant": variant,
            "metrics": res.metrics,
            "metadata": res.metadata,
            "breadcrumb": breadcrumb,
        }, f, indent=2)


def main():
    ap = argparse.ArgumentParser(description="Emotion GSM8K via rollouts + NNsight server")
    ap.add_argument("--name", default="nnsight-gsm8k-demo")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--min-vram", type=int, default=12)
    ap.add_argument("--max-price", type=float, default=0.40)
    ap.add_argument("--model", default="willcb/Qwen3-0.6B")
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--variants", type=str, default="control,frustration,impatience,anxiety,collaborative,patience,calm")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--layers", type=str, default="all", help="'all' or comma-separated indices")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    exp_dir = Path(args.out or f"examples/emotion_prefix_gsm8k/results/emotion_rollouts_nsamples_{args.samples}_{time.strftime('%Y%m%d_%H%M%S')}")
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Ensure server
    inst = _ensure_instance(args.name, args.port, args.min_vram, args.max_price)
    ssh = inst.ssh_connection_string()
    bc = BifrostClient(ssh)
    workspace = bc.push(uv_extra="examples_gsm8k_remote_nnsight")
    bc.exec("which tmux || (apt-get update -y && apt-get install -y tmux) || true")
    bc.exec("tmux kill-session -t nnsight-server 2>/dev/null || true")
    run_cmd = (
        "cd ~/.bifrost/workspace && "
        f"bash examples/gsm8k_remote_nnsight/server/start_server.sh --host 0.0.0.0 --port {args.port} --model {args.model} --device-map {args.device_map}"
    )
    bc.exec(f"tmux new-session -d -s nnsight-server '{run_cmd}'")

    # Wait health
    base_urls = []
    proxy = inst.get_proxy_url(args.port)
    if proxy: base_urls.append(proxy)
    if inst.public_ip: base_urls.append(f"http://{inst.public_ip}:{args.port}")
    base = _wait_health(base_urls, timeout=900)
    if not base:
        raise SystemExit("Server did not become healthy in time")
    print(f"✅ Health OK: {base}")

    # Structure + interventions
    st = _structure(base)
    print(f"🔧 Model structure: num_layers={st.get('num_layers')} hook_points={st.get('hook_points')}")
    layers_payload: Any
    if args.layers.strip().lower() == 'all':
        layers_payload = 'all'
    else:
        layers_payload = [int(x.strip()) for x in args.layers.split(',') if x.strip()]
    iv = {
        "enabled": True,
        "layers": layers_payload,
        "hook_points": st.get("hook_points", []),
        "mode": "trace",
        "save_dir": "./activations",
        "per_request_subdir": True,
        "sample_hidden": None,
        "save_format": "pt",
    }
    r = requests.post(base + "/v1/interventions", json=iv, timeout=60)
    r.raise_for_status()
    print("✅ Interventions enabled")

    # Dataset
    rows = _select_gsm8k(args.samples, args.seed)
    variants = [v.strip() for v in args.variants.split(',') if v.strip()]

    # Rollouts config
    rcfg = RunConfig(on_chunk=stdout_handler)
    manifest: Dict[str, Any] = {}
    correct = 0
    total = 0

    for row in rows:
        for variant in variants:
            base_msgs = _prepare_messages(row)
            transform = PROMPT_VARIANTS.get(variant)
            msgs = transform(base_msgs) if transform else base_msgs
            # Per-request activation override; do not create subdir (we'll append request_id)
            save_dir = (
                Path("~/.bifrost/workspace").expanduser()
                / f"examples/emotion_prefix_gsm8k/results/{exp_dir.name}/activations/{variant}/{row['sample_id']}"
            )
            endpoint = Endpoint(
                provider="vllm",
                model=args.model,
                api_base=base.rstrip("/") + "/v1",
                api_key="dummy",
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                extra_params={"ndif": {"save_dir": str(save_dir), "per_request_subdir": False}},
            )
            # Rewards for this sample
            rewards = [
                ("correctness", _correctness_reward(row)),
                ("format", _format_reward),
                ("efficiency", _efficiency_reward),
            ]
            # Run
            import asyncio
            async def _go():
                return await evaluate_sample(row, row["sample_id"], lambda _s: msgs, rewards, type("E", (), {"get_tools": lambda self: []})(), endpoint, rcfg, max_turns=10, verbose=False)
            res: EvalSample = asyncio.run(_go())
            # Extract request id from completion id
            try:
                completion_id = res.trajectory.completions[-1].id
                req_id = completion_id.split("chatcmpl-")[-1]
            except Exception:
                req_id = "unknown"
            breadcrumb = {"save_dir": str(save_dir), "request_id": req_id}
            manifest[f"{variant}/{row['sample_id']}"] = breadcrumb
            _save_rollouts_outputs(exp_dir, variant, row["sample_id"], res, breadcrumb)
            ok = res.metrics.get("correctness", 0.0) > 0.5
            correct += 1 if ok else 0
            total += 1

    (exp_dir / "activations_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    acc = correct / max(1, total)
    (exp_dir / "report.json").write_text(json.dumps({"num_samples": total, "accuracy": acc}, indent=2), encoding="utf-8")
    print(f"🎉 Done. Accuracy={acc:.3f}")
    print(f"📁 Results: {exp_dir}")


if __name__ == "__main__":
    main()

