#!/usr/bin/env python3
from __future__ import annotations

import json
import time
import sys
from pathlib import Path
from typing import Any, Dict, List

import requests
from datasets import load_dataset

from rollouts.dtypes import Message, Endpoint, RunConfig
from rollouts.evaluation import evaluate_sample
from rollouts.agents import stdout_handler

from examples.emotion_prefix_gsm8k.prompt_variants import PROMPT_VARIANTS


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


def main():
    if len(sys.argv) != 2:
        print("Usage: python -m examples.emotion_prefix_gsm8k.remote_runner_rollouts <config_path>")
        sys.exit(2)
    cfg_path = Path(sys.argv[1])
    cfg = json.loads(cfg_path.read_text())

    base = cfg["base_url"].rstrip("/")
    model = cfg["model"]
    samples = int(cfg["samples"]) 
    seed = int(cfg["seed"]) 
    variants = list(cfg["variants"]) 
    max_tokens = int(cfg["max_tokens"]) 
    temperature = float(cfg["temperature"]) 
    layers = cfg["layers"]
    exp_name = cfg["exp_name"]

    # Ensure structure and configure interventions
    r = requests.get(base + "/v1/model/structure", timeout=30)
    r.raise_for_status()
    struct = r.json()
    iv = {
        "enabled": True,
        "layers": layers,
        "hook_points": struct.get("hook_points", []),
        "mode": "trace",
        "save_dir": "./activations",
        "per_request_subdir": True,
        "sample_hidden": None,
        "save_format": "pt",
    }
    r = requests.post(base + "/v1/interventions", json=iv, timeout=60)
    r.raise_for_status()
    print("[runner] interventions enabled", flush=True)

    # Dataset selection
    ds = load_dataset("gsm8k", "main")
    test = ds["test"]
    import random
    rng = random.Random(seed)
    idxs = list(range(len(test)))
    rng.shuffle(idxs)
    rows: List[Dict[str, Any]] = []
    for i in idxs[:samples]:
        row = test[i]
        ans = row["answer"]
        gold = ans.split("####")[-1].strip() if "####" in ans else ans
        rows.append({"question": row["question"], "answer": gold, "sample_id": f"gsm8k_{i+1:04d}"})

    # Output directories (remote)
    out_root = Path(f"examples/emotion_prefix_gsm8k/results/{exp_name}")
    out_root.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {}
    rcfg = RunConfig(on_chunk=stdout_handler)
    correct = 0
    total = 0

    for row in rows:
        for variant in variants:
            base_msgs = _prepare_messages(row)
            transform = PROMPT_VARIANTS.get(variant)
            msgs = transform(base_msgs) if transform else base_msgs
            # Per-request save dir override
            save_dir = Path.home() / ".bifrost/workspace" / f"examples/emotion_prefix_gsm8k/results/{exp_name}/activations/{variant}/{row['sample_id']}"
            endpoint = Endpoint(
                provider="vllm",
                model=model,
                api_base=base + "/v1",
                api_key="dummy",
                max_tokens=max_tokens,
                temperature=temperature,
                extra_params={"ndif": {"save_dir": str(save_dir), "per_request_subdir": False}},
            )
            rewards = [("correctness", _correctness_reward(row)), ("format", _format_reward), ("efficiency", _efficiency_reward)]
            import asyncio
            async def _go():
                return await evaluate_sample(row, row["sample_id"], lambda _s: msgs, rewards, type("E", (), {"get_tools": lambda self: []})(), endpoint, rcfg, max_turns=10, verbose=False)
            res = asyncio.run(_go())
            try:
                completion_id = res.trajectory.completions[-1].id
                req_id = completion_id.split("chatcmpl-")[-1]
            except Exception:
                req_id = "unknown"
            breadcrumb = {"save_dir": str(save_dir), "request_id": req_id}
            manifest[f"{variant}/{row['sample_id']}"] = breadcrumb
            # Save rollouts-style outputs
            vdir = out_root / variant / row["sample_id"]
            vdir.mkdir(parents=True, exist_ok=True)
            with open(vdir / "trajectory.jsonl", "w") as f:
                for m in res.trajectory.messages:
                    f.write(m.to_json() + "\n")
            from dataclasses import asdict
            with open(vdir / "agent_state.json", "w") as f:
                json.dump([asdict(s) for s in res.agent_states], f, indent=2, default=str)
            with open(vdir / "sample.json", "w") as f:
                json.dump({"sample_id": row["sample_id"], "variant": variant, "metrics": res.metrics, "metadata": res.metadata, "breadcrumb": breadcrumb}, f, indent=2)
            ok = res.metrics.get("correctness", 0.0) > 0.5
            correct += 1 if ok else 0
            total += 1
            print(f"[runner] {variant}/{row['sample_id']} ok={ok}", flush=True)

    (out_root / "activations_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    acc = correct / max(1, total)
    (out_root / "report.json").write_text(json.dumps({"num_samples": total, "accuracy": acc}, indent=2), encoding="utf-8")
    print(f"[runner] done accuracy={acc:.3f} results={out_root}", flush=True)


if __name__ == "__main__":
    main()

