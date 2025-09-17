#!/usr/bin/env python3
"""Emotion Prefix GSM8K: deploy NNsight server, run rollouts with per-request activation capture, and save results.

Stages (resumable): deploy, configure, generate, analyze, download-activations, monitor.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import yaml
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from broker.client import GPUClient
from broker import api as broker_api
from broker.types import InstanceStatus
from bifrost.client import BifrostClient

from rollouts.dtypes import Message, Endpoint, RunConfig
from rollouts.evaluation import evaluate_sample, EvalSample
from rollouts.agents import stdout_handler

from datasets import load_dataset

from examples.emotion_prefix_gsm8k.prompt_variants import PROMPT_VARIANTS


def _prepare_messages(sample: Dict[str, Any]) -> List[Message]:
    return [
        Message(
            role="system",
            content=(
                "You are an expert at solving math word problems. "
                "Follow these instructions:\n\n"
                "1. Read the problem carefully\n"
                "2. Think through the solution step by step\n"
                "3. Show your reasoning clearly\n"
                "4. Provide your final answer in this exact format: Answer: [number]\n\n"
                "Important: Your final line must be 'Answer: [your numeric answer]' with nothing else on that line."
            ),
        ),
        Message(role="user", content=f"Solve the following math problem step by step:\n\n{sample['question']}")
    ]


def _extract_answer(response_text: str) -> str:
    import re
    m = re.findall(r"Answer:\s*([^\n]+)", response_text, flags=re.IGNORECASE)
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
        try:
            from fractions import Fraction
            return Fraction(a) == Fraction(b)
        except Exception:
            return False


def _correctness_reward(sample: Dict[str, Any]):
    def check(traj):
        asst = [m for m in traj.messages if m.role == "assistant"]
        if not asst:
            return 0.0
        text = " ".join(m.content or "" for m in asst)
        pred = _extract_answer(text)
        gold = str(sample.get("answer", "")).strip()
        return 1.0 if pred and _eq(pred, gold) else 0.0
    return check


def _format_reward(traj) -> float:
    import re
    asst = [m for m in traj.messages if m.role == "assistant"]
    if not asst:
        return 0.0
    text = " ".join(m.content or "" for m in asst)
    return 1.0 if re.search(r"Answer:\s*[^\n]+", text, re.IGNORECASE) else 0.0


def _efficiency_reward(traj) -> float:
    total = sum(len(m.content or "") for m in traj.messages)
    if total < 500:
        return 1.0
    if total > 2000:
        return 0.0
    return 1.0 - (total - 500) / 1500


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


def _variants_list(s: str) -> List[str]:
    return [v.strip() for v in s.split(',') if v.strip()]


def _normalize_config_defaults(data: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare config defaults so argparse can consume them safely."""
    out: Dict[str, Any] = {}
    for key, value in data.items():
        # Allow variants as list or comma-separated string
        if key == "variants" and isinstance(value, list):
            out[key] = ",".join(str(v).strip() for v in value)
            continue
        # Allow stages as comma or list
        if key == "stages" and isinstance(value, str):
            out[key] = [p for p in value.replace(',', ' ').split() if p]
            continue
        out[key] = value
    return out


def _ensure_instance(name: str, port: int, min_vram: int, max_price: float, gpu_type: Optional[str], manufacturer: Optional[str]):
    # Try reuse running instance with same name (runpod)
    try:
        insts = broker_api.list_instances(provider="runpod")
        for inst in insts:
            if inst.name == name and inst.status == InstanceStatus.RUNNING:
                print(f"✅ Reusing existing instance: {inst.id} ({inst.gpu_type})")
                if not inst.wait_until_ssh_ready(timeout=900):
                    raise RuntimeError("Existing instance SSH not ready")
                return inst
    except Exception:
        pass
    gc = GPUClient()
    query = (gc.vram_gb >= min_vram) & (gc.price_per_hour <= max_price)
    if gpu_type:
        query = query & (gc.gpu_type == gpu_type)
    if manufacturer:
        query = query & (gc.manufacturer == manufacturer)
    inst = gc.create(query=query, exposed_ports=[port], enable_http_proxy=True, name=name)
    if not inst or not inst.wait_until_ssh_ready(timeout=900):
        raise RuntimeError("GPU SSH not ready")
    return inst


def _save_yaml(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _exp_dir(name: str) -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    return Path(f"examples/emotion_prefix_gsm8k/results/{name}_{ts}")


def stage_deploy(args, exp_dir: Path) -> Dict[str, Any]:
    print("🚀 Provisioning GPU instance…")
    inst = _ensure_instance(args.name, args.port, args.min_vram, args.max_price, args.gpu_type, args.manufacturer)
    ssh = inst.ssh_connection_string()
    bc = BifrostClient(ssh)
    print("📦 Pushing code (uv extra: examples_emotion_prefix_gsm8k)…")
    workspace = bc.push(uv_extra="examples_emotion_prefix_gsm8k")
    print(f"✅ Workspace: {workspace}")
    bc.exec("which tmux || (apt-get update -y && apt-get install -y tmux) || true")
    bc.exec("tmux kill-session -t nnsight-server 2>/dev/null || true")
    run_cmd = (
        "cd ~/.bifrost/workspace && "
        f"bash examples/emotion_prefix_gsm8k/server/start_server.sh --host 0.0.0.0 --port {args.port} --model {args.model} --device-map {args.device_map}"
    )
    bc.exec(f"tmux new-session -d -s nnsight-server '{run_cmd}'")
    print("✅ Launched 'nnsight-server' in tmux")
    base_urls = []
    proxy = inst.get_proxy_url(args.port)
    if proxy:
        base_urls.append(proxy)
    if inst.public_ip:
        base_urls.append(f"http://{inst.public_ip}:{args.port}")
    ready = None
    if not getattr(args, "no_wait_health", False):
        # Wait up to 1800s, printing periodic log tails
        deadline = time.time() + 1800
        last_log = 0
        while time.time() < deadline and not ready:
            ready = _wait_health(base_urls, timeout=15)
            now = time.time()
            if now - last_log > 60:
                last_log = now
                try:
                    out = bc.exec("tail -n 100 ~/nnsight_server.log 2>/dev/null || true")
                    print("--- server log tail ---\n" + "\n".join(out.splitlines()[-20:]))
                except Exception:
                    pass
        if not ready:
            # Final tail
            try:
                out = bc.exec("tail -n 200 ~/nnsight_server.log 2>/dev/null || true")
                print("--- final server log ---\n" + out)
            except Exception:
                pass
            print("⚠️  Proceeding without health ready due to --no-wait-health")
    print(f"✅ Health OK: {ready}")
    return {
        "instance_id": inst.id,
        "ssh": ssh,
        "ready_url": ready or (proxy or (f"http://{inst.public_ip}:{args.port}" if inst.public_ip else "")),
        "proxy_url": proxy,
        "direct_url": f"http://{inst.public_ip}:{args.port}" if inst.public_ip else None,
    }


def stage_configure(args, exp_dir: Path, deploy_info: Dict[str, Any]) -> Dict[str, Any]:
    base = deploy_info["ready_url"]
    print("🔎 Querying model structure…")
    structure = _structure(base)
    print(f"   model={structure.get('model')} num_layers={structure.get('num_layers')} hook_points={structure.get('hook_points')}")
    # Enable capture globally (layers=all); we will override per-request save_dir
    iv = {
        "enabled": True,
        "layers": "all",
        "hook_points": structure.get("hook_points", []),
        "mode": "trace",
        "save_dir": "./activations",
        "per_request_subdir": True,
        "sample_hidden": None,
        "save_format": "pt",
    }
    r = requests.post(base + "/v1/interventions", json=iv, timeout=60)
    if r.status_code != 200 or not r.json().get("ok"):
        raise RuntimeError(f"Failed to configure interventions: {r.text}")
    print("✅ Interventions enabled (layers=all)")
    return {"structure": structure}


def _select_gsm8k(nsamples: int, seed: int) -> List[Dict[str, Any]]:
    ds = load_dataset("gsm8k", "main")
    test = ds["test"]
    # deterministic selection
    import random
    rng = random.Random(seed)
    idxs = list(range(len(test)))
    rng.shuffle(idxs)
    rows: List[Dict[str, Any]] = []
    for i in idxs[:nsamples]:
        row = test[i]
        answer_text = row["answer"]
        gold = answer_text.split("####")[-1].strip() if "####" in answer_text else answer_text
        rows.append({"question": row["question"], "answer": gold, "sample_id": f"gsm8k_{i+1:04d}"})
    return rows


def _make_endpoint(base_url: str, model: str, max_tokens: int, temperature: float, ndif: Dict[str, Any]) -> Endpoint:
    # Endpoint now accepts extra_params (added to request body)
    return Endpoint(
        provider="vllm",
        model=model,
        api_base=base_url.rstrip("/") + "/v1",
        api_key="dummy",
        max_tokens=max_tokens,
        temperature=temperature,
        extra_params={"ndif": ndif},
    )


def _save_eval_outputs(exp_dir: Path, variant: str, sample_id: str, result: EvalSample, breadcrumb: Dict[str, Any]) -> None:
    vdir = exp_dir / variant / sample_id
    vdir.mkdir(parents=True, exist_ok=True)
    traj_path = vdir / "trajectory.jsonl"
    with open(traj_path, "w") as f:
        for m in result.trajectory.messages:
            f.write(m.to_json() + "\n")
    agent_state_path = vdir / "agent_state.json"
    with open(agent_state_path, "w") as f:
        if result.agent_states:
            from dataclasses import asdict
            json.dump([asdict(s) for s in result.agent_states], f, indent=2, default=str)
        else:
            f.write("[]")
    sample_path = vdir / "sample.json"
    meta = {
        "sample_id": sample_id,
        "variant": variant,
        "metrics": result.metrics,
        "breadcrumb": breadcrumb,
        "metadata": result.metadata,
    }
    sample_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def stage_generate(args, exp_dir: Path, deploy_info: Dict[str, Any], structure: Dict[str, Any], variants: List[str]) -> Dict[str, Any]:
    base = deploy_info["ready_url"]
    model = args.model
    rows = _select_gsm8k(args.samples, args.seed)
    print(f"📚 Selected {len(rows)} GSM8K samples")
    report: Dict[str, Any] = {"correct": 0, "total": 0}
    manifest: Dict[str, Any] = {}
    # Rewards
    rewards = [("correctness", _correctness_reward), ("format", _format_reward), ("efficiency", _efficiency_reward)]
    rcfg = RunConfig(on_chunk=stdout_handler, inline_thinking=None, user_message_for_thinking=None)

    for row in rows:
        for variant in variants:
            # Prepare base messages + apply transform
            base_msgs = _prepare_messages(row)
            transform = PROMPT_VARIANTS.get(variant)
            msgs = transform(base_msgs) if transform else base_msgs
            # Compute per-request activation override
            save_dir = (
                Path("~/.bifrost/workspace").expanduser()
                / f"examples/emotion_prefix_gsm8k/results/{exp_dir.name}/activations/{variant}/{row['sample_id']}"
            )
            ndif = {"save_dir": str(save_dir), "per_request_subdir": False}
            endpoint = _make_endpoint(base, model, args.max_tokens, args.temperature, ndif)
            # Environment (no tools)
            class NoEnv:
                def get_tools(self):
                    return []
            env = NoEnv()
            # Evaluate single sample
            res = asyncio_run_evaluate(msgs, row, rewards, env, endpoint, rcfg)
            # Extract request_id from completion id
            try:
                completion_id = res.trajectory.completions[-1].id
                req_id = completion_id.split("chatcmpl-")[-1]
            except Exception:
                req_id = "unknown"
            breadcrumb = {"save_dir": str(save_dir), "request_id": req_id}
            manifest[f"{variant}/{row['sample_id']}"] = breadcrumb
            _save_eval_outputs(exp_dir, variant, row["sample_id"], res, breadcrumb)
            # Simple correctness accounting
            ok = res.metrics.get("correctness", 0.0) > 0.5
            report["correct"] += 1 if ok else 0
            report["total"] += 1

    acc = report["correct"] / max(1, report["total"]) if report["total"] else 0.0
    (exp_dir / "report.json").write_text(json.dumps({"accuracy": acc, **report}, indent=2), encoding="utf-8")
    (exp_dir / "activations_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"🎉 Generate complete. Accuracy={acc:.3f}")
    return {"accuracy": acc}


def asyncio_run_evaluate(msgs: List[Message], row: Dict[str, Any], rewards, env, endpoint: Endpoint, rcfg: RunConfig) -> EvalSample:
    import asyncio
    async def _go():
        # Bind rewards to sample
        reward_fns = [(name, fn(row)) if callable(fn) else (name, fn) for name, fn in rewards]
        return await evaluate_sample(row, row["sample_id"], lambda _s: msgs, reward_fns, env, endpoint, rcfg, max_turns=10, verbose=False)
    return asyncio.run(_go())


def stage_analyze(exp_dir: Path) -> None:
    # Minimal: trust report.json; for deeper analysis, add separate script later
    p = exp_dir / "report.json"
    if p.exists():
        print(p.read_text())
    else:
        print("No report.json found")


def stage_download_activations(args, exp_dir: Path, deploy_info: Dict[str, Any]) -> None:
    base_manifest = json.loads((exp_dir / "activations_manifest.json").read_text())
    ssh = deploy_info["ssh"]
    bc = BifrostClient(ssh)
    for key, bc_info in base_manifest.items():
        remote_dir = Path(bc_info["save_dir"]).expanduser() / bc_info["request_id"]
        local_dir = exp_dir / "activations" / key
        local_dir.mkdir(parents=True, exist_ok=True)
        print(f"⬇️  {key} ← {remote_dir}")
        bc.download_files(str(remote_dir), str(local_dir), recursive=True)


def stage_monitor(args, deploy_info: Dict[str, Any]) -> None:
    ssh = deploy_info["ssh"]
    bc = BifrostClient(ssh)
    log = "~/nnsight_server.log"
    print(f"📡 tail -f {log}")
    # One-shot tail for now
    out = bc.exec(f"tail -n 200 {log}")
    print(out)


def main():
    # Parse --config first so we can seed argparse defaults from YAML
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, help="YAML file with argument defaults")
    pre_args, remaining_argv = pre.parse_known_args()

    config_defaults: Dict[str, Any] = {}
    if pre_args.config:
        cfg_path = Path(pre_args.config)
        if not cfg_path.exists():
            raise SystemExit(f"Config not found: {cfg_path}")
        loaded = yaml.safe_load(cfg_path.read_text()) or {}
        if not isinstance(loaded, dict):
            raise SystemExit(f"Config must be a mapping: {cfg_path}")
        config_defaults = _normalize_config_defaults(loaded)

    ap = argparse.ArgumentParser(description="Deploy NNsight server and evaluate GSM8K with emotional prefixes")
    ap.add_argument("--config", type=str, default=None, help="YAML file with argument defaults")
    ap.add_argument("--stages", nargs="*", default=["deploy", "configure", "generate"], help="Stages to run")
    ap.add_argument("--name", default="emopfx")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--gpu-type", default=None)
    ap.add_argument("--manufacturer", default=None)
    ap.add_argument("--min-vram", type=int, default=12)
    ap.add_argument("--max-price", type=float, default=0.40)
    ap.add_argument("--model", default="willcb/Qwen3-0.6B")
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--variants", type=str, default="control,frustration,impatience,anxiety,collaborative,patience,calm")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--no-wait-health", action="store_true")
    if config_defaults:
        valid_dests = {action.dest for action in ap._actions}
        filtered_defaults = {k: v for k, v in config_defaults.items() if k in valid_dests}
        ap.set_defaults(**filtered_defaults)
        if "config" not in filtered_defaults and pre_args.config:
            ap.set_defaults(config=str(pre_args.config))
    args = ap.parse_args(remaining_argv)
    if pre_args.config and not args.config:
        args.config = pre_args.config

    variants = _variants_list(args.variants)
    if args.out:
        exp_dir = Path(args.out)
    else:
        # Use latest existing results dir for this name if present; otherwise create new
        base = Path("examples/emotion_prefix_gsm8k/results")
        pattern = f"{args.name}_"
        if base.exists():
            candidates = sorted([p for p in base.iterdir() if p.is_dir() and p.name.startswith(pattern)], key=lambda p: p.stat().st_mtime, reverse=True)
            # Prefer those with experiment.yaml present
            candidates_with_yaml = [p for p in candidates if (p / "experiment.yaml").exists()]
            if candidates_with_yaml:
                candidates = candidates_with_yaml
        else:
            candidates = []
        exp_dir = candidates[0] if candidates else _exp_dir(args.name)
    exp_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Experiment dir: {exp_dir}")

    # Load or create experiment config
    yaml_path = exp_dir / "experiment.yaml"
    config: Dict[str, Any] = {}
    if yaml_path.exists():
        config = yaml.safe_load(yaml_path.read_text()) or {}

    # Stage: deploy
    deploy_info: Dict[str, Any] = config.get("deploy", {})
    if "deploy" in args.stages:
        deploy_info = stage_deploy(args, exp_dir)
        config["deploy"] = deploy_info
        _save_yaml(yaml_path, config)

    # Stage: configure
    cfg_info: Dict[str, Any] = config.get("configure", {})
    if "configure" in args.stages:
        cfg_info = stage_configure(args, exp_dir, deploy_info)
        config["configure"] = cfg_info
        _save_yaml(yaml_path, config)

    # Stage: generate
    if "generate" in args.stages:
        gen_info = stage_generate(args, exp_dir, deploy_info, cfg_info.get("structure", {}), variants)
        config["generate"] = gen_info
        _save_yaml(yaml_path, config)
        # JSON mirror
        (exp_dir / "experiment_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    if "analyze" in args.stages:
        stage_analyze(exp_dir)

    if "download-activations" in args.stages:
        stage_download_activations(args, exp_dir, deploy_info)

    if "monitor" in args.stages:
        stage_monitor(args, deploy_info)


if __name__ == "__main__":
    main()
