#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from bifrost.client import BifrostClient


def check_sessions(bc: BifrostClient) -> str:
    try:
        return bc.exec("tmux list-sessions 2>/dev/null || echo none").strip()
    except Exception as e:
        return f"error: {e}"


def tail_log(bc: BifrostClient, exp_name: str, lines: int = 40) -> str:
    # Prefer HOME path
    log_home = f"~/.bifrost/workspace/emotion_runs/{exp_name}.log"
    out = bc.exec(f"tail -n {lines} {log_home} 2>/dev/null || echo MISSING")
    if 'MISSING' in out:
        # Fallback to mislocated /Users path from earlier runs
        log_users = f"/Users/chiraagbalu/.bifrost/workspace/emotion_runs/{exp_name}.log"
        out2 = bc.exec(f"tail -n {lines} {log_users} 2>/dev/null || echo MISSING")
        return out2.strip()
    return out.strip()


def progress(bc: BifrostClient, exp_name: str) -> Dict[str, Any]:
    remote_cfg_home = f"~/.bifrost/workspace/emotion_runs/{exp_name}.json"
    remote_cfg_users = f"/Users/chiraagbalu/.bifrost/workspace/emotion_runs/{exp_name}.json"
    cfg_txt = bc.exec(f"cat {remote_cfg_home} 2>/dev/null || cat {remote_cfg_users} 2>/dev/null || echo MISSING")
    cfg: Dict[str, Any] = {}
    if 'MISSING' not in cfg_txt:
        try:
            cfg = json.loads(cfg_txt)
        except Exception:
            cfg = {}
    samples = int(cfg.get('samples', 0))
    variants = cfg.get('variants', [])
    total = samples * len(variants) if samples and variants else None
    # Count sample.json files on remote
    res_root = f"~/.bifrost/workspace/examples/emotion_prefix_gsm8k/results/{exp_name}"
    cnt_txt = bc.exec(f"find {res_root} -name 'sample.json' 2>/dev/null | wc -l || echo 0").strip()
    try:
        count = int(cnt_txt.splitlines()[-1].strip())
    except Exception:
        count = 0
    return {"expected": total, "found": count, "config": cfg}


def main():
    ap = argparse.ArgumentParser(description="Monitor remote emotion rollouts run")
    ap.add_argument("--ssh", required=True, help="SSH like root@host:port")
    ap.add_argument("--exp", required=True, help="Experiment name (dir name)")
    ap.add_argument("--tail", type=int, default=40, help="Lines of log to show")
    args = ap.parse_args()

    bc = BifrostClient(args.ssh)
    print("tmux sessions:\n" + check_sessions(bc))
    print("\nprogress:")
    pr = progress(bc, args.exp)
    exp = pr["expected"]
    found = pr["found"]
    if exp is not None:
        print(f"  {found}/{exp} completed")
    else:
        print(f"  {found} completed (expected unknown)")
    if pr["config"]:
        print(f"  config: samples={pr['config'].get('samples')} variants={pr['config'].get('variants')}")
    print("\nlog tail:")
    print(tail_log(bc, args.exp, args.tail))


if __name__ == "__main__":
    main()

