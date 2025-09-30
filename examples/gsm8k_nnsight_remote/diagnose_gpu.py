#!/usr/bin/env python3
"""
Diagnose a remote GPU running the single-pass NNsight server.

Checks:
- Who is listening on the given port (default 8001)
- /health HTTP code and /openapi.json presence + expected routes
- Running server processes (uvicorn/server_singlepass.py)
- tmux session output and last lines of server log
- Workspace venv presence and Python version

Usage examples:
  uv run python examples/gsm8k_nnsight_remote/diagnose_gpu.py --gpu-id <id> --port 8001
  uv run python examples/gsm8k_nnsight_remote/diagnose_gpu.py --ssh root@IP:PORT --port 8011
"""

import argparse
from typing import Optional

from broker.client import GPUClient
from bifrost.client import BifrostClient


def _exec(bc: BifrostClient, cmd: str) -> str:
    try:
        return bc.exec(cmd)
    except Exception as e:
        return f"<exec error: {e}>"


def diagnose(ssh: str, port: int, tail_lines: int) -> None:
    bc = BifrostClient(ssh)

    print("== Basic Info ==")
    print(f"SSH: {ssh}")
    print(f"Port: {port}")

    print("\n== Port Listeners ==")
    ss_out = _exec(bc, f"ss -ltnp | grep ':{port} ' || true")
    lsof_out = _exec(bc, f"lsof -iTCP -sTCP:LISTEN -P -n | grep ':{port} ' || true")
    print("ss:" if ss_out.strip() else "ss: (none)")
    if ss_out.strip():
        print(ss_out.strip())
    print("lsof:" if lsof_out.strip() else "lsof: (none)")
    if lsof_out.strip():
        print(lsof_out.strip())

    print("\n== Health and OpenAPI ==")
    code = _exec(bc, f"curl -s -o /dev/null -w '%{{http_code}}' http://localhost:{port}/health").strip()
    print(f"/health HTTP code: {code}")
    openapi = _exec(bc, f"curl -s http://localhost:{port}/openapi.json || true").strip()
    if openapi.startswith("{"):
        routes_ok = ("/models/load" in openapi) and ("/v1/chat/completions" in openapi)
        print(f"openapi.json present (routes OK: {routes_ok})")
        if not routes_ok:
            print("Note: openapi.json didn't include expected routes.")
    else:
        print("openapi.json missing or not JSON; first 200 chars:")
        print(openapi[:200])

    print("\n== Server Processes ==")
    ps = _exec(bc, "ps aux | grep -E 'uvicorn|server_singlepass.py' | grep -v grep || true").strip()
    print(ps if ps else "(no matching processes)")

    print("\n== tmux Sessions ==")
    tmux_ls = _exec(bc, "tmux ls || true").strip()
    print(tmux_ls if tmux_ls else "(no tmux sessions)")

    print("\n== tmux:nnsight-singlepass (last lines) ==")
    tmux_tail = _exec(bc, f"tmux capture-pane -t nnsight-singlepass -p | tail -n {tail_lines} || true").strip()
    print(tmux_tail if tmux_tail else "(no tmux output)")

    print("\n== ~/nnsight_singlepass.log (last lines) ==")
    log_tail = _exec(bc, f"tail -n {tail_lines} ~/nnsight_singlepass.log || true").strip()
    print(log_tail if log_tail else "(no log file)")

    print("\n== Workspace / Venv ==")
    venv = _exec(bc, "test -x ~/.bifrost/workspace/.venv/bin/python && echo present || echo missing").strip()
    print(f"venv: {venv}")
    if venv == "present":
        py_info = _exec(bc, "cd ~/.bifrost/workspace && . .venv/bin/activate && which python && python -V || true").strip()
        print(py_info)

    print("\n== Summary ==")
    likely_conflict = False
    if code == "200" and not openapi.startswith("{"):
        likely_conflict = True
    if not ps and likely_conflict:
        print("- Port appears to be served by a non-FastAPI process (e.g., nginx).")
        print("- Consider rerunning with a different port (e.g., --port 8011) and/or using --fresh.")
    elif ps and openapi.startswith("{"):
        print("- Server seems up and serving OpenAPI; POSTs should work.")
    else:
        print("- Server may not have started correctly; check tmux/log outputs above.")


def main():
    ap = argparse.ArgumentParser(description="Diagnose a GPU running the single-pass NNsight server")
    ap.add_argument("--gpu-id", default=None, help="GPU instance id to connect to")
    ap.add_argument("--ssh", default=None, help="Direct SSH string user@host:port")
    ap.add_argument("--port", type=int, default=8001, help="Port to check (default 8001)")
    ap.add_argument("--tail-lines", type=int, default=120, help="Lines to show from logs/tmux")
    args = ap.parse_args()

    if not args.gpu_id and not args.ssh:
        ap.error("Provide either --gpu-id or --ssh")

    if args.ssh:
        ssh = args.ssh
    else:
        gc = GPUClient()
        inst = gc.get_instance(args.gpu_id)
        if not inst:
            raise SystemExit(f"GPU {args.gpu_id} not found")
        # Try to ensure SSH is reachable quickly
        inst.wait_until_ssh_ready(timeout=120)
        ssh = inst.ssh_connection_string()

    diagnose(ssh, args.port, args.tail_lines)


if __name__ == "__main__":
    main()

