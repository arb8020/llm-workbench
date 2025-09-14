GSM8K Remote NNsight — Debug Handoff

This document is for quickly debugging the NNsight example server on a remote GPU when the proxy returns 502 or the server appears to exit right after dependency resolution.

Current Layout
- Server package: examples/gsm8k_remote_nnsight/server/
  - Entry module: examples.gsm8k_remote_nnsight.server.server
  - Launcher: server/start_server.sh (installs uv, syncs deps, starts server, logs to ~/nnsight_server.log)
  - Utilities: server/test_client.py, server/smoke.py
- Example deploy helpers:
  - examples/gsm8k_remote_nnsight/deploy.py (provisions, pushes, starts tmux session nnsight-server)
  - examples/gsm8k_remote_nnsight/deploy_and_collect.py (demo run + collect)
- Defaults: port 8000

Symptoms Seen
- Curling proxy https://<pod>-8000.proxy.runpod.net/health returns 502
- Remote logs show server returns 200 locally
- Manually running start_server.sh prints “resolved/audited packages” then exits immediately (no ~/nnsight_server.log), or no tmux session found

Most Likely Causes
1) Proxy port mismatch
   - Instance was provisioned with a different exposed port (e.g., 8002). The proxy for 8000 won’t work and returns 502.
2) Wrong module path after refactor
   - Launcher or command still tries -m examples.gsm8k_remote_nnsight.server instead of ...server.server, causing uv run to exit immediately after sync.
3) tmux not installed on the image
   - Session never starts, so no ~/nnsight_server.log.

Quick Triage (copy/paste)
Replace IP and SSHPORT below.

1) Verify tmux + start the session
- Ensure tmux is installed:
  - python -m bifrost.bifrost.exec 'root@IP:SSHPORT' "which tmux || (apt-get update -y && apt-get install -y tmux)"
- Start a fresh session on 8000:
  - python -m bifrost.bifrost.exec 'root@IP:SSHPORT' "tmux kill-session -t nnsight-server 2>/dev/null || true; tmux new-session -d -s nnsight-server 'cd ~/.bifrost/workspace && bash examples/gsm8k_remote_nnsight/server/start_server.sh --host 0.0.0.0 --port 8000 --model willcb/Qwen3-0.6B --device-map auto 2>&1 | tee ~/nnsight_server.log'"
- Check session exists:
  - python -m bifrost.bifrost.exec 'root@IP:SSHPORT' "tmux list-sessions || echo 'no tmux sessions'"

2) Inspect logs
- Tmux pane:
  - python -m bifrost.bifrost.exec 'root@IP:SSHPORT' "tmux capture-pane -pt nnsight-server -S -5000"
- Persistent log:
  - python -m bifrost.bifrost.exec 'root@IP:SSHPORT' "tail -n 200 ~/nnsight_server.log"

3) Confirm server is listening locally
- python -m bifrost.bifrost.exec 'root@IP:SSHPORT' "ss -lntp | grep ':8000' || lsof -i :8000"
- python -m bifrost.bifrost.exec 'root@IP:SSHPORT' "curl -sS -i http://localhost:8000/health"

4) If the launcher exits immediately, confirm module path
- Show launcher file:
  - python -m bifrost.bifrost.exec 'root@IP:SSHPORT' "sed -n '1,120p' ~/.bifrost/workspace/examples/gsm8k_remote_nnsight/server/start_server.sh"
  - Must contain: python -m examples.gsm8k_remote_nnsight.server.server
- Test import directly:
  - python -m bifrost.bifrost.exec 'root@IP:SSHPORT' "cd ~/.bifrost/workspace && uv run python - <<'PY'\nimport importlib\nimport sys\nprint('Python:', sys.version)\nimport examples.gsm8k_remote_nnsight.server.server as srv\nprint('import OK')\nPY"

Proxy Mapping Gotcha
- 502 on proxy usually means wrong exposed port for the instance.
- Use the “HTTP (proxy)” printed by deploy. If it ends with -8002.proxy..., you must either:
  - Start server on 8002, or
  - Provision a fresh instance name with --port 8000 so the proxy maps 8000.

Manual Commands (no tmux) – to surface errors
- python -m bifrost.bifrost.exec 'root@IP:SSHPORT' "cd ~/.bifrost/workspace && bash -x examples/gsm8k_remote_nnsight/server/start_server.sh --host 0.0.0.0 --port 8000 --model willcb/Qwen3-0.6B --device-map auto | tail -n 200"
- If you see it terminate after uv sync, it’s almost certainly the module path or import error — fix the launcher or re-push latest code.

Redeploy (clean)
- Fresh name to guarantee correct port mapping (8000):
  - python examples/gsm8k_remote_nnsight/deploy.py --name nano-ndif-8000 --port 8000 --wait-health --run-smoke
- Or top-level script:
  - python scripts/deploy_nano_ndif.py --name nano-ndif-8000 --port 8000 --wait-health --run-smoke

Post-Deploy Validation
- Health:
  - curl -sS -i 'https://<pod>-8000.proxy.runpod.net/health'
- Test client (configures interventions + chat):
  - python examples/gsm8k_remote_nnsight/server/test_client.py --base-url https://<proxy-url> --ssh root@IP:SSHPORT

Notes
- Code organization: all server concerns live under server/; GSM8K evaluation/demo scripts live at the example root.
- Logs: server logs → ~/nnsight_server.log; tmux session → nnsight-server.
- If the server listens locally but proxy fails, it’s a provisioning port mismatch — redeploy with a new name and desired port.

