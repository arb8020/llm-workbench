# TODOs from Code Review (2025-09-19)

Each item below documents an issue found during the review. Treat every TODO as a self-contained task: understand the bug, reproduce it if needed, and implement the fix. Code snippets include file paths and line numbers to help you get oriented quickly.

For every TODO you will find:
- A summary of the problem and why it matters.
- Relevant code excerpts (sometimes multiple snippets for additional context).
- Several possible solution paths with notes on tradeoffs so you can choose the most appropriate remediation plan.

## Execution Order

**Phase 1: Evaluation Infrastructure (Foundation)** ✅ COMPLETED
1. ✅ TODO 4 – Fix checkpoint deserialization registry handling
2. ✅ TODO 5 – Generate reward functions per GSM8K sample
3. ✅ TODO 10 – Handle empty or failed GSM8K datasets safely

**Phase 2: Deployment Reliability (Credential Threading & Error Handling)** ← COMPLETE ✅
4. ✅ TODO 6 – Respect API key arguments in GPU client
5. ✅ TODO 8 – Pass SSH key and timeout through Bifrost deployments
6. ✅ TODO 9 – Harden GPU provisioning error paths
7. ✅ TODO 11 – Wait for vLLM readiness before evaluation

**Phase 3: Security & Isolated Fixes**
8. TODO 1 – Replace unsafe `eval` in calculator tool
9. TODO 3 – Treat `MAX_TURNS` as failure inside search tools
10. TODO 7 – Restore streaming SSH support in broker

---

## TODO 1 – Replace unsafe `eval` in calculator tool
- **Severity:** Critical (remote code execution)
- **Primary Location:** `examples/gsm8k_remote/deploy_and_evaluate.py:252-279`
- **Problem:** The calculator environment executes `eval` on arbitrary strings coming from the LLM. This gives the model full Python execution on the orchestrator.
- **Context Snippets:**
  ```python
  # examples/gsm8k_remote/deploy_and_evaluate.py:231-276
  class CalculatorEnvironment(Environment):
      ...
      async def exec_tool(self, tool_call: ToolCall, current_state: AgentState,
                         run_config: RunConfig, checkpoint_store=None) -> ToolResult:
          if tool_call.name == "calculate":
              expression = tool_call.args.get("expression", "")
              try:
                  result = eval(expression.replace("^", "**"))  # ⚠️ arbitrary execution
                  return ToolResult(...)
  ```
  ```python
  # Example payload the model can send today
  {"name": "calculate", "arguments": {"expression": "__import__('os').system('uname -a')"}}
  ```
- **Potential Approaches & Tradeoffs:**
  - *Use a safe math interpreter* (e.g., `asteval`, `numexpr`, or PyPy’s `simpleeval`). These libraries cover common arithmetic but add a dependency and may still need configuration to disable advanced features.
  - *Build a minimal parser* (tokenize digits, operators, parentheses). Provides maximum control and zero extra dependencies, but requires more engineering effort and thorough testing (order of operations, unary minus, decimals, etc.).
  - *Call out to an external calculator process* (e.g., run `python -c` in a constrained container or use `mathjs` via subprocess). Isolates risk at the cost of latency and operational complexity.
  - Whichever path you choose, include tests ensuring payloads like `__import__('os')`, `open('/etc/passwd')`, or `lambda: 0` raise errors instead of executing.

---

## ✅ TODO 2 – Rehydrate environments per evaluation sample (COMPLETED)
- **Severity:** High (state leakage, race conditions)
- **Primary Location:** `rollouts/rollouts/evaluation.py:167-205`
- **Problem:** `evaluate_sample` reused the same `Environment` instance for every sample and also handed the same object to each concurrent task. Stateful environments (e.g., calculator with accumulated value, search wrappers capturing depth) would pollute subsequent runs and break concurrency.
- **Solution Implemented:** Changed `evaluate()` and `simple_evaluate()` to accept `environment_factory: Callable[[], Environment]` instead of `environment: Environment`. Each sample now gets a fresh environment instance by calling the factory function. Updated all example scripts to pass factory functions (e.g., `lambda: CalculatorEnvironment()`).
- **Files Modified:**
  - `rollouts/rollouts/evaluation.py` - Updated function signatures and implementation
  - `examples/gsm8k_remote/deploy_and_evaluate.py` - Uses environment factories
  - `examples/gsm8k_local/gsm8k_rewards.py` - Uses environment factories
  - `examples/gsm8k_local/gsm8k_catgirl_prompt.py` - Uses environment factories
- **Context Snippets:**
  ```python
  # rollouts/rollouts/evaluation.py:167-188
  actor = Actor(
      trajectory=initial_trajectory,
      endpoint=endpoint,
      tools=environment.get_tools()
  )

  initial_state = AgentState(
      actor=actor,
      environment=environment,        # Reused across samples
      max_turns=max_turns
  )
  ```
  ```python
  # rollouts/rollouts/evaluation.py:232-267 (parallel path)
  semaphore = asyncio.Semaphore(max_concurrent)
  tasks = [
      eval_with_semaphore(sample_id, sample_data)
      for sample_id, sample_data in samples_to_eval
  ]
  results = await asyncio.gather(*tasks)
  ```
- **Potential Approaches & Tradeoffs:**
  - *Provide an `environment_factory` callable* that returns a new environment per sample. Simple conceptually but requires touching every call site (GSM8K scripts, tests, etc.).
  - *Add cloning support to `Environment`* (e.g., require `serialize`/`deserialize` and use those to deep-copy before each run). Minimal API churn but depends on every environment implementing serialization correctly; failure surfaces as runtime errors.
  - *Instantiate environments inside `evaluate_sample` based on metadata in the sample.* Keeps API ergonomic for basic use but shifts responsibility to dataset authors and complicates existing pipelines.
  - After refactor, add tests where a stateful environment increments a counter and verify counts reset between samples and across concurrent runs.

---

## TODO 3 – Treat `MAX_TURNS` as failure inside search tools
- **Severity:** High (incorrect success reporting)
- **Primary Location:** `rollouts/rollouts/environments/advanced_search.py:334-387` & `412-470`
- **Problem:** When a branch or decompose sub-agent times out (`StopReason.MAX_TURNS`), the wrapper reports success. The parent agent receives no signal that work failed.
- **Context Snippets:**
  ```python
  # rollouts/rollouts/environments/advanced_search.py:365-377
  if not final_sub_state.stop or final_sub_state.stop in [StopReason.MAX_TURNS, StopReason.TASK_COMPLETED]:
      print(f"  ✅ Approach '{approach['name']}' succeeded!")
      return ToolResult(ok=True, content=...)
  ```
  ```python
  # rollouts/rollouts/environments/advanced_search.py:440-458
  results.append({
      "name": subproblem['name'],
      "success": not final_sub_state.stop or final_sub_state.stop in [StopReason.MAX_TURNS, StopReason.TASK_COMPLETED],
      "summary": result_content,
  })
  ```
- **Potential Approaches & Tradeoffs:**
  - *Reclassify timeouts as failures and bubble detailed reasons back.* Straightforward but might break existing consumers relying on the current (incorrect) semantics—validate downstream usages.
  - *Expose a configurable allowlist of “success” stop reasons.* Adds flexibility and backward compatibility but increases API surface and cognitive overhead.
  - *Return partial success data plus explicit `timeout` metadata.* Provides richer telemetry but requires updating serialization/storage formats.
  - Regardless, add tests covering branch/decompose scenarios where subagents exhaust turns to assert failures propagate properly.

---

## ✅ TODO 4 – Fix checkpoint deserialization registry handling (COMPLETED)
- **Severity:** High (checkpoint restore currently broken)
- **Primary Location:** `rollouts/rollouts/checkpoints.py:36-92`
- **Problem:** `deserialize_agent_state` relies on an `environment_registry` mutable default `{}` that starts empty. Unless callers mutate the store's internals, loading any checkpoint raises `KeyError` when looking up the environment class.
- **Solution Implemented:** Made `environment_registry` a required parameter (removed mutable default), fixed type annotation to `Dict[str, type[Environment]]`, and updated `FileCheckpointStore.__init__` to require the registry. All call sites in `binary_search.py` and `search_agent.py` now pass explicit registries.
- **Files Modified:**
  - `rollouts/rollouts/checkpoints.py` - Updated function and class signatures
  - `rollouts/rollouts/environments/binary_search.py` - Passes registry to FileCheckpointStore
  - `rollouts/examples/search_agent.py` - Passes registry to FileCheckpointStore
- **Context Snippets:**
  ```python
  # rollouts/rollouts/checkpoints.py:36-52
  async def deserialize_agent_state(
          data: Dict[str, Any],
          environment_registry: Dict[str, Environment] = {},
      ) -> AgentState:
      env_class_name = data["environment"]["class_name"]
      env_class = environment_registry[env_class_name]  # KeyError
  ```
  ```python
  # rollouts/rollouts/checkpoints.py:55-66
  class FileCheckpointStore:
      def __init__(self, directory: str = "/tmp/rollouts-agent-checkpoints"):
          ...
          self.environment_registry = {
              # "CalculatorEnvironment": CalculatorEnvironment,
          }
  ```
- **Potential Approaches & Tradeoffs:**
  - *Require the registry as a mandatory argument.* Forces explicit configuration, preventing silent failure, but breaks existing call sites.
  - *Ship a global registry module and auto-populate common environments.* Lowest friction for users but may hide missing registrations or drift if new environments aren’t added in time.
  - *Store the fully-qualified class path in the checkpoint and import dynamically on load.* Most flexible but introduces module import risks and security considerations (loading arbitrary modules from checkpoints).
  - Add tests that save and load agent states using at least one environment to prove the fix.

---

## ✅ TODO 5 – Generate reward functions per GSM8K sample (COMPLETED)
- **Severity:** High (accuracy metrics invalid)
- **Primary Location:** `examples/gsm8k_remote/deploy_and_evaluate.py:486-534`
- **Problem:** Evaluation uses rewards derived from `dataset_samples[0]` for every sample. Correctness checks therefore compare each trajectory against the wrong ground truth.
- **Solution Implemented:** Changed `RewardFunction` signature from `Callable[[Trajectory], float]` to `Callable[[Trajectory, Dict[str, Any]], float]`. Reward functions now receive sample data as an explicit parameter, eliminating closure-based bugs. Updated all reward functions to take both trajectory and sample, added assert statements for required keys, and removed the closure-based factory pattern.
- **Files Modified:**
  - `rollouts/rollouts/evaluation.py` - Updated type signature and evaluate_sample to pass sample_data
  - `examples/gsm8k_remote/deploy_and_evaluate.py` - Updated all reward functions, removed factory pattern
  - `examples/gsm8k_local/gsm8k_rewards.py` - Updated all reward functions, removed factory pattern
  - `examples/gsm8k_local/gsm8k_catgirl_prompt.py` - Updated all reward functions, removed factory pattern
- **Context Snippets:**
  ```python
  # examples/gsm8k_remote/deploy_and_evaluate.py:486-514
  def create_reward_functions_for_sample(sample: Dict[str, Any]):
      correctness_fn = make_correctness_reward(sample)
      sample_rewards = [
          ("correctness", correctness_fn),
          ("format", format_reward),
          ("efficiency", efficiency_reward),
      ]
      if mode == "with-tools":
          sample_rewards.append(("tool_usage", tool_usage_reward))
      return sample_rewards

  # Only applied to first sample
  demo_rewards = create_reward_functions_for_sample(dataset_samples[0])
  report = await evaluate(..., reward_functions=demo_rewards, ...)
  ```
- **Potential Approaches & Tradeoffs:**
  - *Allow `reward_functions` to be a callable receiving `(sample_id, sample_data)`.* Minimal changes to existing code but requires altering the evaluation signature and handling backward compatibility carefully.
  - *Extend `evaluate` to accept a list/iterator of reward function sets aligned with the dataset.* Straightforward but may increase memory usage and complicate streaming scenarios.
  - *Embed expected answers inside the `Environment` or `RunConfig` so rewards can be computed post-hoc.* Keeps evaluation API stable but entangles reward logic with environment internals.
  - Add regression coverage that runs with two samples having different answers and asserts accuracy counts both correctly.

---

## ✅ TODO 6 – Respect API key arguments in GPU client (COMPLETED)
- **Severity:** High (authentication failures when keys are passed programmatically)
- **Primary Locations:**
  - `broker/broker/client.py:49-138`
  - `broker/broker/providers/runpod.py:43-81`
- **Problem:** `GPUClient(api_key=...)` stores the key on the client but provider code ignores it and still reads `RUNPOD_API_KEY` from the environment. Users supplying credentials in code receive auth errors.
- **Solution Implemented:** Threaded `api_key` parameter through all layers (GPUClient → broker/api.py → broker/providers/runpod.py). Removed `_get_api_key()` fallback function. All provider functions now require explicit `api_key` parameter. Updated examples to use pattern: `load_dotenv()` → `RUNPOD_API_KEY = os.environ.get('RUNPOD_API_KEY')` → `GPUClient(api_key=RUNPOD_API_KEY)`.
- **Files Modified:**
  - `broker/broker/providers/runpod.py` - Added `api_key` parameter to all functions, removed `_get_api_key()`
  - `broker/broker/api.py` - Added `api_key` parameter to all functions
  - `broker/broker/client.py` - Pass `self._api_key` to all api layer calls
  - `examples/gsm8k_remote/deploy_and_evaluate.py` - Load and pass RUNPOD_API_KEY explicitly
  - `broker/examples/client_example.py` - Updated to show explicit pattern
  - `broker/examples/simple_example.py` - Updated to use GPUClient with explicit api_key
- **Context Snippets:**
  ```python
  # broker/broker/client.py:58-70
  if api_key:
      self._api_key = api_key
  else:
      load_dotenv()
      self._api_key = os.environ.get('RUNPOD_API_KEY')
      if not self._api_key:
          raise ValueError(...)
  ```
  ```python
  # broker/broker/providers/runpod.py:43-56
  def _get_api_key() -> str:
      load_dotenv()
      api_key = os.environ.get("RUNPOD_API_KEY")
      if not api_key:
          raise ValueError("RUNPOD_API_KEY environment variable not set")
      return api_key
  ```
  ```python
  # broker/broker/api.py:102-122
  def create(...):
      suitable_offers = search(...)
      for offer in suitable_offers[:max_attempts]:
          request = ProvisionRequest(...)
          if offer.provider == "runpod":
              instance = runpod.provision_instance(request, request.ssh_startup_script)
  ```
- **Potential Approaches & Tradeoffs:**
  - *Thread the API key through `ProvisionRequest` or context objects.* Provides explicit control but requires updating many signatures.
  - *Use a provider registry object carrying shared configuration.* Cleaner design for multi-provider support yet a larger refactor.
  - *Set environment variables dynamically before provider calls.* Quick fix but can lead to surprising behavior in multi-threaded contexts or when mixing providers.
  - Add unit tests mocking `requests.post` to verify the supplied key is used when provided via constructor.

---

## TODO 7 – Restore streaming SSH support in broker
- **Severity:** High (runtime crash on streaming commands)
- **Primary Location:** `broker/broker/client.py:368-418`
- **Problem:** `exec_streaming` imports `.ssh_clients`, but only `ssh_clients_compat.py` exists. Attempting to stream logs raises `ModuleNotFoundError`.
- **Context Snippets:**
  ```python
  # broker/broker/client.py:395-407
  from .ssh_clients import ParamikoSSHClient, get_ssh_connection_info  # ImportError
  ...
  hostname, port, username = get_ssh_connection_info(self._instance)
  client = ParamikoSSHClient()
  ```
  ```python
  # repo structure
  broker/broker/ssh_clients_compat.py
  ```
- **Potential Approaches & Tradeoffs:**
  - *Update the import to reference `ssh_clients_compat`.* Fastest path but evaluate whether the compat module exposes the same API.
  - *Restore the original module name (add a shim file).* Maintains backwards compatibility for any external imports but slightly increases maintenance burden.
  - *Re-export the desired symbols from `__init__.py`.* Keeps import sites unchanged but may mask missing functionality if compat layer is incomplete.
  - Add a smoke test calling `exec_streaming` with a mocked SSH backend to ensure no import errors remain.

---

## ✅ TODO 8 – Pass SSH key and timeout through Bifrost deployments (COMPLETED)
- **Severity:** High (deployment failures when non-default SSH configuration is used)
- **Primary Locations:**
  - `bifrost/bifrost/client.py:37-205`
  - `bifrost/bifrost/deploy.py:404-520`
- **Problem:** `BifrostClient` accepts `ssh_key_path` and custom timeouts but `GitDeployment` reconnects via Paramiko without those parameters. Deployments succeed only when the default SSH agent has access to the key.
- **Solution Implemented:** Refactored GitDeployment methods to accept pre-configured `paramiko.SSHClient` instead of creating new clients. BifrostClient now passes its managed `_ssh_client` (which includes ssh_key_path and timeout) to all GitDeployment methods. This follows the **facade pattern** - BifrostClient owns connection lifecycle, GitDeployment focuses on git operations.
- **Files Modified:**
  - `bifrost/bifrost/deploy.py` - All deployment methods now accept `client: paramiko.SSHClient` as first parameter:
    - `deploy_to_workspace(client, ...)` - Removed client creation/connect/close
    - `deploy_code_only(client, ...)` - Removed client creation/connect/close
    - `deploy_and_execute(client, ...)` - Removed client creation/connect/close
    - `deploy_and_execute_detached(client, ...)` - Removed client creation/connect/close
    - `deploy_and_execute_detached_workspace(client, ...)` - Removed client creation/connect/close
    - `_execute_detached_deployment()` - Removed redundant client.connect() call
    - Removed `_create_ssh_client()` method entirely
  - `bifrost/bifrost/client.py` - BifrostClient now passes its `_ssh_client` to GitDeployment:
    - `push()` method calls `self._get_ssh_client()` and passes to deployment methods
    - `run_detached()` method calls `self._get_ssh_client()` and passes to deployment methods
- **Context Snippets:**
  ```python
  # bifrost/bifrost/client.py:40-66
  class BifrostClient:
      def __init__(..., ssh_key_path: Optional[str] = None, timeout: int = 30, ...):
          self.ssh_key_path = ssh_key_path
          self.timeout = timeout
  ```
  ```python
  # bifrost/bifrost/client.py:129-166
  deployment = GitDeployment(self.ssh.user, self.ssh.host, self.ssh.port)
  deployment.deploy_to_workspace(workspace_path, uv_extra)
  ```
  ```python
  # bifrost/bifrost/deploy.py:424-470
  client.connect(hostname=self.ssh_host, port=self.ssh_port, username=self.ssh_user)
  # No key or timeout passed; relies on default SSH agent
  ```
- **Potential Approaches & Tradeoffs:**
  - *Pass `ssh_key_path` into `GitDeployment` and load the key explicitly.* Keeps behavior deterministic but needs careful handling of key formats (RSA, Ed25519, etc.).
  - *Let callers inject a ready-made Paramiko client.* Most flexible but bigger API change; may complicate simple scripts.
  - *Export the key via environment variables for the bootstrap commands.* Quick change yet leaks secrets into process environment and relies on remote shell configuration.
  - Add mocked unit tests verifying Paramiko connects with supplied credentials instead of defaulting to the agent.

---

## ✅ TODO 9 – Harden GPU provisioning error paths (COMPLETED)
- **Severity:** Medium (resource leaks, crashes on failure)
- **Primary Location:** `examples/gsm8k_remote/deploy_and_evaluate.py:84-142`
- **Problem:** Provisioning assumes success. If `GPUClient.create` returns `None` or SSH never becomes ready, the code dereferences `None` or exits without cleaning up.
- **Solution Implemented:** Applied **try/finally pattern** with early None check. Checks if `gpu_instance` is None immediately after creation and exits cleanly (no cleanup needed). Once instance exists, wraps all deployment steps in try/finally to guarantee termination on any error (SSH timeout, deployment failure, etc.). Includes nested try/except for cleanup itself in case termination fails.
- **Files Modified:**
  - `examples/gsm8k_remote/deploy_and_evaluate.py`:
    - `deploy_qwen_vllm_server()` - Added None check after `gpu_client.create()` with helpful error message
    - `deploy_qwen_vllm_server()` - Wrapped SSH wait and deployment in try/finally with guaranteed cleanup
    - `deploy_qwen_vllm_server()` - Raises RuntimeError on SSH timeout instead of sys.exit()
    - `deploy_qwen_vllm_server()` - Exception handler terminates GPU and provides manual cleanup command if termination fails
    - `main()` - Added try/except around `deploy_qwen_vllm_server()` call to handle re-raised exceptions gracefully
- **Context Snippets:**
  ```python
  # examples/gsm8k_remote/deploy_and_evaluate.py:84-107
  gpu_instance = gpu_client.create(
      query=query,
      exposed_ports=[8000],
      enable_http_proxy=True,
      name="qwen-vllm-server",
      cloud_type="secure",
      sort=lambda x: x.price_per_hour,
  )
  print(f"✅ GPU ready: {gpu_instance.id}")  # crashes if create returns None
  ```
  ```python
  # examples/gsm8k_remote/deploy_and_evaluate.py:100-107
  if not gpu_instance.wait_until_ssh_ready(timeout=300):
      print("❌ Failed to get SSH connection ready")
      sys.exit(1)  # exits without terminating instance
  ```
- **Potential Approaches & Tradeoffs:**
  - *Guard for `None` returns and retry or abort gracefully.* Quick fix but may mask underlying broker issues unless errors are logged clearly.
  - *Wrap provisioning in a context manager that guarantees cleanup.* More robust but requires broader changes to deployment scripts.
  - *Implement exponential backoff for SSH readiness.* Improves reliability, yet could extend startup time unnecessarily if not tuned.
  - Add tests or simulated runs that mock `GPUClient` to return `None` or a slow-starting instance and assert clean shutdown.

---

## ✅ TODO 10 – Handle empty or failed GSM8K datasets safely (COMPLETED)
- **Severity:** Medium (crash on edge cases)
- **Primary Location:** `examples/gsm8k_remote/deploy_and_evaluate.py:500-523`
- **Problem:** The script assumes `dataset_samples` has at least one entry. When `--samples 0` or the dataset fetch fails, indexing `[0]` raises `IndexError` before any messaging to the user.
- **Solution Implemented:** Added fail-fast validation with specific error messages for three failure modes: (1) dataset file not found (FileNotFoundError), (2) dataset loading fails (general Exception), and (3) empty dataset or samples=0 requested. Each error provides clear guidance to the user.
- **Files Modified:**
  - `examples/gsm8k_remote/deploy_and_evaluate.py` - Added dataset validation with try/except and empty checks
  - `examples/gsm8k_local/gsm8k_rewards.py` - Added dataset validation with try/except and empty checks
  - `examples/gsm8k_local/gsm8k_catgirl_prompt.py` - Added dataset validation with try/except and empty checks
- **Context Snippet:**
  ```python
  # examples/gsm8k_remote/deploy_and_evaluate.py:500-509
  dataset_samples = list(load_jsonl(dataset_path))
  ...
  demo_rewards = create_reward_functions_for_sample(dataset_samples[0])
  ```
- **Potential Approaches & Tradeoffs:**
  - *Exit early with a descriptive error if the dataset is empty.* Minimal change but introduces another branch to maintain.
  - *Allow evaluation to continue with zero samples and output an empty report.* Makes the script more automation-friendly, yet downstream systems must handle empty reports.
  - *Fetch rewards lazily per sample inside the evaluation loop.* Avoids touching index 0 prematurely but requires coordinating with the fix from TODO 5.
  - Add a regression test invoking the script with `--samples 0` (or mock dataset) and confirm the behavior is user-friendly.

---

## ✅ TODO 11 – Wait for vLLM readiness before evaluation (COMPLETED)
- **Severity:** Medium (flaky startup)
- **Primary Location:** `examples/gsm8k_remote/deploy_and_evaluate.py:116-140` & subsequent evaluation block
- **Problem:** After launching `uv run python -m vllm.entrypoints.openai.api_server`, the script immediately starts evaluation even though model load can take minutes. Early evaluation attempts fail with connection errors.
- **Solution Implemented:** Added `wait_for_vllm_ready()` function that polls the `/v1/models` endpoint every 5 seconds until the server responds with a valid model list or timeout (300s). Called this function as Step 6 between dataset loading and evaluation start. Provides clear progress feedback showing elapsed time and attempt count.
- **Files Modified:**
  - `examples/gsm8k_remote/deploy_and_evaluate.py`:
    - Added `wait_for_vllm_ready()` function - polls `/v1/models` with 5s interval, 300s timeout
    - Added `requests` import for HTTP polling
    - Inserted readiness check as Step 6 (between dataset loading and evaluation)
    - Exits with helpful error message if server doesn't become ready
    - Renumbered subsequent steps (7-10)
- **Context Snippets:**
  ```python
  # examples/gsm8k_remote/deploy_and_evaluate.py:116-140
  tmux_cmd = "tmux new-session -d -s qwen-vllm 'cd ~/.bifrost/workspace && {vllm_cmd} 2>&1 | tee ~/qwen_vllm_server.log'"
  bifrost_client.exec(tmux_cmd)
  print("✅ Qwen3-0.6B vLLM server starting...")
  print("📋 Server will be ready in 2-3 minutes for model loading")
  # No readiness check before evaluation below
  ```
  ```python
  # Later (Step 6)
  report = await evaluate(...)
  ```
- **Potential Approaches & Tradeoffs:**
  - *Poll `/v1/models` until success with a timeout.* Provides explicit readiness but adds delay even when the server comes up quickly.
  - *Tail the remote tmux log until a known “server ready” message appears.* Works even without HTTP readiness but couples logic to log format.
  - *Expose a health-check command in the deployment (e.g., run a lightweight prompt) before launching the full evaluation.* Highest confidence but may incur extra token cost or latency.
  - Add an integration or scripted manual step verifying readiness detection works and aborts gracefully on timeout.

---

Please keep this document updated as items are addressed. Each TODO should be closed out with a linked PR and accompanying tests or validation notes.
