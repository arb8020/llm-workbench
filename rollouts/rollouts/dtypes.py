import json
from pathlib import Path
from enum import Enum
import os
import asyncio
from abc import ABC
from dataclasses import dataclass, field, asdict, fields
from typing import Any, Dict, List, Optional, Mapping, Union, TypeVar, Type, Iterator, Callable, Awaitable, Tuple
from datetime import datetime, timezone
import dacite

# TODO: Better torch typing options explored:
# 1. Create a Protocol for tensor-like objects (has .tolist(), .shape, .dtype) - cleanest approach
# 2. Use torch-stubs package if available for lightweight type info
# 3. Define proper Union types for tensor alternatives
# 4. Previous approach used TYPE_CHECKING conditional imports but created dependency issues
# 
# Current: Simple fallback for type hints - actual tensor handling is done at runtime via hasattr checks
TorchTensor = Any

# Verbose function for debugging
def verbose(level=1):
    """Check if verbose logging is enabled at given level"""
    return int(os.getenv("VERBOSE", 0)) >= level

T = TypeVar('T', bound='SerialDataclass')

class SerialDataclass:
    """Base class for dataclasses with JSON serialization support"""
    
    def to_json(self) -> str:
        """Serialize to JSON string"""
        return json.dumps(asdict(self), ensure_ascii=False) #type:ignore
    
    @classmethod
    def from_json(cls: Type[T], json_str: str) -> T:
        """Deserialize from JSON string using dacite"""
        data = json.loads(json_str)
        return dacite.from_dict(data_class=cls, data=data)
    
    def to_path(self, path: str | Path) -> None:
        """Save to file as JSON"""
        Path(path).write_text(self.to_json(), encoding="utf-8")
    
    @classmethod
    def from_path(cls: Type[T], path: str | Path) -> T:
        """Load from JSON file"""
        json_str = Path(path).read_text(encoding="utf-8")
        return cls.from_json(json_str)

@dataclass
class TrajectorySample:
    """
    One fully-processed GRPO sample, already padded / masked the way
    `ts_to_tensors` produced:

        seq         – [L]        int64    full token sequence (prompt+completion)
        attn        – [L]        bool     attention mask for `seq`
        action_mask – [L-1]      bool     1 ⇔ this position is in the completion
        old_logps   – [L-1]      float32  log π_old(a_t | s_t)
        advantages  – [L-1]      float32  per-token advantages (broadcast if scalar)
        ref_logps   – [L-1]      float32  log π_ref(a_t | s_t) (optional)
    """
    # torch.Tensor when available, object when not available
    seq:         'TorchTensor'                   # shape [L]
    attn:        'TorchTensor'                   # shape [L]  
    action_mask: 'TorchTensor'                   # shape [L-1]
    old_logps:   'TorchTensor'                   # shape [L-1]
    advantages:  'TorchTensor'                   # shape [L-1]
    ref_logps:   Optional['TorchTensor'] = None  # shape [L-1] or None

    # ───────────────────────────── JSON helpers ──────────────────────────────
    @staticmethod
    def _tensor_to_payload(t: Any) -> Any:
        try:
            import torch  # type: ignore[import-untyped]  # TODO: Same as above - improve torch typing
            if t is None:
                return None
            if hasattr(t, 'tolist') and hasattr(t, 'shape') and hasattr(t, 'dtype'):
                return {"values": t.tolist(), "shape": list(t.shape), "dtype": str(t.dtype)}
            return t
        except ImportError:
            return t

    @staticmethod
    def _payload_to_tensor(p: Any) -> Any:
        try:
            import torch  # type: ignore[import-untyped]  # TODO: Same as above - improve torch typing
            if p is None:
                return None
            if isinstance(p, dict) and "values" in p and "shape" in p and "dtype" in p:
                dtype = getattr(torch, p["dtype"].replace("torch.", ""))
                return torch.tensor(p["values"], dtype=dtype).reshape(p["shape"])
            return p
        except ImportError:
            return p

    # single-row
    def to_json(self) -> str:
        as_dict = {f.name: self._tensor_to_payload(getattr(self, f.name))
                   for f in fields(self)}
        return json.dumps(as_dict, ensure_ascii=False)

    @staticmethod
    def from_json(s: str) -> "TrajectorySample":
        data = json.loads(s)
        kwargs: Dict[str, Any] = {k: TrajectorySample._payload_to_tensor(v)
                                  for k, v in data.items()}
        return TrajectorySample(**kwargs)           # type: ignore[arg-type]

    # convenient wrappers for *.jsonl files ----------------------------------
    @staticmethod
    def to_jsonl(samples: List["TrajectorySample"]) -> str:
        return "\n".join(s.to_json() for s in samples)

    @staticmethod
    def from_jsonl(buf: str) -> List["TrajectorySample"]:
        return [TrajectorySample.from_json(line)
                for line in buf.strip().splitlines() if line]

    # disk I/O helpers -------------------------------------------------------
    @staticmethod
    def save_jsonl(samples: List["TrajectorySample"], path: str) -> None:
        from pathlib import Path
        Path(path).write_text(TrajectorySample.to_jsonl(samples), encoding="utf-8")

    @staticmethod
    def load_jsonl(path: str) -> List["TrajectorySample"]:
        from pathlib import Path
        return TrajectorySample.from_jsonl(Path(path).read_text(encoding="utf-8"))

@dataclass
class VLLMConfig:
    """VLLM server configuration"""
    model: str = "willcb/Qwen3-0.6B"
    max_model_len: int = 32678
    n_gpus: int = 1
    gpu_memory_utilization: float = 0.7
    enable_lora: bool = True
    enable_auto_tool_choice: bool = True
    tool_call_parser: str = "hermes"
    max_lora_rank: int = 32
    max_loras: int = 4
    enforce_eager: bool = False
    seed: int = 42
    port: int = 9999
    verbose: bool = verbose(5)

    @property
    def api_base(self):
        return f"http://localhost:{self.port}"
    
    @staticmethod
    def to_launch_command(config: 'VLLMConfig') -> List[str]:
        """Convert VLLMConfig to vllm launch command for subprocess."""
        tensor_parallel_size = config.n_gpus
        command = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--max-model-len", str(config.max_model_len),
            "--model", config.model,
            "--port", str(config.port),
            "--tensor-parallel-size", str(tensor_parallel_size),
            "--gpu-memory-utilization", str(config.gpu_memory_utilization),
            "--seed", str(config.seed),
            "--disable-custom-all-reduce",
        ]
        
        if config.enable_lora:
            command.extend([
                "--enable-lora",
                "--max-lora-rank", str(config.max_lora_rank),
                "--max-loras", str(config.max_loras),
            ])
        
        if config.enable_auto_tool_choice:
            command.extend([
                "--enable-auto-tool-choice",
                "--tool-call-parser", config.tool_call_parser,
            ])
        
        if config.enforce_eager:
            command.append("--enforce-eager")
        
        return command


# TODO: this class is way too load bearing, control flow should be outsourced
# to different component.
@dataclass
class RLConfig:
    experiment: str

    sft_epochs:     int = 1
    outer_epochs:   int = 1
    inner_epochs:   int = 1
    batch_epochs:   int = 1

    # Model & Optimizer -------------------------------------------------------
    base_model: str = "willcb/Qwen3-0.6B"
    old:str         = ""
    ref:str         = ""

    micro_batch_size:           int = 8
    grad_accumulation_steps:    int = 1
    ckpt_interval:              int = 1
    max_grad_norm:              float = 1.0
    clip_eps:                   float = 0.2
    kl_weight:                  float = 0.01
    learning_rate:              float = 1e-5

    # LoRA --------------------------------------------------------------------
    lora_r: int = 16
    lora_alpha: int = 32
    lora_target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj", "down_proj", "up_proj",
        ]
    )
    lora_dropout: float = 0.1
    lora_target_linear: bool = False
    lora_mlp_kernel: bool = True
    lora_qkv_kernel: bool = True

    # Resources ---------------------------------------------------------------
    vllm_gpus:      int = 1
    vllm_cpus:      int = 16
    trainer_gpus:   int = 1
    trainer_cpus:   int = 16

    vllm: VLLMConfig = field(default_factory=VLLMConfig) # base config

    # sampling ----------------------------------------------------------------
    n_groups = 4
    n_replicas = 4

    # Runtime bookkeeping -----------------------------------------------------
    seed: int = 42
    loop: int = 0

    # path roots (populated in `setup_paths`) ---------------------------------
    base_path: str              = "."
    loop_path: str              = ""
    trajectory_path: str        = ""
    trajectory_sample_path: str = ""
    checkpoint_path: str        = ""
    config_path: str            = ""
    results_path: str           = ""

    def __post_init__(self) -> None:
        # vllm
        self.vllm.model         = self.base_model
        self.vllm.seed          = self.seed
        self.vllm.max_lora_rank = self.lora_r

    # ------------------------- Path helpers ------------------------- #
    def setup(self) -> None:
        # model names
        self.old = self.experiment + "_old"
        self.ref = self.experiment + "_ref"

        self.base_path: str             = os.path.join(self.base_path, self.experiment)
        self.config_path: str           = os.path.join(self.base_path, f"config_{self.experiment}.json")
        self.results_path: str          = os.path.join(self.base_path, f"result_{self.experiment}.json")

        Path(self.base_path).mkdir(parents=True, exist_ok=True)
        self._update_paths_for_loop(0)

    def _update_paths_for_loop(self, loop_num: int) -> None:

        """Regenerate all per‑loop paths, ensuring directories exist."""
        self.loop = loop_num
        self.loop_path: str             = os.path.join(self.base_path, f"loop-{self.loop:04d}")
        self.trajectory_path: str       = os.path.join(self.loop_path, "trajectory.jsonl")
        self.trajectory_sample_path: str= os.path.join(self.loop_path, "trajectory_sample.jsonl")
        self.checkpoint_path: str       = os.path.join(self.loop_path, "checkpoints/")

        for p in (self.loop_path, self.checkpoint_path):
            Path(p).mkdir(parents=True, exist_ok=True)

    def save_json(self) -> None:
        Path(self.config_path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    def step(self) -> 'RLConfig':
        self._update_paths_for_loop(self.loop + 1)
        self.save_json()
        return self

@dataclass
class Results:
    rewards: List[float]        = field(default_factory=list)
    losses: List[float]         = field(default_factory=list)
    g_norms: List[float]        = field(default_factory=list)
    turns: List[float]          = field(default_factory=list)
    rollout_time: List[float]   = field(default_factory=list)  
    # gathering + parsing trajectories
    training_time: List[float]  = field(default_factory=list)  
    # train step + save checkpoint
    swap_time: List[float]      = field(default_factory=list)  
    # time it took to swap loras/checkpoints
    compute_ratio: List[float]  = field(default_factory=list)  
    # rollout_time / training_time, should be optimized to 1.0
    vllm_startup_time: float    = 0.0
    model_init_time: float      = 0.0

    # eval
    eval_rewards: List[float]   = field(default_factory=list)
    eval_turns: List[float]     = field(default_factory=list)
    config: Optional[
            RLConfig]    = None



@dataclass(frozen=True)
class ToolCall(SerialDataclass):
    id: str
    name: str
    args: Mapping[str, Any]

@dataclass(frozen=True)
class StreamChunk(SerialDataclass):
    """A chunk of data emitted during streaming"""
    kind: str  # "token", "tool_call_complete", "tool_result", etc.
    data: Mapping[str, Any]

@dataclass(frozen=True)
class Message(SerialDataclass):
    role: str
    content: Optional[str]
    reasoning_content: Optional[Any] = None
    thinking_content: Optional[str] = None
    thinking_signature: Optional[str] = None
    tool_calls: List[ToolCall] = field(default_factory=list)
    tool_call_id: Optional[str] = None

@dataclass(frozen=True)
class Usage(SerialDataclass):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: Optional[Any] = None

@dataclass(frozen=True)
class Logprob(SerialDataclass):
    token: str
    logprob: float
    bytes: List[int]
    top_logprobs: List[float]

@dataclass(frozen=True)
class Logprobs(SerialDataclass):
    content: List[Logprob] = field(default_factory=list)

@dataclass(frozen=True)
class Choice(SerialDataclass):
    index: int
    message: Message
    finish_reason: str
    logprobs: Optional[Logprobs] = None
    stop_reason: Optional[Any] = None


@dataclass(frozen=True)
class TokenInfo(SerialDataclass):
    logprob: float
    rank: int
    decoded_token: str

PromptLogprob = Optional[Dict[str, TokenInfo]]
"""
{
"8948": { # key is different every token
"logprob": -12.845086097717285,
"rank": 60822,
"decoded_token": "system"
}
}
"""

@dataclass(frozen=True)
class ChatCompletion(SerialDataclass):
    id: str
    object: str
    created: int
    model: str
    usage: Usage
    kv_transfer_params: Optional[Any] = None
    choices: List[Choice] = field(default_factory=list)
    prompt_logprobs: Optional[List[PromptLogprob]] = None

@dataclass
class Trajectory: # TODO: Port to serial
    completions: List[ChatCompletion] = field(default_factory=list)
    messages: List[Message] = field(default_factory=list)   # debugging only
    rewards: float = 0.0
    group: int = 0
    replica: int = 0
    advantages: float = 0.0     # scalar; broadcast later if needed

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Trajectory":
        """Rebuild nested dataclasses so type hints stay correct."""
        comps: List[ChatCompletion] = []
        for comp in data.get("completions", []):
            usage_dict = comp.get("usage", {})
            usage = Usage(
                prompt_tokens       = usage_dict["prompt_tokens"],
                completion_tokens   = usage_dict["completion_tokens"],
                total_tokens        = usage_dict["total_tokens"],
            )
            # Construct ChatCompletion with explicit parameters for type safety
            comps.append(ChatCompletion(
                id=comp.get("id", "unknown"),
                object=comp.get("object", "chat.completion"),
                created=comp.get("created", 0),
                model=comp.get("model", "unknown"),
                usage=usage,
                kv_transfer_params=comp.get("kv_transfer_params"),
                choices=comp.get("choices", []),
                prompt_logprobs=comp.get("prompt_logprobs")
            ))

        return Trajectory(
            completions=comps,
            messages=data.get("messages", []),
            rewards=data.get("rewards", 0.0),
            group=data.get("group", 0),
            replica=data.get("replica", 0),
            advantages=data.get("advantages", 0.0),
        )

    # ---------- JSONL convenience layer -----------------------------------
    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @staticmethod
    def to_jsonl(trajectories: List["Trajectory"]) -> str:
        return "\n".join(t.to_json() for t in trajectories)

    @staticmethod
    def from_json(json_str: str) -> "Trajectory":
        return Trajectory.from_dict(json.loads(json_str))

    @staticmethod
    def from_jsonl(jsonl_str: str) -> List["Trajectory"]:
        return [Trajectory.from_json(line) for line in jsonl_str.strip().splitlines() if line]

    # ---------- disk I/O ---------------------------------------------------
    @staticmethod
    def save_jsonl(trajectories: List["Trajectory"], filepath: str) -> None:
        Path(filepath).write_text(Trajectory.to_jsonl(trajectories), encoding="utf-8")

    @staticmethod
    def load_jsonl(filepath: str) -> List["Trajectory"]:
        return Trajectory.from_jsonl(Path(filepath).read_text(encoding="utf-8"))

    @staticmethod
    def load_jsonl_streaming(filepath: str) -> Iterator["Trajectory"]:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    yield Trajectory.from_json(line)

    # ---------- helpers that work pre-/post-serialisation ------------------
    @staticmethod
    def _usage_total(usage: Union[Usage, Dict[str, Any]], key: str) -> int:
        if isinstance(usage, Usage):
            return getattr(usage, key, 0)
        return usage.get(key, 0)

    @staticmethod
    def get_completion_tokens(traj: "Trajectory") -> int:
        return sum(Trajectory._usage_total(c.usage, "completion_tokens") for c in traj.completions)

    @staticmethod
    def get_total_tokens(traj: "Trajectory") -> int:
        return sum(Trajectory._usage_total(c.usage, "total_tokens") for c in traj.completions[-1:])

    @staticmethod
    def hash(trajectory: "Trajectory") -> str:
        import hashlib
        """Generate a hash for a single trajectory."""
        traj_dict = asdict(trajectory)
        traj_str = json.dumps(traj_dict, sort_keys=True)
        return hashlib.sha256(traj_str.encode()).hexdigest()[:16]

@dataclass(frozen=True)
class ToolFunctionParameter(SerialDataclass):
    properties: Dict[str, Any]
    type: str = "object"

@dataclass(frozen=True)
class ToolFunction(SerialDataclass):
    name: str
    description: str
    parameters: ToolFunctionParameter
    required: List[str] = field(default_factory=list)

@dataclass(frozen=True)
class Tool(SerialDataclass):
    function: ToolFunction
    type: str = "function"

class StopReason(Enum):
    MAX_TURNS = "MAX_TURNS"
    TOOL_ERROR = "TOOL_ERROR"
    USER_ABORT = "USER_ABORT"
    PROVIDER_ERROR = "PROVIDER_ERROR"
    NO_TOOL_CALLED = "NO_TOOL_CALLED"
    TASK_COMPLETED = "TASK_COMPLETED"

@dataclass(frozen=True)
class ToolResult(SerialDataclass):
    call_id: str = ""
    ok: bool = False
    content: str = ""
    error: Optional[str] = None
    stop_reason: Optional[StopReason] = None

@dataclass(frozen=True)
class ToolConfirmResult(SerialDataclass):
    """Result of tool confirmation"""
    proceed: bool
    tool_result: Optional[ToolResult] = None
    user_message: Optional[str] = None

# ── Core Agent Framework Types ────────────────────────────────────────────────

class Environment(ABC):
    """Base class for environments managing external resources."""
    
    def get_tools(self) -> List[Tool]:
        return []
    
    async def exec_tool(self, tool_call: ToolCall, current_state: 'AgentState',
                       run_config: 'RunConfig', checkpoint_store = None) -> ToolResult:
        return ToolResult()
    
    def requires_confirmation(self, tool_call: ToolCall) -> bool:
        """By default, no tool requires confirmation."""
        return False

    async def serialize(self) -> dict:
        return {}
    
    @staticmethod
    async def deserialize(data: dict) -> 'Environment':
        raise NotImplementedError

@dataclass(frozen=True)
class Endpoint(SerialDataclass):
    provider: str
    model: str
    api_base: str = ""
    api_key: str = ""
    max_tokens: int = 8192
    temperature: float = 1.0
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None  
    parallel_tool_calls: Optional[bool] = None  
    reasoning_effort: Optional[str] = None  # for openai
    max_completion_tokens: Optional[int] = None  # for openai
    thinking: Optional[Dict[str, Any]] = None # for anthropic 
    # Extra params merged into the raw chat request for custom servers
    extra_params: Optional[Dict[str, Any]] = None

@dataclass(frozen=True)
class Actor(SerialDataclass):
    trajectory: Trajectory
    endpoint: Endpoint
    tools: List[Tool] = field(default_factory=list)

@dataclass(frozen=True)
class AgentState:
    actor: Actor
    environment: Environment
    max_turns: int
    stop: Optional[StopReason] = None
    turn_idx: int = 0
    pending_tool_calls: List[ToolCall] = field(default_factory=list)
    next_tool_idx: int = 0  # Which tool we're about to process
    timestamp: str = datetime.now(timezone.utc).isoformat() + 'Z'

# Forward declarations for RunConfig (needs to be after AgentState but before default handlers)
async def default_stdin_handler(prompt: str) -> str:
    """Default input handler using asyncio.to_thread for non-blocking input."""
    return await asyncio.to_thread(input, prompt)

async def default_confirm_tool(tc: ToolCall, state: 'AgentState', run_config: 'RunConfig') -> Tuple['AgentState', ToolConfirmResult]:
    """Default tool confirmation handler - auto-confirm all tools."""
    return state, ToolConfirmResult(proceed=True)

async def default_no_tool_handler(state: 'AgentState', run_config: 'RunConfig') -> 'AgentState':
    """Default no-tool handler - do nothing."""
    return state

@dataclass(frozen=True)
class RunConfig:
    # TODO: Add runtime validation for on_chunk parameter to catch sync functions early
    # Currently if a sync function is passed, it gets set to None silently, causing
    # "object NoneType can't be used in 'await' expression" errors later. Should validate
    # that on_chunk is properly async and has correct signature at construction time.
    on_chunk: Callable[[StreamChunk], Awaitable[None]]
    on_input: Callable[[str], Awaitable[str]] = field(default_factory=lambda: default_stdin_handler)
    confirm_tool: Callable[[ToolCall, 'AgentState', 'RunConfig'], Awaitable[Tuple['AgentState', ToolConfirmResult]]] = field(default_factory=lambda: default_confirm_tool)
    handle_tool_error: Callable[[ToolResult, 'AgentState'], 'AgentState'] = lambda tr, s: s
    on_step_start: Callable[['AgentState'], 'AgentState'] = lambda s: s
    handle_stop: Callable[['AgentState'], 'AgentState'] = lambda s: s
    handle_no_tool: Callable[['AgentState', 'RunConfig'], Awaitable['AgentState']] = field(default_factory=lambda: default_no_tool_handler)
    user_message_for_thinking: Optional[str] = None
    inline_thinking: Optional[str] = None
    checkpoint_store: Optional[Any] = None
