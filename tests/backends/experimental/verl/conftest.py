"""Shared fakes for the experimental verl backend tests.

verl itself is a heavyweight optional dependency (torch, ray, vllm), so these
tests stub the small surface `agent_loop.py` imports from
``verl.experimental.agent_loop.agent_loop`` (AgentLoopBase/AgentLoopOutput/
AgentLoopMetrics/register). Everything else — the gateway, adapters, trajectory
manager, sampling backend — is exercised for real.
"""

import sys
import types
from dataclasses import dataclass, field
from typing import Any, Optional

import pytest

# -- verl stub (installed into sys.modules before agent_loop import) -----------


def _install_verl_stub() -> None:
    if "verl.experimental.agent_loop.agent_loop" in sys.modules:
        return

    mod = types.ModuleType("verl.experimental.agent_loop.agent_loop")

    class AgentLoopMetrics:
        def __init__(self, generate_sequences: float = 0.0, tool_calls: float = 0.0, **kw):
            self.generate_sequences = generate_sequences
            self.tool_calls = tool_calls

    class AgentLoopOutput:
        """Field-compatible stand-in for verl's pydantic AgentLoopOutput."""

        def __init__(
            self,
            *,
            prompt_ids,
            response_ids,
            response_mask,
            response_logprobs=None,
            reward_score=None,
            num_turns=0,
            metrics=None,
            extra_fields=None,
            **kw,
        ):
            self.prompt_ids = prompt_ids
            self.response_ids = response_ids
            self.response_mask = response_mask
            self.response_logprobs = response_logprobs
            self.reward_score = reward_score
            self.num_turns = num_turns
            self.metrics = metrics
            self.extra_fields = extra_fields or {}

    class AgentLoopBase:
        def __init__(self, trainer_config, server_manager, tokenizer, processor, dataset_cls, data_config, **kwargs):
            self.config = trainer_config.config
            self.rollout_config = self.config.actor_rollout_ref.rollout
            self.server_manager = server_manager
            self.tokenizer = tokenizer
            self.processor = processor
            self.dataset_cls = dataset_cls
            self.data_config = data_config

    registry: dict[str, type] = {}

    def register(name):
        def deco(cls):
            registry[name] = cls
            return cls

        return deco

    mod.AgentLoopMetrics = AgentLoopMetrics
    mod.AgentLoopOutput = AgentLoopOutput
    mod.AgentLoopBase = AgentLoopBase
    mod.register = register
    mod._agent_loop_registry = registry

    # parent packages
    for pkg_name in ("verl", "verl.experimental", "verl.experimental.agent_loop"):
        if pkg_name not in sys.modules:
            pkg = types.ModuleType(pkg_name)
            pkg.__path__ = []  # mark as package
            sys.modules[pkg_name] = pkg
    sys.modules["verl.experimental.agent_loop.agent_loop"] = mod
    sys.modules["verl.experimental.agent_loop"].agent_loop = mod


_install_verl_stub()


# -- config helpers -------------------------------------------------------------


class AttrDict(dict):
    """dict with attribute access + .get, standing in for OmegaConf DictConfig."""

    __getattr__ = dict.__getitem__

    def __setattr__(self, k, v):
        self[k] = v


def make_trainer_config(*, use_v1: bool = True, prompt_length: int = 64, response_length: int = 32):
    config = AttrDict(
        trainer=AttrDict(use_v1=use_v1),
        actor_rollout_ref=AttrDict(
            model=AttrDict(path="test/model"),
            rollout=AttrDict(prompt_length=prompt_length, response_length=response_length),
        ),
    )
    wrap = types.SimpleNamespace(config=config)
    return wrap


# -- fakes shared with the sampling-backend/gateway tests -----------------------


@dataclass
class FakeTokenOutput:
    token_ids: list[int]
    log_probs: Optional[list[float]] = None
    stop_reason: Optional[str] = None
    extra_fields: dict[str, Any] = field(default_factory=dict)


class FakeLLMServerClient:
    """Scripted per-call responses; defaults to echoing two tokens."""

    def __init__(self, outputs: Optional[list[FakeTokenOutput]] = None):
        self.outputs = outputs
        self.calls: list[dict[str, Any]] = []

    async def generate(self, request_id, *, prompt_ids, sampling_params, **kwargs):
        self.calls.append(
            {"request_id": request_id, "prompt_ids": list(prompt_ids), "sampling_params": sampling_params}
        )
        if self.outputs:
            return self.outputs.pop(0)
        return FakeTokenOutput(token_ids=[101, 102], log_probs=[-0.5, -0.6], stop_reason="completed")


class FakeTokenizer:
    """Minimal HF-tokenizer stand-in whose renders prefix-extend across turns
    (like a real chat template): an assistant message re-renders as the
    generation prompt (99) + the exact ids the fake backend generated
    (101, 102), so replayed conversations take the CLEAN merge path."""

    eos_token = "</s>"
    eos_token_id = 0
    pad_token_id = 0

    def apply_chat_template(self, messages, *, tools=None, tokenize=True, add_generation_prompt=True, **kw):
        ids = []
        for i, m in enumerate(messages):
            if m.get("role") == "assistant":
                ids += [99, 101, 102]
            else:
                ids += [10 + i, 20 + i]
        if add_generation_prompt:
            ids.append(99)
        return ids

    def decode(self, ids, skip_special_tokens=False):
        return "hello world"

    def convert_tokens_to_ids(self, token):
        return None


@pytest.fixture(autouse=True)
def reset_singletons():
    yield
    from agentcore_rl_toolkit.backends.experimental.verl import agent_loop, gateway_host

    agent_loop._reset_client_for_tests()
    gateway_host._reset_for_tests()
