"""Shared fixtures for the experimental verl backend tests.

These tests run against the installed verl distribution (the pinned
``verl-experimental`` extra): ``AgentLoopBase.__init__`` (system-prompt /
turn-separator probing), pydantic ``AgentLoopOutput``/``TokenOutput``
validation, ``DictConfigWrap``/OmegaConf config plumbing, and hydra
instantiation are all verl's own. The whole directory is skipped when verl is
not installed (``uv sync --extra dev`` alone), and runs in the dedicated
verl-integration CI job.

Only two seams are faked, deliberately:

- ``FakeLLMServerClient`` — verl's ``LLMServerClient`` fronts Ray actors and a
  live inference engine; ``generate`` (token-in/token-out) is the whole
  contract ``VerlSamplingBackend`` consumes. The fake returns verl
  ``TokenOutput`` models so response-shape drift is still caught.
- ``FakeTokenizer`` — a deterministic tokenizer so token-level assertions
  (loss masks, prefix extension, truncation) are exact. HF-tokenizer coverage
  lives in ``tests/rollout_gateway/`` and ``test_payload_dataset.py``.
"""

from typing import Any, Optional

import pytest

verl = pytest.importorskip("verl", reason="requires the verl-experimental extra")

from omegaconf import OmegaConf  # noqa: E402
from verl.experimental.agent_loop.agent_loop import DictConfigWrap  # noqa: E402
from verl.workers.rollout.replica import TokenOutput  # noqa: E402

# -- config helpers -------------------------------------------------------------


def make_trainer_config(*, use_v1: bool = True, prompt_length: int = 64, response_length: int = 32) -> DictConfigWrap:
    """The slice of verl's trainer config AgentLoopBase + AgentCoreAgentLoop read."""
    return DictConfigWrap(
        OmegaConf.create(
            {
                "trainer": {"use_v1": use_v1},
                "actor_rollout_ref": {
                    "model": {"path": "test/model"},
                    "rollout": {"prompt_length": prompt_length, "response_length": response_length},
                },
            }
        )
    )


def make_data_config() -> DictConfigWrap:
    """The slice of verl's data config AgentLoopBase reads
    (``continuous_token`` is accessed unconditionally in its ``__init__``)."""
    return DictConfigWrap(OmegaConf.create({"continuous_token": {"enable": False}}))


# -- fakes ----------------------------------------------------------------------


class FakeLLMServerClient:
    """Duck-typed ``LLMServerClient``: scripted per-call responses as verl
    ``TokenOutput`` models; defaults to echoing two tokens."""

    def __init__(self, outputs: Optional[list[TokenOutput]] = None):
        self.outputs = list(outputs) if outputs else None
        self.calls: list[dict[str, Any]] = []

    async def generate(self, request_id, *, prompt_ids, sampling_params, **kwargs):
        self.calls.append(
            {"request_id": request_id, "prompt_ids": list(prompt_ids), "sampling_params": sampling_params, **kwargs}
        )
        if self.outputs:
            return self.outputs.pop(0)
        return TokenOutput(token_ids=[101, 102], log_probs=[-0.5, -0.6], stop_reason="completed")


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
