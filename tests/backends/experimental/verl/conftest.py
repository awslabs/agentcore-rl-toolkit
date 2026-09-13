"""Shared fixtures for the ``RolloutSessionAgentLoop`` tests.

The loop is built through verl's own ``AgentLoopBase`` against a *live* (threaded) rollout
gateway, the way an ``AgentLoopWorker`` process builds it: verl's config plumbing, hydra
instantiation of the loop's config node, and pydantic ``AgentLoopOutput`` validation are all
real, so contract drift on either side is caught here. The whole directory is skipped when
verl is absent (``uv sync --extra dev`` alone) and runs in the verl-integration CI job.

Four seams are faked, deliberately:

- ``FakeLLMServerClient`` / ``FakeTokenizer`` (shared with ``tests/backends/verl``) --
  token-in/token-out generation and a deterministic tokenizer, so token-level assertions
  are exact.
- :class:`FakeRolloutSession` -- the container is this loop's *dependency*, not its subject.
  It drives real chat turns against the loop's own gateway (as a deployed agent would),
  then returns the ``RolloutDumpResponse`` the test asked for, or raises.
- ``get_rollout_session_bounds`` -- the real one looks up named Ray actors; the bounds are
  rebuilt here from the same config with their process-local implementations.
- ``upload_object`` -- the per-rollout S3 dump, the loop's only AWS call once
  ``dynamodb_table`` is null (which makes the session record a ``NullPersister``).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import aiohttp
import pytest

verl = pytest.importorskip("verl", reason="requires the verl extra")

from omegaconf import OmegaConf  # noqa: E402
from verl.experimental.agent_loop.agent_loop import DictConfigWrap  # noqa: E402

from agentcore_rl_toolkit.backends.experimental.verl import rollout_session_agent_loop as rsal  # noqa: E402
from agentcore_rl_toolkit.backends.experimental.verl.rollout_session_agent_loop import (  # noqa: E402
    RolloutSessionAgentLoop,
)
from agentcore_rl_toolkit.concurrency.priority_assigner import LocalPriorityAssigner  # noqa: E402
from agentcore_rl_toolkit.concurrency.priority_semaphore import LocalPrioritySemaphore  # noqa: E402
from agentcore_rl_toolkit.rollout_session.lifecycle import RolloutSessionBounds  # noqa: E402
from agentcore_rl_toolkit.rollout_session.wire import RolloutDumpResponse  # noqa: E402

from ...verl.conftest import FakeLLMServerClient, FakeTokenizer  # noqa: E402,F401

LOOP_CONFIG_TARGET = "agentcore_rl_toolkit.backends.experimental.verl.config.RolloutSessionAgentLoopConfig"

# -- config helpers -------------------------------------------------------------


def make_loop_config_node(**overrides: Any) -> dict[str, Any]:
    """The ``rollout_session_agent_loop`` yaml node, as hydra hands it to the loop."""
    node: dict[str, Any] = {
        "_target_": LOOP_CONFIG_TARGET,
        # bind and advertise loopback: the "container" is an aiohttp client in-process
        "rollout_gateway": {"host": "127.0.0.1", "public_host": "127.0.0.1"},
        "rollout_session_bounds": {
            "container_concurrency": 4,
            "rollout_concurrency": 4,
            "session_create_rate": 100.0,
            "container_setup_timeout": 30.0,
            "agent_run_timeout": 30.0,
        },
        "rollout_session_backend": {"backend": "fake"},
        "rollout_gateway_sampling_params": {"max_new_tokens": 8},
        "aws_region": "us-west-2",
        "dynamodb_table": None,
        "rollout_output_s3": "s3://test-bucket/rollouts",
        "task_kwargs": {},
    }
    node.update(overrides)
    return node


def make_trainer_config(
    *,
    use_v1: bool = True,
    prompt_length: int = 64,
    response_length: int = 32,
    max_model_len: int | None = 128,
    **loop_overrides: Any,
) -> DictConfigWrap:
    """The slice of verl's trainer config ``AgentLoopBase`` + this loop read."""
    return DictConfigWrap(
        OmegaConf.create(
            {
                "trainer": {
                    "use_v1": use_v1,
                    "experiment_name": "test-experiment",
                    "experiment_start_at": "2026-01-01T00:00:00",
                },
                "actor_rollout_ref": {
                    "model": {"path": "test/model"},
                    "rollout": {
                        "prompt_length": prompt_length,
                        "response_length": response_length,
                        "max_model_len": max_model_len,
                    },
                },
                "rollout_session_agent_loop": make_loop_config_node(**loop_overrides),
            }
        )
    )


def make_data_config(apply_chat_template_kwargs: dict | None = None) -> DictConfigWrap:
    """The slice of verl's data config ``AgentLoopBase`` reads."""
    return DictConfigWrap(
        OmegaConf.create(
            {
                "continuous_token": {"enable": False},
                "apply_chat_template_kwargs": apply_chat_template_kwargs or {},
            }
        )
    )


def local_bounds(bounds_cfg: dict[str, Any]) -> RolloutSessionBounds:
    """The configured bounds, backed by process-local objects instead of Ray actors."""
    return RolloutSessionBounds(
        container_semaphore=LocalPrioritySemaphore(bounds_cfg["container_concurrency"]),
        rollout_semaphore=LocalPrioritySemaphore(bounds_cfg["rollout_concurrency"]),
        container_priority_assigner=LocalPriorityAssigner(),
        rollout_priority_assigner=LocalPriorityAssigner(),
        container_setup_timeout=bounds_cfg["container_setup_timeout"],
        agent_run_timeout=bounds_cfg["agent_run_timeout"],
        # The rate limiter is a cluster-wide throttle on container creation; nothing to
        # throttle here, and None is a supported configuration.
        session_rate_limiter=None,
    )


# -- fakes ----------------------------------------------------------------------


#: distinguishes "left at the default" from an explicit ``task_output=None`` (a dump that
#: reports no result, which is a failed rollout)
_DEFAULT = object()


def make_dump(
    *,
    reward: float | None = 1.0,
    metrics: dict[str, float] | None = None,
    task_output: Any = _DEFAULT,
    exception: str | None = None,
) -> RolloutDumpResponse:
    """A dump as the container's ``/invocations`` would return it."""
    return RolloutDumpResponse(
        metrics=metrics or {},
        task_output={"answer": "4"} if task_output is _DEFAULT else task_output,
        reward=reward,
        exception=exception,
    )


async def drive_gateway_turns(
    task: dict,
    *,
    turns: int = 1,
    session_key: str | None = None,
) -> None:
    """Chat against the gateway named in ``task["llm"]``, as the deployed agent would.

    ``session_key`` overrides the Bearer token, standing in for an agent that sends a
    fixed api key instead of the one the task carries.
    """
    llm = task["llm"]
    sid = session_key if session_key is not None else llm["api_key"]
    messages: list[dict[str, Any]] = [{"role": "user", "content": "hi"}]

    async with aiohttp.ClientSession() as http:
        for _ in range(turns):
            response = await http.post(
                f"{llm['base_url']}/chat/completions",
                json={"model": llm["model"], "messages": messages},
                headers={"Authorization": f"Bearer {sid}"},
            )
            assert response.status == 200, await response.text()
            body = await response.json()
            messages = [*messages, body["choices"][0]["message"], {"role": "user", "content": "more"}]


class FakeRolloutSession:
    """A ``RolloutSession`` that drives the gateway, then reports what the test asked for.

    ``error`` raises out of ``run`` (a failure on the trainer's side of the wire: transport,
    timeout, a contract violation); ``dump`` reports one from inside the container.
    """

    def __init__(
        self,
        dump: RolloutDumpResponse | None = None,
        *,
        turns: int = 1,
        error: BaseException | None = None,
        setup_error: BaseException | None = None,
        session_key: str | None = None,
    ):
        self.dump = dump if dump is not None else make_dump()
        self.turns = turns
        self.error = error
        self.setup_error = setup_error
        self.session_key = session_key
        self.events: list[str] = []
        self.tasks: list[dict] = []

    async def setup(self, task: dict) -> None:
        self.events.append("setup")
        if self.setup_error is not None:
            raise self.setup_error

    async def run(self, task: dict) -> RolloutDumpResponse:
        self.events.append("run")
        self.tasks.append(task)
        await drive_gateway_turns(task, turns=self.turns, session_key=self.session_key)
        if self.error is not None:
            raise self.error
        return self.dump

    async def shutdown(self) -> None:
        self.events.append("shutdown")

    async def __aenter__(self) -> "FakeRolloutSession":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.shutdown()


def make_loop(
    *,
    llm: Any = None,
    tokenizer: Any = None,
    session: Any = None,
    trainer_config: DictConfigWrap | None = None,
    data_config: DictConfigWrap | None = None,
    **config_overrides: Any,
) -> RolloutSessionAgentLoop:
    """Build the loop as verl would, with the session and the bounds actors faked.

    The session it ran with is available as ``loop.test_session``.
    """
    session = FakeRolloutSession() if session is None else session
    with (
        patch.object(rsal, "make_session", return_value=session),
        patch.object(rsal, "get_rollout_session_bounds", side_effect=local_bounds),
    ):
        loop = RolloutSessionAgentLoop(
            trainer_config or make_trainer_config(**config_overrides),
            llm or FakeLLMServerClient(),
            tokenizer or FakeTokenizer(),
            None,
            None,
            data_config or make_data_config(),
            name="rollout_session_agent_loop",  # hydra passes the YAML entry's name through
        )
    loop.test_session = session  # type: ignore[attr-defined]
    return loop


# -- fixtures -------------------------------------------------------------------


@pytest.fixture(autouse=True)
def fake_upload():
    """Stub the per-rollout S3 dump; yields the mock so tests can assert on it."""
    with patch.object(rsal, "upload_object", new=AsyncMock()) as upload:
        yield upload


@pytest.fixture(autouse=True)
def reset_gateway():
    yield
    from agentcore_rl_toolkit.backends.verl import gateway_host

    gateway_host._reset_for_tests()
