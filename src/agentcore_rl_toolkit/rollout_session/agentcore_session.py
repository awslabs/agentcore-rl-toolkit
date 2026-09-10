"""A rollout session that runs the agent in a Bedrock AgentCore runtime session."""

import json
import logging
import os
from contextlib import AsyncExitStack
from typing import Any

import backoff

from agentcore_rl_toolkit.aws_tools.agentcore_tools import (
    agentcore_client,
    invoke_agentcore_session,
    region_of,
    start_agentcore_session,
    stop_agentcore_session,
)
from agentcore_rl_toolkit.aws_tools.persistent_dict import PersistentDict, measure_span_persistent
from agentcore_rl_toolkit.rollout_session.lifecycle import RolloutSession
from agentcore_rl_toolkit.rollout_session.wire import (
    InvocationInput,
    InvocationOutput,
    InvocationRequest,
    InvocationResponse,
    RolloutDumpRequest,
    RolloutDumpResponse,
    RolloutSetupRequest,
    RolloutStartRequest,
    RolloutStatusRequest,
    RolloutStatusResponse,
)

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

logging.getLogger("backoff").setLevel(logging.ERROR)


class AgentCoreSession(RolloutSession):
    """A rollout session backed by an HTTP server in a Bedrock AgentCore runtime session.

    Entering the session opens one ``bedrock-agentcore`` client and every call it makes
    -- start, setup, each status poll, the dump, the delete -- rides on that one client;
    see :mod:`agentcore_rl_toolkit.aws_tools.agentcore_tools` for why. Outside that scope
    the phases still work, each opening a client for itself, so ``shutdown`` remains safe
    to call on a session that was never entered or is already closed.
    """

    def __init__(
        self,
        session_id,
        session_state: PersistentDict,
        runtime_arn: str,
        capacity_provider_arn: str,
    ):
        self.session_id = session_id
        self.session_state = session_state
        self.runtime_arn = runtime_arn
        self.capacity_provider_arn = capacity_provider_arn
        # Both live only inside `async with self`; `None` means "no shared client", which
        # is what makes the phases fall back to a client per call.
        self._client: Any | None = None
        self._scope: AsyncExitStack | None = None

    async def __aenter__(self) -> "AgentCoreSession":
        # One client serves the runtime invokes and the capacity-provider delete, so both
        # arns must name the same region -- they always do: the provider hosts the
        # runtime's sessions.
        region = region_of(self.runtime_arn)
        assert region_of(self.capacity_provider_arn) == region, "runtime and capacity provider are in two regions"
        scope = AsyncExitStack()
        self._client = await scope.enter_async_context(agentcore_client(region))
        # Pushed after the client, so it unwinds first: the delete goes over the shared
        # client, which is closed only once shutdown has returned.
        scope.push_async_callback(self.shutdown)
        self._scope = scope
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        scope, self._scope = self._scope, None
        if scope is None:
            # Never entered (or already exited): still honour the teardown contract.
            await self.shutdown()
            return
        try:
            await scope.aclose()
        finally:
            self._client = None

    async def setup(self, task: dict) -> None:
        await self.session_state.update(
            {
                "capacity_provider_arn": self.capacity_provider_arn,
                "runtime_arn": self.runtime_arn,
            }
        )

        async with measure_span_persistent("agentcore_setup", self.session_state):
            await start_agentcore_session(self.runtime_arn, self.session_id, self._client)

        async with measure_span_persistent("task_setup", self.session_state):
            await start_and_wait_setup(self.runtime_arn, self.session_id, task, self._client)

    async def run(self, task: dict) -> RolloutDumpResponse:
        return await start_and_wait_rollout(self.runtime_arn, self.session_id, task, self._client)

    async def shutdown(self) -> None:
        await stop_agentcore_session(self.capacity_provider_arn, self.session_id, self._client)


async def invoke_agent(
    runtime_arn: str, session_id: str, payload: InvocationInput, client: Any | None = None
) -> InvocationOutput:
    json_body = json.dumps(InvocationRequest(payload=payload).model_dump(mode="json"))
    response = await invoke_agentcore_session(runtime_arn, session_id, json_body.encode(), client)
    assert (
        response.status_code == 200
    ), f"Received {response.status_code=} for {session_id=} request {json_body} and response: {response.body}"
    return InvocationResponse.model_validate_json(response.body).payload


@backoff.on_predicate(backoff.constant, lambda x: not x.done, interval=10)
async def wait_status_done(runtime_arn: str, session_id: str, client: Any | None = None) -> RolloutStatusResponse:
    resp = await invoke_agent(runtime_arn, session_id, RolloutStatusRequest(), client)
    assert isinstance(resp, RolloutStatusResponse), f"Unexpected response type {resp}"
    return resp


async def start_and_wait_setup(runtime_arn: str, session_id: str, task: dict, client: Any | None = None):
    await invoke_agent(runtime_arn, session_id, RolloutSetupRequest(task_input=task), client)
    status = await wait_status_done(runtime_arn, session_id, client)
    assert status.exception is None, f"Setup {session_id=} returned exception: {status.exception}"


async def start_and_wait_rollout(
    runtime_arn: str, session_id: str, task: dict, client: Any | None = None
) -> RolloutDumpResponse:
    rollout_start = RolloutStartRequest(
        rollout_id=session_id,
        task_input=task,
    )
    await invoke_agent(runtime_arn, session_id, rollout_start, client)
    status = await wait_status_done(runtime_arn, session_id, client)
    assert status.exception is None, f"Rollout {session_id=} error: {status.exception}"

    rollout = await invoke_agent(runtime_arn, session_id, RolloutDumpRequest(), client)
    assert isinstance(rollout, RolloutDumpResponse)
    return rollout
