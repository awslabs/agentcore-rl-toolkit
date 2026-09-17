"""A rollout session that drives the agent over the four-POST HTTP protocol.

The agent runs as an HTTP server in a Bedrock AgentCore runtime session, and this session
talks to it directly in :mod:`.wire`'s protocol -- setup, status polls, start, dump -- so
completion and the trajectory arrive on an invoke response. The sibling
:mod:`.agentcore_s3_session` reaches the same runtime with one fire-and-forget invoke and
polls S3 for the result instead; both are registered in :mod:`.factory` (``agentcore_http``
and ``agentcore_s3``).
"""

import json
import logging
import os
from typing import Any

import backoff

from agentcore_rl_toolkit.aws_tools.agentcore_tools import (
    invoke_agentcore_session,
    region_of,
    shared_agentcore_client,
    start_agentcore_session,
    stop_agentcore_instance_session,
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


class AgentCoreHttpSession(RolloutSession):
    """A rollout session backed by an HTTP server in a Bedrock AgentCore runtime session.

    Every call it makes -- start, setup, each status poll, the dump, the delete -- rides on
    the process-wide ``bedrock-agentcore`` client (:func:`shared_agentcore_client`), because
    sessions are constructed one per trajectory and a client per session would be a
    connection pool per rollout. Nothing here owns that client's lifetime, so every phase
    works whether or not the session was entered, and ``shutdown`` stays safe to call on a
    session that never ran.
    """

    def __init__(
        self,
        session_id,
        session_state: PersistentDict,
        runtime_arn: str,
        capacity_provider_arn: str,
    ):
        # One client serves the runtime invokes and the capacity-provider delete, so both
        # arns must name the same region -- they always do: the provider hosts the
        # runtime's sessions. Checked here because nothing later reads the provider's region.
        self.region = region_of(runtime_arn)
        assert region_of(capacity_provider_arn) == self.region, "runtime and capacity provider are in two regions"
        self.session_id = session_id
        self.session_state = session_state
        self.runtime_arn = runtime_arn
        self.capacity_provider_arn = capacity_provider_arn

    async def __aenter__(self) -> "AgentCoreHttpSession":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        # Nothing is suppressed and no exception is juggled: a failing shutdown propagates
        # with the body's exception as its `__context__` (implicit chaining), so the error
        # that ended the rollout stays reachable underneath a teardown error --
        # `describe_with_root_cause` is what digs it out.
        await self.shutdown()

    async def _acr(self) -> Any:
        """The shared client this session's calls ride on."""
        return await shared_agentcore_client(self.region)

    async def setup(self, task: dict) -> None:
        await self.session_state.update(
            {
                "capacity_provider_arn": self.capacity_provider_arn,
                "runtime_arn": self.runtime_arn,
            }
        )
        client = await self._acr()

        async with measure_span_persistent("agentcore_setup", self.session_state):
            await start_agentcore_session(self.runtime_arn, self.session_id, client)

        async with measure_span_persistent("task_setup", self.session_state):
            await start_and_wait_setup(self.runtime_arn, self.session_id, task, client)

    async def run(self, task: dict) -> RolloutDumpResponse:
        return await start_and_wait_rollout(self.runtime_arn, self.session_id, task, await self._acr())

    async def shutdown(self) -> None:
        await stop_agentcore_instance_session(self.capacity_provider_arn, self.session_id, await self._acr())


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
