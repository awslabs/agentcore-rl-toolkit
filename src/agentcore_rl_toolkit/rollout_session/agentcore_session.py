"""A rollout session that runs the agent in a Bedrock AgentCore runtime session."""

import json
import logging
import os

import backoff

from agentcore_rl_toolkit.aws_tools.agentcore_tools import (
    invoke_agentcore_session,
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
    """A rollout session backed by an HTTP server in a Bedrock AgentCore runtime session."""

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

    async def __aenter__(self) -> "AgentCoreSession":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.shutdown()

    async def setup(self, task: dict) -> None:
        await self.session_state.update(
            {
                "capacity_provider_arn": self.capacity_provider_arn,
                "runtime_arn": self.runtime_arn,
            }
        )

        async with measure_span_persistent("agentcore_setup", self.session_state):
            await start_agentcore_session(self.runtime_arn, self.session_id)

        async with measure_span_persistent("task_setup", self.session_state):
            await start_and_wait_setup(self.runtime_arn, self.session_id, task)

    async def run(self, task: dict) -> RolloutDumpResponse:
        return await start_and_wait_rollout(self.runtime_arn, self.session_id, task)

    async def shutdown(self) -> None:
        await stop_agentcore_session(self.capacity_provider_arn, self.session_id)


async def invoke_agent(runtime_arn: str, session_id: str, payload: InvocationInput) -> InvocationOutput:
    json_body = json.dumps(InvocationRequest(payload=payload).model_dump(mode="json"))
    response = await invoke_agentcore_session(runtime_arn, session_id, json_body.encode())
    assert (
        response.status_code == 200
    ), f"Received {response.status_code=} for {session_id=} request {json_body} and response: {response.body}"
    return InvocationResponse.model_validate_json(response.body).payload


@backoff.on_predicate(backoff.constant, lambda x: not x.done, interval=10)
async def wait_status_done(runtime_arn: str, session_id: str) -> RolloutStatusResponse:
    resp = await invoke_agent(runtime_arn, session_id, RolloutStatusRequest())
    assert isinstance(resp, RolloutStatusResponse), f"Unexpected response type {resp}"
    return resp


async def start_and_wait_setup(runtime_arn: str, session_id: str, task: dict):
    await invoke_agent(runtime_arn, session_id, RolloutSetupRequest(task_input=task))
    status = await wait_status_done(runtime_arn, session_id)
    assert status.exception is None, f"Setup {session_id=} returned exception: {status.exception}"


async def start_and_wait_rollout(runtime_arn: str, session_id: str, task: dict) -> RolloutDumpResponse:
    rollout_start = RolloutStartRequest(
        rollout_id=session_id,
        task_input=task,
    )
    await invoke_agent(runtime_arn, session_id, rollout_start)
    status = await wait_status_done(runtime_arn, session_id)
    assert status.exception is None, f"Rollout {session_id=} error: {status.exception}"

    rollout = await invoke_agent(runtime_arn, session_id, RolloutDumpRequest())
    assert isinstance(rollout, RolloutDumpResponse)
    return rollout
