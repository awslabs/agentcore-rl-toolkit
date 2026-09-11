"""AgentCore data plane: starting, invoking and stopping one session, once per rollout.
"""

from contextlib import asynccontextmanager
from typing import Any

from botocore.config import Config
from pydantic import BaseModel

from agentcore_rl_toolkit.aws_tools.boto3_tools import get_aioboto3_session

SESSION_CLIENT_CONFIG = Config(retries={"max_attempts": 16, "mode": "standard"})


class HttpResponse(BaseModel):
    status_code: int
    body: str


def region_of(arn: str) -> str:
    return arn.split(":")[3]


@asynccontextmanager
async def agentcore_client(region_name: str, config: Config = SESSION_CLIENT_CONFIG):
    """One ``bedrock-agentcore`` data-plane client, open for the whole block."""
    async with (await get_aioboto3_session()).client(
        "bedrock-agentcore", region_name=region_name, config=config
    ) as acr:  # type: ignore
        yield acr


@asynccontextmanager
async def _client_for(arn: str, client: Any | None):
    """The caller's client, or one opened just for this call if it passed none."""
    if client is not None:
        yield client
        return
    async with agentcore_client(region_of(arn)) as acr:
        yield acr


async def start_agentcore_session(runtime_arn: str, session_id: str, client: Any | None = None):
    async with _client_for(runtime_arn, client) as acr:
        resp = await acr.invoke_agent_runtime_command(
            agentRuntimeArn=runtime_arn,
            runtimeSessionId=session_id,
            contentType="application/json",
            accept="application/json",
            body={"command": "echo hello", "timeout": 60},
        )
        async for chunk in resp["stream"]:
            last_chunk = chunk
    assert last_chunk["chunk"]["contentStop"]["exitCode"] == 0, "Failed to start the agent runtime"


async def invoke_agentcore_session(
    runtime_arn: str, session_id: str, payload: bytes, client: Any | None = None
) -> HttpResponse:
    async with _client_for(runtime_arn, client) as acr:
        resp = await acr.invoke_agent_runtime(
            agentRuntimeArn=runtime_arn, runtimeSessionId=session_id, contentType="application/json", payload=payload
        )
        async with resp["response"] as body:
            return HttpResponse(status_code=resp["statusCode"], body=(await body.read()).decode())


async def stop_agentcore_session(capacity_provider_arn: str, session_id: str, client: Any | None = None):
    capacity_provider_id = capacity_provider_arn.split("/")[-1]
    try:
        async with _client_for(capacity_provider_arn, client) as acr:
            await acr.delete_capacity_provider_session(capacityProviderId=capacity_provider_id, sessionId=session_id)
    except Exception as e:
        if type(e).__name__ == "ResourceNotFoundException":
            # already deleted
            return
        else:
            raise e


@asynccontextmanager
async def agentcore_session(capacity_provider_arn: str, runtime_arn: str, session_id: str, client: Any | None = None):
    assert region_of(capacity_provider_arn) == region_of(runtime_arn)
    async with _client_for(runtime_arn, client) as acr:
        await start_agentcore_session(runtime_arn, session_id, acr)
        yield
        await stop_agentcore_session(capacity_provider_arn, session_id, acr)
