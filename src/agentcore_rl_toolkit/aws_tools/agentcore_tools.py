"""AgentCore *data plane*: one session's lifecycle, once per rollout.

Every function here takes the ARNs it needs as arguments, so this module reads no
config of its own -- which is what makes it safe on the rollout path, where config
has already arrived as hydra config. Provisioning the pool and runtime these
sessions run on is the control plane, once per image roll and out of scope here.
"""

from contextlib import asynccontextmanager

from botocore.config import Config
from pydantic import BaseModel

from agentcore_rl_toolkit.aws_tools.boto3_tools import get_aioboto3_session


class HttpResponse(BaseModel):
    status_code: int
    body: str


async def start_agentcore_session(runtime_arn: str, session_id: str):
    region_name = runtime_arn.split(":")[3]
    config = Config(retries={"max_attempts": 16, "mode": "standard"})
    async with (await get_aioboto3_session()).client(
        "bedrock-agentcore", region_name=region_name, config=config
    ) as acr:  # type: ignore
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


async def invoke_agentcore_session(runtime_arn: str, session_id: str, payload: bytes) -> HttpResponse:
    region_name = runtime_arn.split(":")[3]
    async with (await get_aioboto3_session()).client("bedrock-agentcore", region_name=region_name) as acr:  # type: ignore
        resp = await acr.invoke_agent_runtime(
            agentRuntimeArn=runtime_arn, runtimeSessionId=session_id, contentType="application/json", payload=payload
        )
        async with resp["response"] as body:
            return HttpResponse(status_code=resp["statusCode"], body=(await body.read()).decode())


async def stop_agentcore_session(capacity_provider_arn: str, session_id: str):
    region_name = capacity_provider_arn.split(":")[3]
    capacity_provider_id = capacity_provider_arn.split("/")[-1]
    try:
        async with (await get_aioboto3_session()).client("bedrock-agentcore", region_name=region_name) as acr:  # type: ignore
            await acr.delete_capacity_provider_session(capacityProviderId=capacity_provider_id, sessionId=session_id)
    except Exception as e:
        if type(e).__name__ == "ResourceNotFoundException":
            # Session is already deleted, nothing else to do
            return
        else:
            # otherwise let caller know that we did not succeed
            raise e


@asynccontextmanager
async def agentcore_session(capacity_provider_arn: str, runtime_arn: str, session_id: str):
    await start_agentcore_session(runtime_arn, session_id)
    yield
    await stop_agentcore_session(capacity_provider_arn, session_id)
