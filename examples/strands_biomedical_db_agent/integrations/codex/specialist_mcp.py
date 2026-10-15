"""Bridge an IAM-authenticated AgentCore MCP Runtime to local stdio."""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from urllib.parse import quote

import boto3
import httpx2
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

logger = logging.getLogger(__name__)


@asynccontextmanager
async def connect_runtime():
    arn = os.environ["BIOMEDICAL_RUNTIME_ARN"]
    region = arn.split(":")[3]
    aws = boto3.Session(region_name=region)
    runtime = aws.client("bedrock-agentcore")
    session_id = None

    async def sign(request: httpx2.Request):
        credentials = await asyncio.to_thread(aws.get_credentials)
        if credentials is None:
            raise RuntimeError("AWS credentials are required to invoke the specialist Runtime")
        frozen = await asyncio.to_thread(credentials.get_frozen_credentials)
        signed = AWSRequest(
            method=request.method,
            url=str(request.url),
            data=request.content,
            headers=dict(request.headers),
        )
        SigV4Auth(frozen, "bedrock-agentcore", region).add_auth(signed)
        request.headers.update(dict(signed.headers))

    async def capture_session(response: httpx2.Response):
        nonlocal session_id
        if response.is_success:
            session_id = response.headers.get("mcp-session-id", session_id)

    url = (
        f"https://bedrock-agentcore.{region}.amazonaws.com/runtimes/{quote(arn, safe='')}/invocations?qualifier=DEFAULT"
    )
    try:
        async with httpx2.AsyncClient(
            timeout=httpx2.Timeout(660, connect=10),
            event_hooks={"request": [sign], "response": [capture_session]},
        ) as client:
            async with streamable_http_client(url, http_client=client, terminate_on_close=False) as (read, write):
                async with ClientSession(read, write, read_timeout_seconds=660) as session:
                    await session.initialize()
                    yield session
    finally:
        try:
            if session_id:
                await asyncio.to_thread(
                    runtime.stop_runtime_session,
                    agentRuntimeArn=arn,
                    runtimeSessionId=session_id,
                )
        except Exception:
            logger.exception("Failed to stop the specialist Runtime session")
        finally:
            runtime.close()


async def main():
    async with connect_runtime() as remote:

        async def list_tools(_ctx, params):
            return await remote.list_tools(params=params)

        async def call_tool(_ctx, params):
            return await remote.call_tool(params.name, params.arguments)

        server = Server("biomedical-agents", on_list_tools=list_tools, on_call_tool=call_tool)
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
