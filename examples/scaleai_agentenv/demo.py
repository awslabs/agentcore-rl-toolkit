"""Run a Strands agent against SlackEnv, then grade the environment state."""

import argparse
import asyncio
import json
import sys
import uuid
from contextlib import asynccontextmanager

import httpx
from strands import Agent
from strands.models import BedrockModel
from strands.tools.mcp import MCPClient

from agentcore_rl_toolkit.aws_tools.agentcore_tools import agentcore_client, region_of
from agentcore_rl_toolkit.aws_tools.session_endpoints import session_endpoint_client


async def data(client: httpx.AsyncClient, method: str, **params) -> dict:
    response = await client.post(
        "/agentenv", json={"jsonrpc": "2.0", "id": 1, "method": f"data/{method}", "params": params}
    )
    response.raise_for_status()
    body = response.json()
    if "error" in body:
        raise RuntimeError(body["error"])
    return body["result"]


async def messages(client: httpx.AsyncClient) -> list[dict]:
    return (await data(client, "get"))["parts"][0]["data"]["messages"]


async def exercise(client: httpx.AsyncClient, model: BedrockModel) -> dict:
    print("1. Discover the environment: GET /.well-known/agent-env.json", file=sys.stderr)
    response = await client.get("/.well-known/agent-env.json")
    response.raise_for_status()
    card = response.json()
    if card["name"] != "slack":
        raise RuntimeError(f"Unexpected environment: {card['name']}")

    print("2. Initialize task state: POST /agentenv (data/reset, data/add)", file=sys.stderr)
    await data(client, "reset")
    seed = {"channel": "general", "text": "Welcome"}
    expected = {"channel": "general", "text": "Hello from AgentCore"}
    await data(client, "add", parts=[{"kind": "data", "data": {"messages": [seed]}}])

    print("3. Run Strands: /mcp (initialize, tools/list, tools/call)", file=sys.stderr)
    # Strands owns its transport; reuse the auth, not the HTTP connection pool.
    with MCPClient(url=str(client.base_url.join("/mcp")), auth_provider=client.auth) as mcp:
        agent = Agent(model=model, tools=mcp.list_tools_sync(), callback_handler=None)
        result = await agent.invoke_async(
            f"Send exactly one Slack message to channel {expected['channel']!r} "
            f"with the exact text {expected['text']!r}. Use the available tool."
        )

    print("4. Read the final state and compute reward: POST /agentenv (data/get)", file=sys.stderr)
    state = await messages(client)
    return {"reward": float(state == [seed, expected]), "messages": state, "agent_response": str(result)}


@asynccontextmanager
async def runtime_environment(runtime_arn: str):
    region = region_of(runtime_arn)
    session_id = str(uuid.uuid4())
    async with agentcore_client(region) as runtime:
        # MCP ping starts the Runtime session without initializing task state.
        print("Open a Runtime session: InvokeAgentRuntime → session endpoint URL", file=sys.stderr)
        response = await runtime.invoke_agent_runtime(
            agentRuntimeArn=runtime_arn,
            # MCP protocol 2026-07-28+ can route using runtimeSessionId alone.
            mcpSessionId=session_id,
            runtimeSessionId=session_id,
            contentType="application/json",
            accept="application/json, text/event-stream",
            payload=json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"}),
        )
        try:
            async with response["response"] as body:
                ping = json.loads(await body.read())
            if "error" in ping:
                raise RuntimeError(ping["error"])
            url = response.get("sessionEndpointUrl") or response["ResponseMetadata"]["HTTPHeaders"].get(
                "x-amzn-bedrock-agentcore-runtime-session-endpoint-url"
            )
            if not url:
                raise RuntimeError("Runtime returned no session endpoint URL. Check allowedPorts.")
            async with session_endpoint_client(url, region) as client:
                yield client
        finally:
            print("5. Release the environment: StopRuntimeSession", file=sys.stderr)
            await runtime.stop_runtime_session(agentRuntimeArn=runtime_arn, runtimeSessionId=session_id)


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--local-url", help="For example, http://127.0.0.1:8000")
    target.add_argument("--runtime-arn")
    parser.add_argument("--model-id", required=True, help="Bedrock model or inference profile with tool support")
    args = parser.parse_args()
    model = BedrockModel(model_id=args.model_id, streaming=False)
    if args.local_url:
        async with httpx.AsyncClient(base_url=args.local_url) as client:
            result = await exercise(client, model)
    else:
        async with runtime_environment(args.runtime_arn) as client:
            result = await exercise(client, model)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
