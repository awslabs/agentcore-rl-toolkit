"""Create a microVM Runtime from an existing ECR image and wait for readiness."""

import argparse
import asyncio
import uuid

from agentcore_rl_toolkit.aws_tools.boto3_tools import (
    close_shared_sigv4_httpx_clients,
    get_async_session,
    shared_sigv4_httpx_client,
)


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--image", required=True, help="ARM64 image URI in ECR")
    parser.add_argument("--role", required=True, help="Existing Runtime execution role ARN")
    parser.add_argument("--region", required=True)
    args = parser.parse_args()
    config = {
        "agentRuntimeName": args.name,
        "agentRuntimeArtifact": {"containerConfiguration": {"containerUri": args.image}},
        "roleArn": args.role,
        "networkConfiguration": {"networkMode": "PUBLIC"},
        "protocolConfiguration": {"serverProtocol": "MCP"},
        "allowedPorts": [{"port": 8000}],
        "lifecycleConfiguration": {"idleRuntimeSessionTimeout": 300, "maxLifetime": 3600},
        "clientToken": str(uuid.uuid4()),
    }
    try:
        session = await get_async_session()
        async with session.create_client("bedrock-agentcore-control", region_name=args.region) as control:
            # REST accepts allowedPorts even when the installed SDK model predates
            # session endpoints. Resolve the service hostname through the SDK.
            http = await shared_sigv4_httpx_client(args.region, "bedrock-agentcore")
            response = await http.put(f"{control.meta.endpoint_url}/runtimes/", json=config)
            response.raise_for_status()
            runtime = response.json()
            print(runtime["agentRuntimeArn"], flush=True)
            async with asyncio.timeout(900):
                while True:
                    current = await control.get_agent_runtime(agentRuntimeId=runtime["agentRuntimeId"])
                    if current["status"] == "READY":
                        print("READY")
                        return
                    if current["status"] not in {"CREATING", "UPDATING"}:
                        raise RuntimeError(current.get("failureReason") or current["status"])
                    await asyncio.sleep(5)
    finally:
        await close_shared_sigv4_httpx_clients()


if __name__ == "__main__":
    asyncio.run(main())
