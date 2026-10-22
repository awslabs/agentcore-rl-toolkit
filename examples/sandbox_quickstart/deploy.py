"""Deploy the sandbox image as an AgentCore runtime.

Reads config.toml (copy config.example.toml and fill in your values), creates
or updates the runtime, waits for the endpoint to be ready, and prints the
runtime ARN to export as SANDBOX_RUNTIME_ARN.

NOTE: temporary scaffolding, like build_and_push.sh — a future phase moves
runtime provisioning into the SDK (SandboxClient.create()), at which point
this script goes away.
"""

import time
import tomllib
from contextlib import closing
from pathlib import Path

import boto3
from botocore.exceptions import ClientError


def deploy(client, config, *, wait_seconds=120):
    """Create or update the named runtime and wait for DEFAULT to serve its version."""
    name = config["agent_name"]
    desired = {
        "agentRuntimeArtifact": {"containerConfiguration": {"containerUri": config["image_uri"]}},
        "roleArn": config["execution_role_arn"],
        "networkConfiguration": {"networkMode": "PUBLIC"},
        "protocolConfiguration": {"serverProtocol": "HTTP"},
        "environmentVariables": {},
    }
    existing = next(
        (
            runtime
            for page in client.get_paginator("list_agent_runtimes").paginate(maxResults=100)
            for runtime in page["agentRuntimes"]
            if runtime["agentRuntimeName"] == name
        ),
        None,
    )
    if existing:
        response = client.update_agent_runtime(agentRuntimeId=existing["agentRuntimeId"], **desired)
    else:
        response = client.create_agent_runtime(agentRuntimeName=name, **desired)

    runtime_id = response["agentRuntimeId"]
    version = response["agentRuntimeVersion"]
    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        runtime = client.get_agent_runtime(agentRuntimeId=runtime_id)
        if runtime["status"].endswith("FAILED"):
            raise RuntimeError(f"Runtime {name}: {runtime.get('failureReason', runtime['status'])}")
        if runtime["status"] == "READY":
            try:
                endpoint = client.get_agent_runtime_endpoint(agentRuntimeId=runtime_id, endpointName="DEFAULT")
            except ClientError as error:
                if error.response["Error"]["Code"] != "ResourceNotFoundException":
                    raise
            else:
                if endpoint["status"].endswith("FAILED"):
                    raise RuntimeError(f"Endpoint {name}: {endpoint.get('failureReason', endpoint['status'])}")
                if endpoint["status"] == "READY" and endpoint["liveVersion"] == version:
                    return runtime["agentRuntimeArn"]
        time.sleep(5)
    raise TimeoutError(f"Runtime {name} version {version} did not become ready within {wait_seconds}s")


def main():
    config_path = Path(__file__).parent / "config.toml"
    if not config_path.exists():
        raise SystemExit("config.toml not found — copy config.example.toml and fill in your values")
    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    agentcore_config = config["agentcore"]
    with closing(boto3.client("bedrock-agentcore-control", region_name=agentcore_config["region"])) as client:
        agent_arn = deploy(client, agentcore_config)

    print(f"Sandbox runtime ready: {agent_arn}")
    print(f'\nRun the demo with:\n  SANDBOX_RUNTIME_ARN="{agent_arn}" python run_sandbox.py')


if __name__ == "__main__":
    main()
