"""Create or update an AgentCore Runtime from an already-pushed container image."""

import argparse
import json
import time
from contextlib import closing
from pathlib import Path

import boto3


def deploy(client, args):
    environment = json.loads(args.env.read_text())
    if not isinstance(environment, dict) or any(
        not isinstance(key, str) or not isinstance(value, str) for key, value in environment.items()
    ):
        raise ValueError("--env must contain a JSON object of string environment variables")
    if bool(args.subnet) != bool(args.security_group):
        raise ValueError("Supply both --subnet and --security-group for VPC networking")
    network = {"networkMode": "PUBLIC"}
    if args.subnet:
        network = {
            "networkMode": "VPC",
            "networkModeConfig": {"subnets": args.subnet, "securityGroups": args.security_group},
        }
    config = {
        "agentRuntimeArtifact": {"containerConfiguration": {"containerUri": args.image}},
        "roleArn": args.role,
        "networkConfiguration": network,
        "protocolConfiguration": {"serverProtocol": args.protocol},
        "environmentVariables": environment,
        "lifecycleConfiguration": {"idleRuntimeSessionTimeout": 300, "maxLifetime": 28800},
        "platformVersion": "V2",
    }
    if args.runtime_id:
        response = client.update_agent_runtime(agentRuntimeId=args.runtime_id, **config)
    else:
        response = client.create_agent_runtime(agentRuntimeName=args.name, **config)
    runtime_id = response["agentRuntimeId"]
    deadline = time.monotonic() + args.wait_seconds
    while time.monotonic() < deadline:
        runtime = client.get_agent_runtime(agentRuntimeId=runtime_id)
        if runtime["status"] == "READY":
            return runtime["agentRuntimeArn"]
        if runtime["status"] not in {"CREATING", "UPDATING"}:
            raise RuntimeError(f"Runtime {runtime_id}: {runtime['status']}: {runtime.get('failureReason', '')}")
        time.sleep(10)
    raise TimeoutError(f"Runtime {runtime_id} is still preparing; inspect it before retrying")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    identity = parser.add_mutually_exclusive_group(required=True)
    identity.add_argument("--name", help="Name for a new Runtime")
    identity.add_argument("--runtime-id", help="ID of an existing Runtime to update")
    parser.add_argument("--image", required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--protocol", choices=["MCP", "HTTP"], default="MCP")
    parser.add_argument("--env", type=Path, required=True, help="External JSON environment configuration")
    parser.add_argument("--subnet", action="append")
    parser.add_argument("--security-group", action="append")
    parser.add_argument("--region", default=None)
    parser.add_argument("--wait-seconds", type=int, default=900)
    args = parser.parse_args()
    if args.wait_seconds <= 0:
        parser.error("--wait-seconds must be positive")
    with closing(boto3.client("bedrock-agentcore-control", region_name=args.region)) as client:
        print(deploy(client, args))


if __name__ == "__main__":
    main()
