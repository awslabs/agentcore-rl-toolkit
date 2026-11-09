"""Create or update the AgentCore runtime that hosts E2B-compatible sandboxes.

Usage:
    uv run --script deploy.py --name e2b_sandbox --image <ECR image URI> --role-arn <execution role ARN>

Prints the runtime ARN, which the E2B SDK uses as its template.
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "boto3>=1.43.123",  # allowedPorts and custom serving ports
# ]
# ///

import argparse
import time

import boto3

SERVING_PORT = 11000  # sandboxd
ENVD_PORT = 10983


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, help="runtime name: letters, digits and underscores")
    parser.add_argument("--image", required=True, help="ECR URI of the image built from this directory")
    parser.add_argument("--role-arn", required=True, help="execution role that can pull the image")
    parser.add_argument("--region", help="defaults to the boto3 session's Region")
    parser.add_argument("--max-lifetime", type=int, default=3600, help="seconds before AgentCore stops a session")
    args = parser.parse_args()

    control = boto3.client("bedrock-agentcore-control", region_name=args.region)
    config = {
        "agentRuntimeArtifact": {"containerConfiguration": {"containerUri": args.image}},
        "roleArn": args.role_arn,
        "networkConfiguration": {"networkMode": "PUBLIC"},
        "protocolConfiguration": {"serverProtocol": "HTTP", "port": SERVING_PORT},
        # The session endpoint reaches only these ports; the proxy sends envd traffic to ENVD_PORT.
        "allowedPorts": [{"port": ENVD_PORT}],
        # sandboxd keeps a started session busy, so maxLifetime bounds a session nobody stops.
        "lifecycleConfiguration": {"idleRuntimeSessionTimeout": 300, "maxLifetime": args.max_lifetime},
    }

    try:
        runtime_id = control.create_agent_runtime(agentRuntimeName=args.name, **config)["agentRuntimeId"]
        print(f"Creating {args.name}", flush=True)
    except control.exceptions.ConflictException:
        # Get takes the generated runtime ID, not the name, so look the name up.
        runtime_id = next(
            runtime["agentRuntimeId"]
            for page in control.get_paginator("list_agent_runtimes").paginate()
            for runtime in page["agentRuntimes"]
            if runtime["agentRuntimeName"] == args.name
        )
        control.update_agent_runtime(agentRuntimeId=runtime_id, **config)
        print(f"Updating {args.name}", flush=True)

    deadline = time.monotonic() + 1200
    while time.monotonic() < deadline:
        runtime = control.get_agent_runtime(agentRuntimeId=runtime_id)
        if runtime["status"].endswith("FAILED"):
            raise SystemExit(f"{args.name} failed: {runtime.get('failureReason', runtime['status'])}")
        if runtime["status"] == "READY":
            endpoint = control.get_agent_runtime_endpoint(agentRuntimeId=runtime_id, endpointName="DEFAULT")
            if endpoint["status"] == "READY" and endpoint["liveVersion"] == runtime["agentRuntimeVersion"]:
                print(runtime["agentRuntimeArn"])
                return
        time.sleep(10)
    raise SystemExit(f"{args.name} did not become ready")


if __name__ == "__main__":
    main()
