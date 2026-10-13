"""Deploy harness and task runtimes from a prepared manifest."""

import argparse
import json
import time
import tomllib
from pathlib import Path

import boto3

from agentcore_rl_toolkit.concurrency.rate_limiter import LocalRateLimiter


def ensure_runtime(client, name, image, role, network, env, *, runtime_arn=None, max_lifetime=3600):
    desired = {
        "agentRuntimeArtifact": {"containerConfiguration": {"containerUri": image}},
        "roleArn": role,
        "networkConfiguration": network,
        "protocolConfiguration": {"serverProtocol": "HTTP"},
        "environmentVariables": env,
        "lifecycleConfiguration": {"idleRuntimeSessionTimeout": 300, "maxLifetime": max_lifetime},
        "platformVersion": "V2",
    }
    if runtime_arn:
        runtime_id = runtime_arn.rsplit("/", 1)[-1]
        current = client.get_agent_runtime(agentRuntimeId=runtime_id)
        # The service omits this field when no environment variables are set.
        current["environmentVariables"] = current.get("environmentVariables") or {}
        # Returned by get, but the service disallows writing it for new runtimes.
        current["networkConfiguration"].get("networkModeConfig", {}).pop("requireServiceS3Endpoint", None)
        if any(current.get(key) != value for key, value in desired.items()):
            client.update_agent_runtime(agentRuntimeId=runtime_id, **desired)
            print(f"Updating {name}", flush=True)
        else:
            print(f"Reusing {name}", flush=True)
    else:
        response = client.create_agent_runtime(agentRuntimeName=name, **desired)
        runtime_id = response["agentRuntimeId"]
        print(f"Creating {name}", flush=True)
    deadline = time.monotonic() + 1200
    while time.monotonic() < deadline:
        current = client.get_agent_runtime(agentRuntimeId=runtime_id)
        if current["status"] == "READY":
            endpoint = client.get_agent_runtime_endpoint(agentRuntimeId=runtime_id, endpointName="DEFAULT")
            if endpoint["status"] == "READY" and endpoint["liveVersion"] == current["agentRuntimeVersion"]:
                return current["agentRuntimeArn"]
        if current["status"].endswith("FAILED"):
            raise RuntimeError(current)
        time.sleep(10)
    raise TimeoutError(f"Runtime {name} did not become ready")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text())
    manifest = json.loads(args.manifest.read_text())
    client = boto3.client("bedrock-agentcore-control", region_name=config["region"])
    limiter = LocalRateLimiter(tps_limit=1)
    # Gate every HTTP attempt, including SDK retries.
    client.meta.events.register("before-send.*.*", lambda **kwargs: limiter.wait_sync())
    harness_network = {
        "networkMode": "VPC",
        "networkModeConfig": {"subnets": config["subnets"], "securityGroups": config["harness_security_groups"]},
    }
    task_network = {
        "networkMode": "VPC",
        "networkModeConfig": {"subnets": config["subnets"], "securityGroups": config["task_security_groups"]},
    }
    max_lifetime = max(
        3600, *(int(task["agent_timeout"] + task["verifier_timeout"]) + 300 for task in manifest["tasks"])
    )
    manifest["harness_arn"] = ensure_runtime(
        client,
        config["name"] + "_harness",
        manifest["harness_image"],
        config["harness_role"],
        harness_network,
        {"VERIFIER_BUCKET": config["verifier_bucket"], "VERIFIER_REGION": config["verifier_region"]},
        runtime_arn=manifest.get("harness_arn"),
        max_lifetime=max_lifetime,
    )
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    for task in manifest["tasks"]:
        task["runtime_arn"] = ensure_runtime(
            client,
            (config["name"] + "_" + task["task_id"]).replace("-", "_"),
            task["image_uri"],
            config["task_role"],
            task_network,
            {},
            runtime_arn=task.get("runtime_arn"),
            max_lifetime=max(3600, int(task["agent_timeout"] + task["verifier_timeout"]) + 300),
        )
        args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
