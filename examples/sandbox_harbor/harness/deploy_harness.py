#!/usr/bin/env python3
"""Deploy an agent HARNESS as an AgentCore runtime under its CANONICAL name.

    python deploy_harness.py --agent claude-code --image-tag claude-code-v1

Creates the runtime ``harness_<agent>_<version>`` (benchmark-agnostic) pointing
at ``$HARNESS_REPO:<image-tag>`` (repo from .env), on the serverless arm64 MicroVM recipe
(PUBLIC network, HTTP, idle 900s / maxLifetime 8h). Idempotent: if the name
already exists it prints the ARN and exits without change, so it never disturbs
a runtime an in-flight run is using.
"""
from __future__ import annotations

import argparse
import sys
import time

import boto3
from harbor_sandbox.config import ECR_REGISTRY, HARNESS_REPO, IDLE_SESSION_TIMEOUT_S, MAX_LIFETIME_S, REGION, ROLE_ARN


def find(ctrl, name: str):
    kw = {"maxResults": 100}
    while True:
        r = ctrl.list_agent_runtimes(**kw)
        for rt in r.get("agentRuntimes", []):
            if rt.get("agentRuntimeName") == name:
                return rt["agentRuntimeArn"]
        kw["nextToken"] = r.get("nextToken")
        if not kw["nextToken"]:
            return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True, help="strands | claude-code")
    ap.add_argument("--image-tag", required=True, dest="image_tag", help="$HARNESS_REPO:<tag> to point the runtime at")
    ap.add_argument("--version", default="v1")
    a = ap.parse_args()

    name = f"harness_{a.agent.replace('-', '_')}_{a.version}"
    uri = f"{ECR_REGISTRY}/{HARNESS_REPO}:{a.image_tag}"
    ctrl = boto3.client("bedrock-agentcore-control", region_name=REGION)

    existing = find(ctrl, name)
    if existing:
        print(f"exists (no change): {name} -> {existing}")
        return

    resp = ctrl.create_agent_runtime(
        agentRuntimeName=name,
        agentRuntimeArtifact={"containerConfiguration": {"containerUri": uri}},
        roleArn=ROLE_ARN,
        networkConfiguration={"networkMode": "PUBLIC"},
        protocolConfiguration={"serverProtocol": "HTTP"},
        lifecycleConfiguration={"idleRuntimeSessionTimeout": IDLE_SESSION_TIMEOUT_S, "maxLifetime": MAX_LIFETIME_S},
    )
    arn = resp["agentRuntimeArn"]
    rid = arn.split("/")[-1]
    print(f"creating {name}\n  image {uri}\n  arn   {arn}")

    for _ in range(90):
        st = ctrl.get_agent_runtime(agentRuntimeId=rid).get("status")
        print(f"  status: {st}")
        if st == "READY":
            print(f"READY: {name} -> {arn}")
            return
        if st and ("FAIL" in st or st == "DELETING"):
            sys.exit(f"deploy failed: status={st}")
        time.sleep(10)
    sys.exit("timed out waiting for READY")


if __name__ == "__main__":
    main()
