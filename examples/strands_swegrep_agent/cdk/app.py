#!/usr/bin/env python3
import os
from pathlib import Path

import aws_cdk as cdk
import yaml
from swegrep_agent_stack import SweGrepAgentCoreStack

app = cdk.App()

# config.yaml supplies the stack's context; `-c key=value` on the command line wins.
cfg: dict = {}
config_path = Path(__file__).parent / (app.node.try_get_context("config") or "config.yaml")
if config_path.exists():
    cfg = yaml.safe_load(config_path.read_text()) or {}

for key, value in cfg.items():
    if key in ("awsAccount", "awsRegion"):
        continue
    if app.node.try_get_context(key) is None:
        app.node.set_context(key, value)

# The VPC lookup needs a concrete account and region.
account = cfg.get("awsAccount") or os.environ.get("CDK_DEFAULT_ACCOUNT")
region = cfg.get("awsRegion") or os.environ.get("CDK_DEFAULT_REGION")

SweGrepAgentCoreStack(
    app,
    "SweGrepAgentCoreStack",
    env=cdk.Environment(account=account, region=region),
    description=(
        "SWE-grep code-region retrieval agent on AWS Bedrock AgentCore Runtime "
        "for RL training (agentcore-rl-toolkit)."
    ),
)

app.synth()
