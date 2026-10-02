"""Sandbox quickstart: start a sandbox session, run commands, terminate.

Usage:
    SANDBOX_RUNTIME_ARN=arn:aws:bedrock-agentcore:...:runtime/... python run_sandbox.py
"""

import os
import sys

from agentcore_rl_toolkit.sandbox import SandboxClient

runtime_arn = os.environ.get("SANDBOX_RUNTIME_ARN")
if not runtime_arn:
    sys.exit("Set SANDBOX_RUNTIME_ARN to the ARN of your deployed sandbox runtime (see README.md)")

client = SandboxClient(runtime_arn=runtime_arn)

with client.start() as sb:
    print(f"Sandbox session: {sb.session_id}")

    result = sb.exec(["uname", "-m"])
    print(f"architecture: {result.stdout.strip()}")

    result = sb.exec(["/bin/sh", "-c", "echo hello from $(uname -m); pwd"], timeout=60)
    print(f"exit_code={result.exit_code} timed_out={result.timed_out}")
    print(f"stdout: {result.stdout}")
    print(f"stderr: {result.stderr}")

    # Each call starts a fresh process — use cwd/env to
    # re-establish state per call.
    result = sb.exec(["/bin/sh", "-c", "echo $GREETING from $PWD"], cwd="/tmp", env={"GREETING": "hi"})
    print(f"stdout: {result.stdout}")
# Context exit terminates the sandbox (ping -> Healthy, then StopRuntimeSession).
print("Sandbox terminated.")
