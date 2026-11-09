"""Run the unmodified E2B SDK against a sandbox on AgentCore Runtime.

Usage:
    AGENTCORE_E2B_RUNTIME_ARN=arn:aws:bedrock-agentcore:...:runtime/... uv run python run_e2b.py
"""

import os
import sys

from e2b import Sandbox

from agentcore_rl_toolkit.sandbox.e2b import AgentCoreE2BProxy

runtime_arn = os.environ.get("AGENTCORE_E2B_RUNTIME_ARN")
if not runtime_arn:
    sys.exit("Set AGENTCORE_E2B_RUNTIME_ARN to the ARN printed by deploy.py (see README.md)")

with AgentCoreE2BProxy() as proxy:
    with Sandbox.create(
        template=runtime_arn, timeout=600, api_key=proxy.api_key, api_url=proxy.url, sandbox_url=proxy.url
    ) as sbx:
        print(f"Sandbox: {sbx.sandbox_id}")
        print(sbx.commands.run("echo hello from $(whoami) in $(pwd) on $(uname -m)").stdout, end="")

        # Output streams while the command runs.
        sbx.commands.run("for i in 1 2 3; do echo tick $i; sleep 1; done", on_stdout=lambda line: print(line, end=""))

        sbx.files.write("/home/user/greet.sh", 'echo "hi from a file"')
        print(sbx.commands.run("sh greet.sh").stdout, end="")
        print([entry.name for entry in sbx.files.list("/home/user")])

        handle = sbx.commands.run("sleep 2; echo done in the background", background=True)
        print(handle.wait().stdout, end="")
    # Leaving the block kills the sandbox, which stops its AgentCore session.
