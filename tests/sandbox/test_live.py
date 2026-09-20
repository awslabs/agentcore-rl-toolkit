"""Opt-in probe against a runtime deployed with this revision of sandboxd."""

import os

import pytest

from agentcore_rl_toolkit.sandbox import SandboxClient


@pytest.mark.skipif(not os.environ.get("SANDBOX_RUNTIME_ARN"), reason="set SANDBOX_RUNTIME_ARN for live ACR coverage")
def test_live_managed_execution():
    client = SandboxClient(runtime_arn=os.environ["SANDBOX_RUNTIME_ARN"])
    with client.start() as sandbox:
        foreground = sandbox.exec("printf foreground; printf warning >&2; exit 3")
        assert (foreground.stdout, foreground.stderr, foreground.exit_code) == ("foreground", "warning", 3)

        handle = sandbox.exec("printf before; sleep 2; printf after", background=True)
        other = SandboxClient(runtime_arn=client.runtime_arn).attach(sandbox.session_id)
        recovered = other.get_exec(handle.invocation_id).result(timeout=30)
        assert recovered.stdout == "beforeafter"
        assert recovered.exit_code == 0
        assert other.exec("echo must-not-run", invocation_id=handle.invocation_id) == recovered

        timed_out = sandbox.exec("printf partial; sleep 30", timeout=1)
        assert timed_out.timed_out
        assert timed_out.stdout == "partial"
