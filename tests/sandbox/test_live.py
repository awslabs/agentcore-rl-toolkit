"""Opt-in probe against a runtime deployed with this revision of sandboxd."""

import os

import pytest
from botocore.exceptions import ReadTimeoutError

from agentcore_rl_toolkit.sandbox import ExecError, SandboxClient

pytestmark = pytest.mark.skipif(
    not os.environ.get("SANDBOX_RUNTIME_ARN"), reason="set SANDBOX_RUNTIME_ARN for live ACR coverage"
)


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


def test_live_foreground_disconnect():
    client = SandboxClient(runtime_arn=os.environ["SANDBOX_RUNTIME_ARN"])
    with client.start() as sandbox:
        # Warm the session before deliberately timing out the foreground request.
        assert sandbox.exec("printf ready").stdout == "ready"
        impatient = SandboxClient(runtime_arn=client.runtime_arn, read_timeout=1, max_retry_attempts=0)
        with pytest.raises(ExecError) as caught:
            impatient.attach(sandbox.session_id).exec(
                "printf x >> /app/execution-count; printf before; sleep 5; printf after"
            )
        assert isinstance(caught.value.__cause__, ReadTimeoutError)

        recovered = sandbox.get_exec(caught.value.handle.invocation_id)
        result = recovered.result(timeout=30)
        assert result.stdout == "beforeafter"
        assert result.exit_code == 0
        assert sandbox.exec("cat /app/execution-count").stdout == "x"


def test_live_output_and_local_wait():
    client = SandboxClient(runtime_arn=os.environ["SANDBOX_RUNTIME_ARN"])
    with client.start() as sandbox:
        value = "spaces ' quotes $HOME"
        result = sandbox.exec('printf "%s\\n" "$VALUE"; pwd', cwd="/app", env={"VALUE": value})
        assert result.stdout == f"{value}\n/app\n"

        handle = sandbox.exec("sleep 5; printf finished", background=True)
        with pytest.raises(TimeoutError):
            handle.result(timeout=0)
        assert handle.result(timeout=30).stdout == "finished"

        result = sandbox.exec("head -c 300000 /dev/zero | tr '\\000' x; head -c 300000 /dev/zero | tr '\\000' y >&2")
        assert result.exit_code == 0
        assert result.stdout == "x" * (256 * 1024)
        assert result.stderr == "y" * (256 * 1024)
        assert result.stdout_truncated and result.stderr_truncated
