"""Real SDK -> botocore HTTP -> sandboxd -> shell contract tests.

Run locally/CI with Go installed. AWS routing/signing alone is replaced; the
request encoding and response parser are the real botocore implementation.
"""

import os
import shutil
import socket
import subprocess
import time
import urllib.request
from pathlib import Path
from unittest.mock import patch

import boto3
import pytest
from botocore import UNSIGNED
from botocore.config import Config

from agentcore_rl_toolkit.sandbox import ExecError, SandboxClient

ARN = "arn:aws:bedrock-agentcore:us-west-2:123456789012:runtime/local-test"


@pytest.fixture(scope="module")
def daemon(tmp_path_factory):
    if not shutil.which("go"):
        pytest.skip("sandboxd contract tests require Go")
    scratch = tmp_path_factory.mktemp("sandboxd")
    binary = scratch / "sandboxd"
    env = {
        **os.environ,
        "GOCACHE": str(scratch / "go-cache"),
        "GOMODCACHE": str(scratch / "go-mod-cache"),
    }
    subprocess.run(
        ["go", "build", "-buildvcs=false", "-o", str(binary), "."],
        cwd=Path(__file__).resolve().parents[2] / "sandboxd",
        env=env,
        check=True,
        timeout=120,
    )
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    address = f"127.0.0.1:{port}"
    url = f"http://{address}"
    with (scratch / "daemon.log").open("wb") as log:
        process = subprocess.Popen(
            [str(binary), "--listen", address, "--state-dir", str(scratch / "records")],
            stdout=log,
            stderr=log,
        )
        try:
            deadline = time.monotonic() + 10
            while True:
                try:
                    with urllib.request.urlopen(f"{url}/ping", timeout=1):
                        break
                except OSError:
                    if process.poll() is not None or time.monotonic() > deadline:
                        pytest.fail((scratch / "daemon.log").read_text())
                    time.sleep(0.02)
            yield url
        finally:
            process.terminate()
            process.wait(timeout=10)


@pytest.fixture
def sdk(daemon):
    transports = []

    def create(read_timeout=5):
        transport = boto3.client(
            "bedrock-agentcore",
            region_name="us-west-2",
            endpoint_url=daemon,
            config=Config(signature_version=UNSIGNED, retries={"total_max_attempts": 1}, read_timeout=read_timeout),
        )

        def route(request, **kwargs):
            request.url = f"{daemon}/invocations"

        transport.meta.events.register("before-sign.*.InvokeAgentRuntime", route)
        transports.append(transport)
        with patch("agentcore_rl_toolkit.sandbox.client.boto3.client", return_value=transport):
            return SandboxClient(runtime_arn=ARN)

    yield create
    for transport in transports:
        transport.close()


def test_sdk_foreground_preserves_shell_and_result(sdk, tmp_path):
    sandbox = sdk().start()
    result = sandbox.exec(
        """printf '%s' "$MESSAGE"; pwd >&2; exit 7""",
        cwd=str(tmp_path),
        env={"MESSAGE": """quotes: ' " $ \\ ;"""},
    )
    assert result.stdout == """quotes: ' " $ \\ ;"""
    assert result.stderr == f"{tmp_path}\n"
    assert result.exit_code == 7
    assert not result.timed_out


def test_sdk_background_reconnect_and_retry(sdk, tmp_path):
    sandbox = sdk().start()
    handle = sandbox.exec(
        "echo once >> counter; while [ ! -e release ]; do sleep 0.01; done; cat counter",
        cwd=str(tmp_path),
        background=True,
    )
    assert handle.status() == "in_progress"
    with pytest.raises(TimeoutError):
        handle.result(timeout=0)

    (tmp_path / "release").touch()
    other = sdk().attach(sandbox.session_id)
    result = other.get_exec(handle.invocation_id).result(timeout=5)
    assert result.stdout == "once\n"
    repeated = other.exec("echo MUST_NOT_RUN", invocation_id=handle.invocation_id)
    assert repeated == result
    assert (tmp_path / "counter").read_text() == "once\n"
    assert other.get_exec("unknown").status() == "not_found"


def test_sdk_foreground_connection_loss_is_recoverable(sdk):
    sandbox = sdk(read_timeout=0.05).start()
    with pytest.raises(ExecError) as caught:
        sandbox.exec("sleep 0.3; printf recovered")
    handle = caught.value.handle
    recovered = sdk().attach(handle.session_id).get_exec(handle.invocation_id).result(timeout=5)
    assert recovered.stdout == "recovered"
