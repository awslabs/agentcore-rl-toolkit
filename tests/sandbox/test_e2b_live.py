"""Opt-in probe of the unmodified E2B SDK against a runtime deployed from examples/sandbox_e2b."""

import os
import time

import pytest

from agentcore_rl_toolkit.sandbox.e2b import AgentCoreE2BProxy

pytestmark = pytest.mark.skipif(
    not os.environ.get("AGENTCORE_E2B_RUNTIME_ARN"), reason="set AGENTCORE_E2B_RUNTIME_ARN for live ACR coverage"
)


@pytest.fixture
def sandbox(monkeypatch):
    e2b = pytest.importorskip("e2b")
    with AgentCoreE2BProxy() as proxy:
        for name, value in proxy.environ.items():
            monkeypatch.setenv(name, value)
        with e2b.Sandbox.create(template=os.environ["AGENTCORE_E2B_RUNTIME_ARN"], timeout=600) as sbx:
            yield sbx


def test_live_commands_and_files(sandbox):
    from e2b import CommandExitException

    result = sandbox.commands.run("echo $A from $(whoami) in $(pwd)", envs={"A": "hi"})
    assert result.stdout == "hi from user in /home/user\n"
    with pytest.raises(CommandExitException) as caught:
        sandbox.commands.run("echo boom >&2; exit 3")
    assert (caught.value.exit_code, caught.value.stderr) == (3, "boom\n")

    started = time.monotonic()
    arrivals = []
    sandbox.commands.run(
        "for i in 1 2 3; do echo $i; sleep 1; done", on_stdout=lambda _: arrivals.append(time.monotonic())
    )
    assert len(arrivals) == 3 and arrivals[-1] - arrivals[0] >= 1.5 and arrivals[0] - started < 1.5

    with pytest.raises(Exception, match="not supported on AgentCore Runtime"):
        sandbox.commands.run("whoami", user="root")

    handle = sandbox.commands.run("sleep 2; echo late", background=True)
    assert sandbox.commands.connect(handle.pid).wait().stdout == "late\n"

    payload = os.urandom(1024 * 1024)
    sandbox.files.write("/home/user/data/blob.bin", payload)
    assert sandbox.files.read("/home/user/data/blob.bin", format="bytes") == payload
    assert "blob.bin" in [entry.name for entry in sandbox.files.list("/home/user/data")]
