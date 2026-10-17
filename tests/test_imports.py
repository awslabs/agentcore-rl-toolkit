"""The HTTP app and client can be imported independently."""

import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    ("name", "other_module"),
    [("AgentCoreRuntimeApp", "client"), ("AgentCoreHttpClient", "app")],
)
def test_runtime_import_isolated(name, other_module):
    # A fresh interpreter keeps imports by other tests out of sys.modules.
    code = f"""
from agentcore_rl_toolkit import {name}
import sys
assert "agentcore_rl_toolkit.app" not in sys.modules
assert "agentcore_rl_toolkit.client" not in sys.modules
assert "agentcore_rl_toolkit.runtime.{other_module}" not in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
