"""Tests for sandbox data types and command composition."""

import pytest

from agentcore_rl_toolkit.sandbox import ExecResult
from agentcore_rl_toolkit.sandbox.client import _compose_command


class TestExecResult:
    def test_fields(self):
        """All four fields are stored as given."""
        result = ExecResult(exit_code=17, stdout="out", stderr="err", timed_out=False)
        assert result.exit_code == 17
        assert result.stdout == "out"
        assert result.stderr == "err"
        assert result.timed_out is False


class TestComposeCommand:
    def test_passthrough(self):
        """Without cwd/env the command is unchanged."""
        assert _compose_command("pytest -q") == "pytest -q"

    def test_cwd_only(self):
        assert _compose_command("pytest -q", cwd="/app") == "cd /app || exit $?; pytest -q"

    def test_env_only(self):
        assert _compose_command("run.sh", env={"FOO": "bar"}) == "export FOO=bar || exit $?; run.sh"

    def test_cwd_and_env(self):
        composed = _compose_command("pytest -q", cwd="/app", env={"FOO": "bar"})
        assert composed == "cd /app || exit $?; export FOO=bar || exit $?; pytest -q"

    @pytest.mark.parametrize(
        "shell,kwargs",
        [
            ("/bin/sh", {"cwd": "missing"}),
            ("/bin/bash", {"env": {"SHELLOPTS": "readonly"}}),
        ],
    )
    def test_failed_setup_never_executes_command(self, tmp_path, shell, kwargs):
        """Neither part of a compound command may run after setup fails."""
        import subprocess

        # SHELLOPTS is readonly in Bash, so exporting it fails.
        composed = _compose_command("echo FIRST; touch executed", **kwargs)
        proc = subprocess.run([shell, "-c", composed], cwd=tmp_path, capture_output=True, text=True)
        assert proc.returncode != 0
        assert proc.stdout == ""
        assert not (tmp_path / "executed").exists()

    def test_multiple_env_vars(self):
        composed = _compose_command("cmd", env={"A": "1", "B": "2"})
        assert composed == "export A=1 B=2 || exit $?; cmd"

    def test_quoting_value_with_spaces(self):
        composed = _compose_command("cmd", env={"MSG": "hello world"})
        assert composed == "export MSG='hello world' || exit $?; cmd"

    def test_quoting_value_with_single_quote(self):
        composed = _compose_command("cmd", env={"MSG": "it's"})
        assert "it" in composed
        # shlex.quote produces a shell-safe form; exact escaping is delegated
        import shlex

        assert shlex.split(composed.removeprefix("export ").removesuffix(" || exit $?; cmd"))[0] == "MSG=it's"

    def test_quoting_cwd_with_spaces(self):
        composed = _compose_command("ls", cwd="/tmp/my dir")
        assert composed == "cd '/tmp/my dir' || exit $?; ls"

    def test_non_string_env_value_coerced(self):
        assert _compose_command("cmd", env={"N": 3}) == "export N=3 || exit $?; cmd"

    def test_invalid_env_key_raises(self):
        with pytest.raises(ValueError, match="Invalid environment variable name"):
            _compose_command("cmd", env={"BAD-KEY": "x"})

    def test_env_key_injection_raises(self):
        with pytest.raises(ValueError):
            _compose_command("cmd", env={"X; rm -rf /": "x"})
