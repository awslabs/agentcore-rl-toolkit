"""Tests for sandbox data types and command composition."""

import pytest

from agentcore_rl_toolkit.sandbox import ExecResult
from agentcore_rl_toolkit.sandbox.client import _compose_command, _wrap_in_shell


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
        assert _compose_command("pytest -q", cwd="/app") == "cd /app && pytest -q"

    def test_env_only(self):
        assert _compose_command("run.sh", env={"FOO": "bar"}) == "export FOO=bar; run.sh"

    def test_cwd_and_env(self):
        composed = _compose_command("pytest -q", cwd="/app", env={"FOO": "bar"})
        assert composed == "cd /app && export FOO=bar; pytest -q"

    def test_multiple_env_vars(self):
        composed = _compose_command("cmd", env={"A": "1", "B": "2"})
        assert composed == "export A=1 B=2; cmd"

    def test_quoting_value_with_spaces(self):
        composed = _compose_command("cmd", env={"MSG": "hello world"})
        assert composed == "export MSG='hello world'; cmd"

    def test_quoting_value_with_single_quote(self):
        composed = _compose_command("cmd", env={"MSG": "it's"})
        assert "it" in composed
        # shlex.quote produces a shell-safe form; exact escaping is delegated
        import shlex

        assert shlex.split(composed.removeprefix("export ").removesuffix("; cmd"))[0] == "MSG=it's"

    def test_quoting_cwd_with_spaces(self):
        composed = _compose_command("ls", cwd="/tmp/my dir")
        assert composed == "cd '/tmp/my dir' && ls"

    def test_non_string_env_value_coerced(self):
        assert _compose_command("cmd", env={"N": 3}) == "export N=3; cmd"

    def test_invalid_env_key_raises(self):
        with pytest.raises(ValueError, match="Invalid environment variable name"):
            _compose_command("cmd", env={"BAD-KEY": "x"})

    def test_env_key_injection_raises(self):
        with pytest.raises(ValueError):
            _compose_command("cmd", env={"X; rm -rf /": "x"})


class TestWrapInShell:
    def test_simple_command(self):
        assert _wrap_in_shell("echo hi") == "/bin/sh -c 'echo hi'"

    def test_shell_metacharacters_preserved(self):
        assert _wrap_in_shell("echo one; echo two | wc -l") == "/bin/sh -c 'echo one; echo two | wc -l'"

    def test_double_quotes_ok(self):
        assert _wrap_in_shell('echo "a b"') == "/bin/sh -c 'echo \"a b\"'"

    def test_single_quote_switches_to_double_quote_wrapper(self):
        # The docs pattern: single quotes ride inside a double-quoted wrapper.
        assert _wrap_in_shell("echo it's fine") == '/bin/sh -c "echo it\'s fine"'

    def test_double_quote_wrapper_escapes_double_quotes(self):
        assert _wrap_in_shell("echo 'a' \"b\"") == '/bin/sh -c "echo \'a\' \\"b\\""'

    def test_double_quote_wrapper_escapes_backslashes(self):
        # Tokenizer consumes one escaping level in double quotes: \\ -> \.
        assert _wrap_in_shell(r"echo 'a\tb'") == "/bin/sh -c \"echo 'a\\\\tb'\""

    def test_custom_shell(self):
        assert _wrap_in_shell("echo hi", shell="/bin/bash") == "/bin/bash -c 'echo hi'"

    def test_custom_shell_double_quote_path(self):
        assert _wrap_in_shell("echo it's", shell="/bin/bash") == '/bin/bash -c "echo it\'s"'
