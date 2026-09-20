"""Data types for the sandbox SDK."""

from dataclasses import dataclass


@dataclass
class ExecResult:
    """Result of a completed sandbox command.

    Nonzero exit codes and timeouts are normal results, not errors — eval and RL
    workloads expect failing commands (that is the reward signal). Exceptions are
    reserved for infrastructure failures (throttling, session not found, network).

    Attributes:
        exit_code: The command's exit code. ``-1`` means it was killed by a signal.
        stdout: Accumulated standard output.
        stderr: Accumulated standard error.
        timed_out: True if the command hit its execution deadline. Partial output
            produced before the timeout is retained.
        stdout_truncated: Output exceeded the daemon's per-stream limit.
        stderr_truncated: Error output exceeded the daemon's per-stream limit.
    """

    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool
    stdout_truncated: bool = False
    stderr_truncated: bool = False


class SandboxProtocolError(RuntimeError):
    """Unexpected response or stream from the sandbox runtime.

    Raised when the deployed container does not behave like agentcore-sandboxd
    (e.g. an agent image or an older sandboxd was deployed instead).
    """
