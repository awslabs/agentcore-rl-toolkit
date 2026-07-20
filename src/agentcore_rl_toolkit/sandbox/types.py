"""Data types for the sandbox SDK."""

from dataclasses import dataclass


@dataclass
class ExecResult:
    """Result of a completed sandbox command.

    Nonzero exit codes and timeouts are normal results, not errors — eval and RL
    workloads expect failing commands (that is the reward signal). Exceptions are
    reserved for infrastructure failures (throttling, session not found, network).

    Attributes:
        exit_code: The command's exit code, passed through verbatim from the
            service (``contentStop.exitCode``). ``-1`` indicates a platform error.
        stdout: Accumulated standard output.
        stderr: Accumulated standard error.
        timed_out: True if the command hit its server-enforced timeout
            (``status == "TIMED_OUT"``). Partial output produced before the
            timeout is retained in ``stdout``/``stderr``.
    """

    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool


class SandboxProtocolError(RuntimeError):
    """Unexpected response or stream from the sandbox runtime.

    Raised when the deployed container does not behave like agentcore-sandboxd
    (e.g. an agent image was deployed instead), when an error event arrives
    in a command stream, or when a command stream ends without a result.
    """
