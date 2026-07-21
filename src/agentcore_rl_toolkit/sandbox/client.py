"""Sync client for running shell commands in AgentCore Runtime sandbox sessions.

A "sandbox" is an AgentCore Runtime session whose container runs the
``agentcore-sandboxd`` health shim (see ``sandboxd/`` at the repo root). The
shim only manages the Healthy/HealthyBusy ping state; command execution goes
through AgentCore Runtime's native ``InvokeAgentRuntimeCommand`` API, which runs
shell commands inside the same session/container.

Usage:
    from agentcore_rl_toolkit.sandbox import SandboxClient

    client = SandboxClient(runtime_arn="arn:aws:bedrock-agentcore:...")
    with client.start() as sb:
        result = sb.exec("cd /app && pytest -q", timeout=900)
        print(result.exit_code, result.stdout)
"""

import json
import logging
import re
import shlex
import uuid

import boto3
from botocore.config import Config

from .types import ExecResult, SandboxProtocolError

logger = logging.getLogger(__name__)

# runtimeSessionId constraints from the bedrock-agentcore service model.
_SESSION_ID_MIN_LEN = 33
_SESSION_ID_MAX_LEN = 256

_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _wrap_in_shell(command: str, shell: str = "/bin/sh") -> str:
    """Wrap a shell command string for InvokeAgentRuntimeCommand.

    The command API does NOT interpret the command through a shell — it
    word-splits argv-style, passing ``;``, ``|``, ``$VAR`` etc. through as
    literal arguments (verified against the live service; every example in the
    official docs likewise wraps commands, in ``/bin/bash -c "..."``:
    https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-execute-command.html).
    Shell semantics therefore require an explicit ``<shell> -c`` wrapper.

    We default to ``/bin/sh`` rather than the docs' ``/bin/bash`` because the
    sandbox use case targets arbitrary images: POSIX sh exists in any image
    with a shell at all (including busybox/alpine), while bash is often absent
    from minimal images.

    Wrapper quoting, per the service tokenizer's semantics (live-verified):

    - Single-quoted args pass through fully verbatim (backslashes untouched),
      but POSIX quote concatenation (``'it'"'"'s'``) is NOT supported — so the
      single-quote wrapper only fits commands without single quotes.
    - Double-quoted args support ``\\"`` and ``\\\\`` escapes, and the tokenizer
      does NOT expand ``$`` or backticks (they reach the inner shell, which
      expands them — the semantics we want). Escaping ``\\`` and ``"`` therefore
      round-trips any command exactly.

    Both forms keep the wire format human-readable — the service logs the input
    command to the runtime's CloudWatch log group for auditing.
    """
    if "'" not in command:
        return f"{shell} -c '{command}'"
    escaped = command.replace("\\", "\\\\").replace('"', '\\"')
    return f'{shell} -c "{escaped}"'


def _compose_command(command: str, cwd: str = None, env: dict = None) -> str:
    """Compose cwd/env into a shell command string.

    Commands are stateless — each ``InvokeAgentRuntimeCommand`` runs in a fresh
    shell, so working directory and environment variables must be re-established
    per call. The user command is appended raw (it is already a shell string);
    only ``cwd`` and env values are quoted.

    Args:
        command: The shell command to run.
        cwd: Working directory to ``cd`` into first.
        env: Environment variables to export first. Keys must be valid shell
            identifiers; values are shell-quoted.

    Returns:
        The composed command string.

    Raises:
        ValueError: If an env key is not a valid shell identifier.
    """
    prefix = ""
    if cwd is not None:
        prefix += f"cd {shlex.quote(cwd)} && "
    if env:
        for key in env:
            if not _ENV_KEY_RE.match(key):
                raise ValueError(f"Invalid environment variable name: {key!r}")
        exports = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in env.items())
        # && (not ;) so a failed cd cannot fall through to running the command
        # in the wrong directory with a clean exit code.
        prefix += f"export {exports} && "
    return prefix + command


class SandboxClient:
    """Client bound to a deployed sandbox runtime (data plane).

    Manages per-session lifecycle against an AgentCore runtime that was already
    created from a sandboxd-wrapped image: ``start()`` a new sandbox session,
    ``attach()`` to a live one, and run commands via ``Sandbox.exec()``.

    Args:
        runtime_arn: ARN of the deployed AgentCore runtime.
        region: AWS region. Defaults to the region parsed from ``runtime_arn``.
        qualifier: Runtime endpoint qualifier.
        max_retry_attempts: Max boto3 retry attempts (adaptive mode).
        max_pool_connections: Max boto3 connection pool size.
        shell: Shell used to interpret ``exec()`` commands in the container.
            Defaults to ``/bin/sh`` (present in any image with a shell,
            including busybox/alpine); set to ``/bin/bash`` if your image has
            bash and your commands use bashisms. Overridable per call via
            ``Sandbox.exec(shell=...)``.
    """

    @staticmethod
    def _parse_region_from_arn(arn: str) -> str:
        """Extract AWS region from an ARN.

        ARN format: arn:partition:service:region:account-id:resource-type/resource-id

        Args:
            arn: The ARN to parse

        Returns:
            The region string (e.g., "us-west-2")

        Raises:
            ValueError: If the ARN format is invalid
        """
        parts = arn.split(":")
        if len(parts) < 4 or not parts[3]:
            raise ValueError(f"Invalid ARN format, cannot extract region: {arn}")
        return parts[3]

    def __init__(
        self,
        runtime_arn: str,
        region: str = None,
        qualifier: str = "DEFAULT",
        max_retry_attempts: int = 5,
        max_pool_connections: int = 10,
        shell: str = "/bin/sh",
    ):
        self.runtime_arn = runtime_arn
        self.region = region or self._parse_region_from_arn(runtime_arn)
        self.qualifier = qualifier
        self.shell = shell

        config = Config(
            retries={"max_attempts": max_retry_attempts, "mode": "adaptive"},
            max_pool_connections=max_pool_connections,
        )
        self._client = boto3.client("bedrock-agentcore", region_name=self.region, config=config)

    def start(self, session_id: str = None) -> "Sandbox":
        """Start a new sandbox session and flip its ping state to HealthyBusy.

        Sends ``{"action": "start"}`` to the runtime's ``/invocations`` endpoint.
        The first invocation with a fresh session id provisions the microVM.

        Args:
            session_id: Session id to use. Defaults to a generated UUID (the
                service requires 33-256 characters; a UUID string is 36).

        Returns:
            A ``Sandbox`` handle, usable as a context manager (``__exit__``
            calls ``terminate()``).

        Raises:
            SandboxProtocolError: If the runtime's response is not the expected
                sandboxd handshake — usually a sign that a non-sandboxd image
                was deployed to this runtime.
        """
        session_id = session_id or str(uuid.uuid4())
        response = self._client.invoke_agent_runtime(
            agentRuntimeArn=self.runtime_arn,
            runtimeSessionId=session_id,
            qualifier=self.qualifier,
            payload=json.dumps({"action": "start"}),
        )
        raw = response["response"].read()
        try:
            body = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise SandboxProtocolError(
                f"Sandbox start returned a non-JSON response (is a sandboxd image deployed?): {raw[:500]!r}"
            ) from None
        if not isinstance(body, dict) or body.get("status") != "ok":
            raise SandboxProtocolError(f"Unexpected sandbox start response: {body!r}")
        logger.info(f"Started sandbox session {session_id[:8]}...")
        return Sandbox(self, session_id)

    def attach(self, session_id: str) -> "Sandbox":
        """Attach to a live sandbox session without invoking the runtime.

        Args:
            session_id: Id of an existing session (33-256 characters).

        Returns:
            A ``Sandbox`` handle for the session.

        Raises:
            ValueError: If ``session_id`` does not satisfy the service's length
                constraints.
        """
        if not (_SESSION_ID_MIN_LEN <= len(session_id) <= _SESSION_ID_MAX_LEN):
            raise ValueError(
                f"session_id must be {_SESSION_ID_MIN_LEN}-{_SESSION_ID_MAX_LEN} characters, " f"got {len(session_id)}"
            )
        return Sandbox(self, session_id)


class Sandbox:
    """Handle to one sandbox session. Create via ``SandboxClient.start()`` or ``attach()``.

    Usable as a context manager: ``__exit__`` calls ``terminate()`` and never
    suppresses exceptions.
    """

    def __init__(self, client: SandboxClient, session_id: str):
        self._client = client
        self.session_id = session_id
        self._terminated = False

    def exec(
        self, command: str, timeout: int = None, cwd: str = None, env: dict = None, shell: str = None
    ) -> ExecResult:
        """Run a shell command in the sandbox and wait for it to complete.

        Nonzero exit codes and timeouts are returned as data on ``ExecResult``,
        never raised. Infrastructure failures raise: ``botocore`` ``ClientError``
        (throttling, session not found) from the call itself,
        ``EventStreamError`` if botocore surfaces an in-stream error event as an
        exception, or ``SandboxProtocolError`` if an error event arrives as a
        stream member or the stream ends without a result.

        Commands are stateless — each call runs in a fresh shell. Use ``cwd``
        and ``env`` to re-establish state per call; they are composed into the
        command string (``cd ... && export ... && <command>``).

        The command is interpreted by a shell (default ``/bin/sh``): pipes,
        ``;``, variable expansion and command substitution all work. (On the
        wire the SDK wraps it in ``<shell> -c`` because the command API itself
        does not invoke a shell — see ``_wrap_in_shell``.)

        Args:
            command: Shell command string to execute.
            timeout: Server-enforced timeout in seconds (1-3600, service default
                300). On expiry the command is terminated and the result has
                ``timed_out=True`` with any partial output.
            cwd: Working directory for the command.
            env: Environment variables for the command. Keys must be valid shell
                identifiers.
            shell: Shell for this command. Defaults to the client's ``shell``
                (``/bin/sh`` unless configured otherwise).

        Returns:
            An ``ExecResult`` with exit code, accumulated stdout/stderr, and the
            timeout flag.

        Raises:
            RuntimeError: If the sandbox has been terminated. (After
                ``StopRuntimeSession`` the session id remains valid, so an
                unguarded exec would silently provision a fresh microVM and run
                against empty state instead of failing.)
        """
        if self._terminated:
            raise RuntimeError(
                f"Sandbox {self.session_id[:8]}... is terminated; use SandboxClient.start() for a new session"
            )
        composed = _compose_command(command, cwd=cwd, env=env)
        body = {"command": _wrap_in_shell(composed, shell=shell or self._client.shell)}
        if timeout is not None:
            body["timeout"] = timeout

        response = self._client._client.invoke_agent_runtime_command(
            agentRuntimeArn=self._client.runtime_arn,
            runtimeSessionId=self.session_id,
            qualifier=self._client.qualifier,
            body=body,
        )

        stdout_parts = []
        stderr_parts = []
        exit_code = None
        status = None
        for event in response["stream"]:
            chunk = event.get("chunk")
            if chunk is None:
                # Error events (throttlingException, runtimeClientError, ...) can
                # arrive as stream members; botocore may instead raise
                # EventStreamError, which propagates from the iteration above.
                raise SandboxProtocolError(f"Error event in command stream: {event!r}")
            if "contentDelta" in chunk:
                delta = chunk["contentDelta"]
                if delta.get("stdout"):
                    stdout_parts.append(delta["stdout"])
                if delta.get("stderr"):
                    stderr_parts.append(delta["stderr"])
            elif "contentStop" in chunk:
                exit_code = chunk["contentStop"]["exitCode"]
                status = chunk["contentStop"]["status"]
            # contentStart: ignore

        if exit_code is None:
            raise SandboxProtocolError("Command stream ended without a result (no contentStop event)")

        return ExecResult(
            exit_code=exit_code,
            stdout="".join(stdout_parts),
            stderr="".join(stderr_parts),
            timed_out=(status == "TIMED_OUT"),
        )

    def terminate(self):
        """Terminate the sandbox session (idempotent, best-effort).

        Two steps, both best-effort: flip the ping state back to Healthy via
        ``{"action": "stop"}`` (so that even if the subsequent stop call fails,
        the idle reaper collects the session within the idle timeout), then call
        ``StopRuntimeSession``. Failures are logged as warnings, never raised —
        this must be safe to call from ``__exit__``.
        """
        if self._terminated:
            return
        self._terminated = True

        try:
            response = self._client._client.invoke_agent_runtime(
                agentRuntimeArn=self._client.runtime_arn,
                runtimeSessionId=self.session_id,
                qualifier=self._client.qualifier,
                payload=json.dumps({"action": "stop"}),
            )
            response["response"].read()
        except Exception as e:
            logger.warning(f"Failed to send stop action to sandbox {self.session_id[:8]}...: {e}")

        try:
            self._client._client.stop_runtime_session(
                agentRuntimeArn=self._client.runtime_arn,
                runtimeSessionId=self.session_id,
                qualifier=self._client.qualifier,
                clientToken=str(uuid.uuid4()),
            )
            logger.info(f"Terminated sandbox session {self.session_id[:8]}...")
        except Exception as e:
            logger.warning(f"Failed to stop runtime session {self.session_id[:8]}...: {e}")

    def __enter__(self) -> "Sandbox":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.terminate()
