"""Sync client for running shell commands in AgentCore Runtime sandbox sessions.

A "sandbox" is an AgentCore Runtime session whose container runs the
``agentcore-sandboxd`` daemon (see ``sandboxd/`` at the repo root). Commands
use Runtime Invocation Protocol start/get requests through InvokeAgentRuntime.
The daemon owns execution and persists results independently of the connection.

Usage:
    from agentcore_rl_toolkit.sandbox import SandboxClient

    client = SandboxClient(runtime_arn="arn:aws:bedrock-agentcore:...")
    with client.start() as sb:
        result = sb.exec("cd /app && pytest -q", timeout=900)
        print(result.exit_code, result.stdout)
"""

from __future__ import annotations

import json
import logging
import re
import shlex
import time
import uuid

import boto3
from botocore.config import Config

from .types import ExecResult, SandboxProtocolError

logger = logging.getLogger(__name__)

# runtimeSessionId constraints from the bedrock-agentcore service model.
_SESSION_ID_MIN_LEN = 33
_SESSION_ID_MAX_LEN = 256

_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _compose_command(command: str, cwd: str = None, env: dict = None) -> str:
    """Compose cwd/env into a shell command string.

    Commands are stateless — each invocation runs in a fresh
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
    # Exit before any part of the user command can run if setup fails.
    if cwd is not None:
        prefix += f"cd {shlex.quote(cwd)} || exit $?; "
    if env:
        for key in env:
            if not _ENV_KEY_RE.match(key):
                raise ValueError(f"Invalid environment variable name: {key!r}")
        exports = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in env.items())
        prefix += f"export {exports} || exit $?; "
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
        read_timeout: Per-request socket read timeout in seconds. Defaults to
            900s so a long, silent command (a heavy verifier, or a runtime still
            warming up) is not cut off by boto3's short default.
        connect_timeout: TCP connect timeout in seconds.
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
        read_timeout: int = 900,
        connect_timeout: int = 15,
        shell: str = "/bin/sh",
    ):
        self.runtime_arn = runtime_arn
        self.region = region or self._parse_region_from_arn(runtime_arn)
        self.qualifier = qualifier
        self.shell = shell

        config = Config(
            retries={"max_attempts": max_retry_attempts, "mode": "adaptive"},
            max_pool_connections=max_pool_connections,
            read_timeout=read_timeout,
            connect_timeout=connect_timeout,
        )
        self._client = boto3.client("bedrock-agentcore", region_name=self.region, config=config)

    def start(self, session_id: str = None) -> Sandbox:
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

    def attach(self, session_id: str) -> Sandbox:
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
                f"session_id must be {_SESSION_ID_MIN_LEN}-{_SESSION_ID_MAX_LEN} characters, got {len(session_id)}"
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
        self,
        command: str,
        timeout: int = None,
        cwd: str = None,
        env: dict = None,
        shell: str = None,
        *,
        background: bool = False,
        invocation_id: str = None,
    ) -> ExecResult | ExecHandle:
        """Run a managed command, waiting by default or returning a background handle.

        Commands run in a fresh shell (default /bin/sh). ``cwd`` and ``env`` are
        established per call. Nonzero exits return an ``ExecResult``. Execution
        timeouts raise ``ExecTimeoutError`` with the captured result in ``.result``
        and the execution handle in ``.handle``. Each output stream retains its
        first 256 KiB; the result explicitly marks truncation.

        ``timeout`` is the daemon-enforced execution deadline in seconds
        (1–3600, default 300), independent of ``ExecHandle.result(timeout=...)``.
        It includes output waiting after shell exit. Expiry kills only the
        direct process and stops reading output; child processes may survive.
        ``invocation_id`` defaults to a UUID generated before the request. Reuse
        it to address an existing execution; a repeated start ignores the new
        command while its record survives. Foreground and background executions
        use the same manager and persist the same result.

        ``ExecError.handle`` permits recovery after an ambiguous submission or
        foreground connection failure. Session termination remains explicit
        (including context-manager exit). Requires a RIP-capable sandboxd image.
        """
        self._ensure_active()
        handle = self.get_exec(invocation_id if invocation_id is not None else str(uuid.uuid4()))
        body = {
            "command": _compose_command(command, cwd=cwd, env=env),
            "shell": shell or self._client.shell,
        }
        if timeout is not None:
            body["timeout"] = timeout
        try:
            response = self._invoke_execution("start", handle.invocation_id, background=background, **body)
        except Exception as exc:
            raise ExecError("start/wait failed; use this handle to recover the execution", handle) from exc
        if background:
            return handle
        if response["status"] == "in_progress":
            return handle.result()
        return handle._result_from_state(response)

    def get_exec(self, invocation_id: str) -> ExecHandle:
        """Reconstruct a command handle without a network call or starting work."""
        self._ensure_active()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", invocation_id):
            raise ValueError(
                "invocation_id must be 1–128 letters, digits, dots, underscores or hyphens, starting alphanumeric"
            )
        return ExecHandle(self, invocation_id)

    def _ensure_active(self):
        if self._terminated:
            raise RuntimeError(
                f"Sandbox {self.session_id[:8]}... is terminated; use SandboxClient.start() for a new session"
            )

    def _invoke_execution(self, operation: str, invocation_id: str, **body) -> dict:
        self._ensure_active()
        config = {"version": 1, "operation": operation, "invocation_id": invocation_id}
        if operation == "start":
            config["background"] = body.pop("background")
        response = self._client._client.invoke_agent_runtime(
            agentRuntimeArn=self._client.runtime_arn,
            runtimeSessionId=self.session_id,
            qualifier=self._client.qualifier,
            payload=json.dumps({"_agentcore_runtime": config, **body}),
        )
        stream = response["response"]
        try:
            raw = stream.read()
        finally:
            stream.close()
        try:
            result = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise SandboxProtocolError(f"Non-JSON execution response: {raw[:500]!r}") from None
        if (
            not isinstance(result, dict)
            or result.get("version") != 1
            or result.get("invocation_id") != invocation_id
            or result.get("status") not in {"in_progress", "completed", "interrupted", "not_found"}
        ):
            raise SandboxProtocolError(
                f"Unexpected execution response (rebuild the image with a RIP-capable sandboxd): {raw[:500]!r}"
            )
        return result

    def terminate(self):
        """Terminate the sandbox session (idempotent, best-effort).

        Two steps, both best-effort: release the session hold via
        ``{"action": "stop"}``, then call ``StopRuntimeSession``. If the stop API
        fails, the daemon becomes Healthy once active commands finish publishing
        results, allowing idle reaping. Failures are logged as warnings, never
        raised — this must be safe to call from ``__exit__``.
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

    def __enter__(self) -> Sandbox:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.terminate()


class ExecHandle:
    """One command, identified by its sandbox session and invocation ID.

    Save both IDs to recover with ``client.attach(session_id).get_exec(invocation_id)``.
    Records survive only while the daemon's configured filesystem survives.
    """

    def __init__(self, sandbox: Sandbox, invocation_id: str):
        self._sandbox = sandbox
        self.invocation_id = invocation_id
        self.session_id = sandbox.session_id

    def status(self) -> str:
        """Return in_progress, completed, interrupted, or not_found."""
        return self._sandbox._invoke_execution("get", self.invocation_id)["status"]

    def result(self, timeout: float | None = None) -> ExecResult:
        """Poll until a terminal result is available.

        ``timeout`` limits local waiting; it never stops the command or session.
        The deadline is checked between requests. In-flight AWS requests remain
        subject to the client's socket timeout and retry configuration.
        ``TimeoutError`` leaves this handle usable for a later wait.
        A completed execution that exceeded its deadline raises
        ``ExecTimeoutError`` with the persisted result in ``.result``.
        """
        if timeout is not None and timeout < 0:
            raise ValueError("timeout must be nonnegative")
        deadline = None if timeout is None else time.monotonic() + timeout
        interval = 0.1
        while True:
            response = self._sandbox._invoke_execution("get", self.invocation_id)
            if response["status"] != "in_progress":
                return self._result_from_state(response)
            delay = interval
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"Invocation {self.invocation_id} is still running")
                delay = min(delay, remaining)
            time.sleep(delay)
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(f"Invocation {self.invocation_id} is still running")
            interval = min(interval * 2, 2.0)

    def _result_from_state(self, response: dict) -> ExecResult:
        status = response["status"]
        if status != "completed":
            raise ExecError(status, self)
        if "error" in response:
            error = response["error"]
            raise ExecError(f"{error['code']}: {error['message']}", self)
        result = ExecResult(**response["result"])
        if result.timed_out:
            raise ExecTimeoutError(self, result)
        return result


class ExecError(RuntimeError):
    """Execution could not be submitted, recovered, or completed by sandboxd.

    ``handle`` retains the invocation identity, including when an initial network
    failure leaves submission ambiguous. The original transport error, if any,
    is available as ``__cause__``. Nonzero command exits remain ``ExecResult`` data.
    """

    def __init__(self, message: str, handle: ExecHandle):
        super().__init__(f"Invocation {handle.invocation_id}: {message}")
        self.handle = handle


class ExecTimeoutError(ExecError):
    """The execution deadline expired, including while waiting for output.

    ``result`` retains captured output, the exit code, and truncation flags.
    ``handle`` identifies the completed invocation; reading it again raises
    this exception with the same persisted result.
    """

    def __init__(self, handle: ExecHandle, result: ExecResult):
        super().__init__("execution timed out", handle)
        self.result = result
