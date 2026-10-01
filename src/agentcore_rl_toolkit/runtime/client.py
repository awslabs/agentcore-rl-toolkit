"""Async HTTP invocations and recoverable execution handles.

Install ``agentcore-rl-toolkit[rollout]`` to use this client. One client shares
its AWS connection pool across sessions on the same asyncio event loop.
"""

import asyncio
import json
import uuid
from typing import Any

from botocore.config import Config

from agentcore_rl_toolkit.concurrency.rate_limiter import RateLimiter

from .protocol import ENVELOPE_KEY, VERSION, InvocationRequest, InvocationState, state

POLL_INTERVAL = 10.0


class InvocationError(RuntimeError):
    """A remote invocation failed, was interrupted, or could not be found."""

    def __init__(self, session_id: str, snapshot: InvocationState):
        self.session_id = session_id
        self.invocation_id = snapshot["invocation_id"]
        self.snapshot = snapshot
        detail = snapshot.get("error", snapshot["status"])
        super().__init__(f"Invocation {self.invocation_id}: {detail}")


class InvocationHandle:
    """A handle for one invocation.

    ``result(timeout=...)`` limits local waiting.
    A handle obtained from an ordinary JSON response holds a completed local
    result. Only RIP invocations support reconstruction via ``get_invocation``.
    """

    def __init__(
        self,
        client: "AgentCoreHttpClient",
        session_id: str,
        invocation_id: str,
        snapshot: InvocationState | None = None,
    ):
        self.client = client
        self.session_id = session_id
        self.invocation_id = invocation_id
        self._snapshot = snapshot

    async def status(self) -> InvocationState:
        """Return the invocation state, reusing an already completed result."""
        if self._snapshot is None or self._snapshot["status"] != "completed":
            value = await self.client._request(
                self.session_id,
                {ENVELOPE_KEY: {"version": VERSION, "operation": "get", "invocation_id": self.invocation_id}},
            )
            self._snapshot = _invocation_state(value, self.invocation_id)
        return self._snapshot

    async def result(self, timeout: float | None = None) -> Any:
        """Wait for and return the original JSON result, or raise InvocationError."""
        async with asyncio.timeout(timeout):
            snapshot = self._snapshot
            while True:
                if snapshot is None:
                    snapshot = await self.status()
                if snapshot["status"] != "in_progress":
                    if snapshot["status"] == "completed" and "result" in snapshot:
                        return snapshot["result"]
                    raise InvocationError(self.session_id, snapshot)
                await asyncio.sleep(POLL_INTERVAL)
                snapshot = await self.status()


class AgentCoreHttpClient:
    """Invoke HTTP agents using a shared asynchronous AWS client.

    Foreground calls return the agent's result. Background calls return a handle.
    An ordinary BedrockAgentCoreApp may ignore the protocol envelope and return
    JSON directly; in that case a background call waits for that response and
    returns an already completed handle. Streaming responses are not supported.

    Session IDs and application configuration belong to the caller. Put any
    configuration in the payload's ``_config`` field.
    An optional ``request_rate_limiter`` gates invoke, get, and stop requests.
    """

    def __init__(
        self,
        runtime_arn: str,
        *,
        max_pool_connections: int = 100,
        read_timeout: float = 900,
        request_rate_limiter: RateLimiter | None = None,
    ):
        self.runtime_arn = runtime_arn
        self._request_rate_limiter = request_rate_limiter
        self._config = Config(
            max_pool_connections=max_pool_connections,
            read_timeout=read_timeout,
            user_agent_extra="agentcore-rl-toolkit",
        )
        self._aws_client = None
        self._aws_context = None
        self._open_lock = asyncio.Lock()

    async def __aenter__(self) -> "AgentCoreHttpClient":
        await self._aws()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.aclose()

    async def _aws(self):
        async with self._open_lock:
            if self._aws_client is None:
                from agentcore_rl_toolkit.aws_tools.boto3_tools import get_aioboto3_session

                context = (await get_aioboto3_session()).client(
                    "bedrock-agentcore", region_name=self.runtime_arn.split(":")[3], config=self._config
                )
                self._aws_client = await context.__aenter__()
                self._aws_context = context
            return self._aws_client

    async def aclose(self) -> None:
        """Close the connection pool after all callers have finished using it."""
        async with self._open_lock:
            if self._aws_context is not None:
                await self._aws_context.__aexit__(None, None, None)
                self._aws_client = self._aws_context = None

    async def invoke(
        self,
        payload: dict,
        *,
        session_id: str,
        invocation_id: str | None = None,
        background: bool = False,
    ) -> Any:
        """Retain invocation_id to retry or reattach after a lost response."""
        invocation_id = invocation_id if invocation_id is not None else uuid.uuid4().hex
        envelope = {
            "version": VERSION,
            "operation": "start",
            "invocation_id": invocation_id,
            "background": background,
        }
        InvocationRequest.parse(envelope)
        if ENVELOPE_KEY in payload:
            raise ValueError(f"{ENVELOPE_KEY} is reserved for the HTTP client")
        value = await self._request(session_id, {**payload, ENVELOPE_KEY: envelope})
        # Match this submission's identity, not generic application keys such as status.
        if (
            isinstance(value, dict)
            and {"version", "status", "invocation_id"} <= value.keys()
            and value["invocation_id"] == invocation_id
        ):
            snapshot = _invocation_state(value, invocation_id)
        else:
            snapshot = {**state(invocation_id, "completed"), "result": value}
        handle = InvocationHandle(self, session_id, invocation_id, snapshot)
        return handle if background else await handle.result()

    def get_invocation(self, *, session_id: str, invocation_id: str) -> InvocationHandle:
        """Reconstruct a RIP handle without submitting work or making a network call."""
        InvocationRequest.parse({"version": VERSION, "operation": "get", "invocation_id": invocation_id})
        return InvocationHandle(self, session_id, invocation_id)

    async def stop_session(self, session_id: str) -> None:
        """Stop the specified Runtime session."""
        client = await self._aws()
        if self._request_rate_limiter is not None:
            await self._request_rate_limiter.wait_async()
        await client.stop_runtime_session(agentRuntimeArn=self.runtime_arn, runtimeSessionId=session_id)

    async def _request(self, session_id: str, payload: dict) -> Any:
        body = json.dumps(payload, allow_nan=False).encode()
        client = await self._aws()
        if self._request_rate_limiter is not None:
            await self._request_rate_limiter.wait_async()
        response = await client.invoke_agent_runtime(
            agentRuntimeArn=self.runtime_arn,
            runtimeSessionId=session_id,
            contentType="application/json",
            accept="application/json",
            payload=body,
        )
        async with response["response"] as stream:
            raw = await stream.read()
        if response["statusCode"] != 200:
            raise RuntimeError(f"InvokeAgentRuntime returned HTTP {response['statusCode']}: {raw.decode()}")
        return json.loads(raw)


def _invocation_state(value: Any, invocation_id: str) -> InvocationState:
    if (
        not isinstance(value, dict)
        or not isinstance(value.get("version"), int)
        or isinstance(value["version"], bool)
        or value["version"] != VERSION
        or value.get("invocation_id") != invocation_id
        or value.get("status") not in ("in_progress", "completed", "interrupted", "not_found")
    ):
        raise ValueError(f"Invalid invocation response for {invocation_id}")
    if value["status"] == "completed":
        if ("result" in value) == ("error" in value) or ("error" in value and not isinstance(value["error"], str)):
            raise ValueError(f"Invalid completed response for {invocation_id}")
    elif "result" in value or "error" in value:
        raise ValueError(f"Unexpected terminal output for {invocation_id}")
    return value
