"""The client half of the A2A rollout protocol: drive one task to its result."""

from __future__ import annotations

import logging
import re
import uuid

import backoff
import httpx
from a2a.client import Client, ClientCallContext, ClientConfig, ClientFactory, minimal_agent_card
from a2a.client.errors import A2AError
from a2a.types import (
    CancelTaskRequest,
    GetTaskRequest,
    SendMessageConfiguration,
    SendMessageRequest,
    Task,
    TaskState,
)
from a2a.utils import TransportProtocol

from agentcore_rl_toolkit.aws_tools.persistent_dict import PersistentDict
from agentcore_rl_toolkit.rollout_session.a2a_protocol import (
    build_task_input_message,
    dump_from_artifacts,
    status_message_text,
)
from agentcore_rl_toolkit.rollout_session.wire import RolloutDumpResponse

logger = logging.getLogger(__name__)

SESSION_HEADER = "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id"

# ACR surfaces two retryable conditions as JSON-RPC error codes: a prior session op still
# running (`RetryableConflictException`, -32054) and throttling (-32053). Both must be
# recognised and retried; every other JSON-RPC error is fatal for this rollout.
SESSION_IN_PROGRESS_CODE = -32054
THROTTLE_CODE = -32053
# The message shape a2a-sdk uses for an HTTP-200 JSON-RPC error body (codes it has no typed
# error for, which includes both ACR codes above): "JSON-RPC Error <code>: <message>".
_JSONRPC_CODE_IN_MESSAGE = re.compile(r"^JSON-RPC Error (-?\d+):")
CONFLICT_MAX_TRIES = 8
CONFLICT_INTERVAL = 2.0
THROTTLE_MAX_TRIES = 6

POLL_INTERVAL = 10.0

_TERMINAL = frozenset(
    {
        TaskState.TASK_STATE_COMPLETED,
        TaskState.TASK_STATE_FAILED,
        TaskState.TASK_STATE_CANCELED,
        TaskState.TASK_STATE_REJECTED,
    }
)


class RolloutA2AError(RuntimeError):
    """A JSON-RPC error the client cannot recover from (anything not classified retryable)."""


class RetryableConflictError(RuntimeError):
    """ACR reported ``-32054`` "session operation in progress"; retry the same call."""


class RetryableThrottleError(RuntimeError):
    """ACR reported ``-32053`` throttling; back off and retry the same call."""


def _is_conflict(exc: BaseException) -> bool:
    return isinstance(exc, RetryableConflictError)


def _is_throttle(exc: BaseException) -> bool:
    return isinstance(exc, RetryableThrottleError)


def _jsonrpc_code(exc: A2AError) -> int | None:
    """The JSON-RPC error code ACR reported, dug out of wherever a2a-sdk left it."""
    cause = exc.__cause__
    if isinstance(cause, httpx.HTTPStatusError):
        try:
            body = cause.response.json()
        except Exception:
            body = None
        if isinstance(body, dict) and isinstance(body.get("error"), dict):
            code = body["error"].get("code")
            if isinstance(code, int):
                return code
    match = _JSONRPC_CODE_IN_MESSAGE.match(str(exc))
    return int(match.group(1)) if match else None


def _http_status(exc: A2AError) -> int | None:
    cause = exc.__cause__
    return cause.response.status_code if isinstance(cause, httpx.HTTPStatusError) else None


def _reraise(exc: A2AError) -> None:
    """Re-raise an a2a-sdk error as a retryable conflict/throttle or a fatal client error."""
    code = _jsonrpc_code(exc)
    status = _http_status(exc)
    # Prefer the JSON-RPC code (specific), then the HTTP status (in case ACR ever returns a
    # non-2xx with a body we can't decode).
    if code == SESSION_IN_PROGRESS_CODE or status == 409:
        raise RetryableConflictError(str(exc)) from exc
    if code == THROTTLE_CODE or status == 429:
        raise RetryableThrottleError(str(exc)) from exc
    raise RolloutA2AError(str(exc)) from exc


def _context(session_id: str) -> ClientCallContext:
    """Per-call context carrying the runtime-session-id header ACR uses to pin the call to this session's VM."""
    return ClientCallContext(state={}, service_parameters={SESSION_HEADER: session_id})


def build_a2a_client(httpx_client, url: str) -> Client:
    """A non-streaming, polling JSON-RPC client over ``httpx_client`` to ``url`` (minimal card, no fetch)."""
    # httpx logs every request at INFO; with polling that is one line per tick, so quiet it.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    config = ClientConfig(
        streaming=False,
        polling=True,
        httpx_client=httpx_client,
        supported_protocol_bindings=[TransportProtocol.JSONRPC],
    )
    return ClientFactory(config).create(minimal_agent_card(url, [TransportProtocol.JSONRPC]))


def _retry_transient(func):
    """Retry ACR's two transient conditions: conflicts (constant) and throttling (expo+jitter).

    Stacked so each classified error is handled by exactly one loop; every other
    ``RolloutA2AError`` gives up immediately and propagates.
    """
    func = backoff.on_exception(
        backoff.constant,
        RetryableConflictError,
        max_tries=CONFLICT_MAX_TRIES,
        interval=CONFLICT_INTERVAL,
        giveup=lambda e: not _is_conflict(e),
        jitter=None,
        logger=logger,
    )(func)
    return backoff.on_exception(
        backoff.expo,
        RetryableThrottleError,
        max_tries=THROTTLE_MAX_TRIES,
        giveup=lambda e: not _is_throttle(e),
        jitter=backoff.full_jitter,
        logger=logger,
    )(func)


@_retry_transient
async def _send_message(client: Client, message, session_id: str) -> Task:
    # return_immediately=True: handler returns the snapshot at once, runs the rollout in the background.
    request = SendMessageRequest(
        message=message,
        configuration=SendMessageConfiguration(return_immediately=True, accepted_output_modes=["text"]),
    )
    task: Task | None = None
    try:
        async for event in client.send_message(request, context=_context(session_id)):
            if event.HasField("task"):
                task = event.task
    except A2AError as e:
        _reraise(e)
    if task is None:
        raise RolloutA2AError("message/send returned no task")
    return task


@_retry_transient
async def _get_task(client: Client, task_id: str, session_id: str) -> Task:
    try:
        return await client.get_task(GetTaskRequest(id=task_id), context=_context(session_id))
    except A2AError as e:
        _reraise(e)
        raise  # unreachable; _reraise always raises


def _message(task_input: dict, *, session_id: str, task_id: str | None):
    return build_task_input_message(task_input, context_id=session_id, task_id=task_id, message_id=uuid.uuid4().hex)


async def send_setup(client: Client, task_input: dict, *, session_id: str) -> str:
    """Start a new task with the setup payload; return the server-minted task id to resume with."""
    task = await _send_message(client, _message(task_input, session_id=session_id, task_id=None), session_id)
    return task.id


@backoff.on_predicate(
    backoff.constant,
    lambda t: t.status.state not in _TERMINAL and t.status.state != TaskState.TASK_STATE_INPUT_REQUIRED,
    interval=POLL_INTERVAL,
    logger=None,  # expected polling, not a retry worth logging on every tick
)
async def wait_input_required(client: Client, task_id: str, *, session_id: str) -> Task:
    """Poll until setup parks the task at ``input-required`` (or it ends early)."""
    return await _get_task(client, task_id, session_id)


async def send_rollout(client: Client, task_id: str, task_input: dict, *, session_id: str) -> None:
    """Resume the parked task with the rollout payload (back to ``working``)."""
    await _send_message(client, _message(task_input, session_id=session_id, task_id=task_id), session_id)


@backoff.on_predicate(backoff.constant, lambda t: t.status.state not in _TERMINAL, interval=POLL_INTERVAL, logger=None)
async def _wait_terminal_task(client: Client, task_id: str, *, session_id: str) -> Task:
    return await _get_task(client, task_id, session_id)


async def wait_terminal(client: Client, task_id: str, *, session_id: str) -> RolloutDumpResponse:
    """Poll until the task ends, then read its artifact or failure reason into a dump; only client failures raise."""
    task = await _wait_terminal_task(client, task_id, session_id=session_id)
    if task.status.state == TaskState.TASK_STATE_COMPLETED:
        dump = dump_from_artifacts(task.artifacts)
        if dump is not None:
            return dump
        return RolloutDumpResponse(task_output=None, reward=None, exception="task completed with no rollout artifact")
    reason = status_message_text(task.status.message) or (
        f"rollout task ended in state {TaskState.Name(task.status.state)}"
    )
    return RolloutDumpResponse(task_output=None, reward=None, exception=reason)


async def cancel_task(client: Client, task_id: str, *, session_id: str) -> None:
    """Best-effort cancel so the agent can clean up; never blocks teardown."""
    try:
        await client.cancel_task(CancelTaskRequest(id=task_id), context=_context(session_id))
    except Exception as e:
        logger.warning("cancel of task %s failed: %s: %s", task_id, type(e).__name__, e)


class A2ARolloutSession:
    """Base for the ACR and Docker A2A sessions: the shared setup/run half.

    Subclasses provide the transport (:meth:`_client`) and teardown (:meth:`shutdown`).
    Session identity is the ACR runtime-session-id, reused as the A2A ``contextId``.
    """

    def __init__(self, session_id: str, session_state: PersistentDict):
        self.session_id = session_id
        self.session_state = session_state
        self._task_id: str | None = None
        # Set when `run` returns a terminal dump; teardown cancels only unfinished tasks.
        self._finished = False

    async def _client(self) -> Client:  # pragma: no cover - overridden
        raise NotImplementedError

    async def __aenter__(self) -> "A2ARolloutSession":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.shutdown()

    async def setup(self, task: dict) -> None:
        client = await self._client()
        self._task_id = await send_setup(client, task, session_id=self.session_id)
        await self.session_state.set("a2a_task_id", self._task_id)
        parked = await wait_input_required(client, self._task_id, session_id=self.session_id)
        if parked.status.state != TaskState.TASK_STATE_INPUT_REQUIRED:
            reason = status_message_text(parked.status.message) or TaskState.Name(parked.status.state)
            raise RolloutA2AError(f"setup did not reach input-required (state {reason})")

    async def run(self, task: dict) -> RolloutDumpResponse:
        assert self._task_id is not None, "run called before setup"
        client = await self._client()
        await send_rollout(client, self._task_id, task, session_id=self.session_id)
        dump = await wait_terminal(client, self._task_id, session_id=self.session_id)
        self._finished = True
        return dump

    async def _cancel_if_unfinished(self) -> None:
        """Best-effort cancel of a task torn down mid-flight; skipped on the happy path."""
        if self._task_id is None or self._finished:
            return
        await cancel_task(await self._client(), self._task_id, session_id=self.session_id)

    async def shutdown(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError
