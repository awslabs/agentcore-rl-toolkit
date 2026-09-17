"""AgentCore data plane: starting, invoking and stopping one session, once per rollout.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any

import backoff
from botocore.config import Config
from pydantic import BaseModel

from agentcore_rl_toolkit.aws_tools.boto3_tools import get_aioboto3_session

logger = logging.getLogger(__name__)

# One client is shared by every rollout of a worker process (see `shared_agentcore_client`),
# so its pool sizes that whole worker's ACR concurrency rather than one rollout's. A rollout
# holds at most one connection at a time (its phases are sequential), so this covers the
# example configs' `container_concurrency: 256` with room for the teardown calls beside it;
# botocore's default of 10 would queue session starts behind the pool. Connections open
# lazily, so unused headroom costs nothing.
MAX_POOL_CONNECTIONS = 512

SESSION_CLIENT_CONFIG = Config(
    retries={"max_attempts": 16, "mode": "standard"},
    max_pool_connections=MAX_POOL_CONNECTIONS,
)

# The warm-up in `start_agentcore_session` races the container's own start-up: ACR creates
# the microVM and the container does boot, but a command dispatched before the app is serving
# comes back as `RuntimeClientError: Received error (500) from runtime`. Measured at ~25% of
# sessions against a heavy (Java toolchain) agent image and ~1% against a light one -- and the
# container is alive either way, so retrying against it has always succeeded.
#
# The retry has to live here rather than in the client's config: `RuntimeClientError` is
# modelled `httpStatusCode: 424, senderFault: true`, so botocore reads it as the caller's
# fault and `SESSION_CLIENT_CONFIG`'s `max_attempts` never fires on it. Sized to sit far
# inside any realistic `container_setup_timeout`: three tries of a ~25s call, 2s apart, where
# the measured recovery was one retry ~2.4s later.
START_MAX_TRIES = 3
START_RETRY_INTERVAL = 2.0

# Only that readiness class is retried. Every other way a start fails is either already
# retried by botocore (throttling, 5xx *service* errors) or permanent for this rollout
# (validation, access denied, quota), where more tries would only multiply the burst.
START_RETRYABLE_ERRORS = frozenset({"RuntimeClientError"})


class HttpResponse(BaseModel):
    status_code: int
    body: str


def region_of(arn: str) -> str:
    return arn.split(":")[3]


@asynccontextmanager
async def agentcore_client(region_name: str, config: Config = SESSION_CLIENT_CONFIG):
    """One ``bedrock-agentcore`` data-plane client, open for the whole block."""
    async with (await get_aioboto3_session()).client(
        "bedrock-agentcore", region_name=region_name, config=config
    ) as acr:  # type: ignore
        yield acr


# --- the process-wide data-plane client ---------------------------------------
#
# Rollout sessions are constructed one per trajectory, so a client per session means a
# client and a connection pool per rollout. Every session of a process shares the one
# below instead. Keyed by running loop as well as region: a client's aiohttp connector
# belongs to the loop that created it, and the agent-loop worker, the driver and each test
# each have their own loop.

_shared_clients: dict[tuple[asyncio.AbstractEventLoop, str], Any] = {}
_shared_contexts: dict[tuple[asyncio.AbstractEventLoop, str], Any] = {}
_shared_locks: dict[asyncio.AbstractEventLoop, asyncio.Lock] = {}


async def shared_agentcore_client(region_name: str) -> Any:
    """The process-wide ``bedrock-agentcore`` client for ``region_name``, opened on first use.

    Shared by every caller on this event loop and closed by none of them, so it must not be
    used as a context manager -- :func:`close_shared_agentcore_clients` ends its life.
    Always built with :data:`SESSION_CLIENT_CONFIG`; a caller wanting different settings
    owns a client of its own via :func:`agentcore_client`.
    """
    loop = asyncio.get_running_loop()
    key = (loop, region_name)
    client = _shared_clients.get(key)
    if client is not None:
        return client
    async with _lock_for(loop):
        client = _shared_clients.get(key)
        if client is None:
            # Entered by hand, because this client outlives the call that first needed it.
            # The context is kept so the close below can unwind it the ordinary way.
            context = (await get_aioboto3_session()).client(
                "bedrock-agentcore", region_name=region_name, config=SESSION_CLIENT_CONFIG
            )
            client = await context.__aenter__()
            _shared_clients[key] = client
            _shared_contexts[key] = context
            logger.info("opened the shared bedrock-agentcore client for %s", region_name)
        return client


def _lock_for(loop: asyncio.AbstractEventLoop) -> asyncio.Lock:
    """The client-creation lock for ``loop``.

    Without it the first burst of rollouts would each open a client and all but one would be
    dropped unclosed. One lock per loop because an ``asyncio.Lock`` binds to the loop that
    first awaits it; the miss and the store below have no await between them, so the lookup
    cannot race with itself.
    """
    lock = _shared_locks.get(loop)
    if lock is None:
        lock = _shared_locks[loop] = asyncio.Lock()
    return lock


async def close_shared_agentcore_clients() -> None:
    """Close the shared clients of the *current* loop; the next call reopens one.

    For tests and for a worker that ends its loop's life while the process continues --
    another loop's client can only be closed on that loop. Leaving one open at process exit
    is harmless beyond aiohttp's "unclosed connector" warning, but a test that closes its
    loop with one still open leaks it into the next test.
    """
    loop = asyncio.get_running_loop()
    _shared_locks.pop(loop, None)
    for key in [k for k in _shared_contexts if k[0] is loop]:
        context = _shared_contexts.pop(key)
        _shared_clients.pop(key, None)
        await context.__aexit__(None, None, None)


@asynccontextmanager
async def _client_for(arn: str, client: Any | None):
    """The caller's client, or one opened just for this call if it passed none.

    The per-call client is for standalone use -- a dev script or a cleanup job calling one
    helper. In-process callers that make more than one call (every rollout session) pass
    :func:`shared_agentcore_client` instead.
    """
    if client is not None:
        yield client
        return
    async with agentcore_client(region_of(arn)) as acr:
        yield acr


def is_runtime_readiness_error(e: BaseException) -> bool:
    """Whether ``e`` is the runtime-not-ready-yet failure a warm-up may be retried through.

    Matched by error code, which reaches us two ways: as the modelled
    ``botocore.errorfactory.RuntimeClientError`` when the initial response carries it (what
    every observed failure was), or wrapped in an ``EventStreamError`` when it arrives as an
    error event mid-stream. Both are ``ClientError`` subclasses, so the code is on the
    response; the class-name check covers the first without importing botocore's factory.
    """
    if type(e).__name__ in START_RETRYABLE_ERRORS:
        return True
    response = getattr(e, "response", None)
    if not isinstance(response, dict):
        return False
    return response.get("Error", {}).get("Code") in START_RETRYABLE_ERRORS


@backoff.on_exception(
    backoff.constant,
    Exception,
    giveup=lambda e: not is_runtime_readiness_error(e),
    # Read through a callable so both are resolved per call, not baked in at decoration:
    # tests turn the interval off, and nothing has to reach inside the decorator to do it.
    max_tries=lambda: START_MAX_TRIES,
    interval=lambda: START_RETRY_INTERVAL,
    jitter=None,  # the interval is the measured recovery time, not a herd-spreading guess
    logger=logger,
)
async def _warm_up_agentcore_session(runtime_arn: str, session_id: str, acr: Any) -> None:
    """Run one trivial command in ``session_id``'s microVM, creating it if it does not exist.

    Retried through the readiness race only (see :data:`START_RETRYABLE_ERRORS`), always
    against the same ``session_id``: the microVM the failed try created is the one that is
    now warm, so a retry is a cheap second dispatch rather than another cold start.
    """
    resp = await acr.invoke_agent_runtime_command(
        agentRuntimeArn=runtime_arn,
        runtimeSessionId=session_id,
        contentType="application/json",
        accept="application/json",
        body={"command": "echo hello", "timeout": 60},
    )
    last_chunk = None
    async for chunk in resp["stream"]:
        last_chunk = chunk
    assert last_chunk is not None, f"The agent runtime streamed no chunks for {session_id=}"
    assert last_chunk["chunk"]["contentStop"]["exitCode"] == 0, "Failed to start the agent runtime"


async def start_agentcore_session(runtime_arn: str, session_id: str, client: Any | None = None):
    async with _client_for(runtime_arn, client) as acr:
        await _warm_up_agentcore_session(runtime_arn, session_id, acr)


async def invoke_agentcore_session(
    runtime_arn: str, session_id: str, payload: bytes, client: Any | None = None
) -> HttpResponse:
    async with _client_for(runtime_arn, client) as acr:
        resp = await acr.invoke_agent_runtime(
            agentRuntimeArn=runtime_arn, runtimeSessionId=session_id, contentType="application/json", payload=payload
        )
        async with resp["response"] as body:
            return HttpResponse(status_code=resp["statusCode"], body=(await body.read()).decode())


async def stop_agentcore_microvm_session(runtime_arn: str, session_id: str, client: Any | None = None) -> bool:
    """End ``session_id``'s microVM through its runtime. Best-effort; returns whether it did.

    The data-plane stop, for a session owned by nothing but the rollout that started it --
    :func:`stop_agentcore_instance_session` instead releases the session through the *capacity
    provider* whose instance pool backs it. Never raises: a stop happens on the way out of a
    rollout, often carrying the error that ended it, and a teardown failure that replaced that
    error would cost the diagnosis to save a session ACR's idle reaper collects anyway.
    """
    try:
        async with _client_for(runtime_arn, client) as acr:
            await acr.stop_runtime_session(agentRuntimeArn=runtime_arn, runtimeSessionId=session_id)
        return True
    except Exception as e:
        logger.warning("failed to stop the runtime session %s: %s: %s", session_id, type(e).__name__, e)
        return False


async def stop_agentcore_instance_session(capacity_provider_arn: str, session_id: str, client: Any | None = None):
    """Release ``session_id`` through the capacity provider whose instance pool backs it.

    For sessions placed on a pooled instance, where the pool -- not the rollout -- owns the
    slot, so failing to delete the session leaks one instance until its idle timeout.
    :func:`stop_agentcore_microvm_session` is the plain runtime stop for unpooled sessions.
    An already-deleted session is not an error; anything else raises.
    """
    capacity_provider_id = capacity_provider_arn.split("/")[-1]
    try:
        async with _client_for(capacity_provider_arn, client) as acr:
            await acr.delete_capacity_provider_session(capacityProviderId=capacity_provider_id, sessionId=session_id)
    except Exception as e:
        if type(e).__name__ == "ResourceNotFoundException":
            # already deleted
            return
        else:
            raise e


@asynccontextmanager
async def agentcore_session(capacity_provider_arn: str, runtime_arn: str, session_id: str, client: Any | None = None):
    assert region_of(capacity_provider_arn) == region_of(runtime_arn)
    async with _client_for(runtime_arn, client) as acr:
        await start_agentcore_session(runtime_arn, session_id, acr)
        yield
        await stop_agentcore_instance_session(capacity_provider_arn, session_id, acr)
