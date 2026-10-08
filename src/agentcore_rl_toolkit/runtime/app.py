"""An ordinary HTTP AgentCore app with optional addressable invocations."""

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import uuid4

from bedrock_agentcore.runtime import BedrockAgentCoreApp
from starlette.responses import JSONResponse

from .protocol import ENVELOPE_KEY, InvocationRequest, InvocationState, state
from .store import InvocationStore


class AgentCoreRuntimeApp(BedrockAgentCoreApp):
    """Add persisted start/get invocations to ``@app.entrypoint``.

    Set ``background=True`` to run requests without
    ``_agentcore_runtime`` in the background with server-generated IDs.
    Otherwise upstream HTTP behavior applies; explicit envelopes follow the protocol.
    Managed invocations require a JSON-serializable return value, and persist
    results in ``state_dir`` (default: ``.agentcore_runtime`` under the OS temp
    directory). Use a managed mount for records that survive compute replacement.

    Run one app process per Runtime session. Session IDs namespace
    records; they do not add an access-control boundary to a shared filesystem.
    Input payloads are not persisted automatically. Ordinary requests and
    invocation completion never stop the Runtime session.
    """

    def __init__(self, *, background: bool = False, state_dir: str | Path | None = None, **kwargs):
        super().__init__(**kwargs)
        self._background = background
        self._store = InvocationStore(state_dir)
        self._lock = asyncio.Lock()
        self._live: dict[str, asyncio.Task] = {}
        self._requests: set[asyncio.Task] = set()

    async def _handle_invocation(self, http_request):
        try:
            payload = await http_request.json()
        except ValueError:
            return await super()._handle_invocation(http_request)
        if not isinstance(payload, dict) or (ENVELOPE_KEY not in payload and not self._background):
            return await super()._handle_invocation(http_request)
        if ENVELOPE_KEY in payload:
            try:
                request = InvocationRequest.parse(payload[ENVELOPE_KEY])
            except ValueError as exc:
                return JSONResponse({"error": str(exc)}, status_code=400)
            application_payload = {key: value for key, value in payload.items() if key != ENVELOPE_KEY}
        else:
            request = InvocationRequest(operation="start", invocation_id=uuid4().hex, background=True)
            application_payload = payload
        context = self._build_request_context(http_request)

        handler = self.handlers.get("main")
        if handler is None:
            return JSONResponse({"error": "No entrypoint defined"}, status_code=500)
        # Own acceptance independently of the HTTP connection, including file I/O:
        # cancellation must not split a persisted start from task registration.
        task = asyncio.create_task(
            self._dispatch(request, handler, context, self._takes_context(handler), application_payload)
        )
        self._requests.add(task)
        task.add_done_callback(self._requests.discard)
        task.add_done_callback(self._observe_failure)
        try:
            return JSONResponse(await asyncio.shield(task))
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)

    async def _dispatch(
        self,
        request: InvocationRequest,
        handler: Callable,
        context: Any,
        takes_context: bool,
        payload: dict,
    ) -> InvocationState:
        session_id = context.session_id or ""
        invocation_id = request.invocation_id
        # Only acceptance and lookup are serialized; handlers run concurrently.
        async with self._lock:
            # Keep the task reference if completion removes it while the store is read.
            execution = self._live.get(invocation_id)
            snapshot = await asyncio.to_thread(self._store.read, session_id, invocation_id)
            if snapshot is None and execution is not None:
                snapshot = state(invocation_id, "in_progress")
            if snapshot is not None:
                if snapshot["status"] == "in_progress" and execution is None:
                    snapshot = state(invocation_id, "interrupted")
            elif request.operation == "get":
                snapshot = state(invocation_id, "not_found")
            else:
                await asyncio.to_thread(self._store.start, session_id, invocation_id)
                busy_id = self.add_async_task(getattr(handler, "__name__", "invocation"))
                execution = asyncio.create_task(
                    self._execute(invocation_id, busy_id, handler, context, takes_context, payload)
                )
                self._live[invocation_id] = execution
                execution.add_done_callback(self._observe_failure)
                snapshot = state(invocation_id, "in_progress")

        if request.operation == "start" and not request.background and execution is not None:
            return await asyncio.shield(execution)
        return snapshot

    async def _execute(
        self,
        invocation_id: str,
        busy_id: int,
        handler: Callable,
        context: Any,
        takes_context: bool,
        payload: dict,
    ) -> InvocationState:
        session_id = context.session_id or ""
        terminal = state(invocation_id, "completed")
        try:
            try:
                terminal["result"] = await super()._invoke_handler(handler, context, takes_context, payload)
                await asyncio.to_thread(self._store.finish, session_id, terminal)
            except Exception as exc:
                self.logger.exception("Invocation %s failed", invocation_id)
                terminal = state(invocation_id, "completed")
                terminal["error"] = str(exc)
                # A serialization/write failure may still permit saving a small error.
                # If that also fails, propagate; never report an unpersisted success.
                await asyncio.to_thread(self._store.finish, session_id, terminal)
            return terminal
        finally:
            self._live.pop(invocation_id, None)
            self.complete_async_task(busy_id)

    def _observe_failure(self, task: asyncio.Task) -> None:
        # Detached requests/executions still need their exceptions consumed and logged.
        if not task.cancelled() and (error := task.exception()) is not None:
            self.logger.error("Runtime invocation operation failed", exc_info=(type(error), error, error.__traceback__))
