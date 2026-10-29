"""Serve SageMaker MTRL rollouts through an entrypoint written for the ``_config`` contract.

``@sagemaker_mtrl`` lets one ``AgentCoreRuntimeApp`` entrypoint serve every rollout caller:

* Payloads carrying ``_config`` (the slime, verl, and Tinker backends) reach the handler unchanged.
* SageMaker MTRL requests, ``{"prompt": <row JSON>, "metadata": {...}, "inferenceParams": {...}}``,
  reach it as ``{**row, "_config": {"base_url", "model_id", "api_key"}}``. ``base_url`` addresses a
  loopback forwarder that adds SageMaker's bearer token and trajectory headers and applies
  ``inferenceParams``. The outcome is reported through the SDK's ``RolloutFeedbackClient``.

SageMaker knowledge lives in ``_MtrlAdapter`` and ``_MtrlRollout``, which leave metadata spellings,
headers, tokens, and feedback to the SDK; the forwarder and the wrapper are protocol-independent.
See ``designs/sagemaker_mtrl.md`` for the decisions. Requires the ``sagemaker_mtrl`` extra only when
a SageMaker request arrives.
"""

from __future__ import annotations

import functools
import inspect
import json
import logging
import math
import re
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)

#: SageMaker's RFT Runtime Service endpoints: ``https://job-runtime[.<stage>].sagemaker.<region>.api.aws``.
SAGEMAKER_ENDPOINT = r"https://job-runtime\.(?:[a-z0-9-]+\.)?sagemaker\.[a-z0-9-]+\.api\.aws"
# The inference headers SageMaker attributes a trajectory by; `make_inference_headers` builds them.
_TRAJECTORY_HEADERS = frozenset({"X-Amzn-SageMaker-Job-Arn", "X-Amzn-SageMaker-Trajectory-Id"})
# Each `inferenceParams` field and the OpenAI request field it sets.
_SAMPLING_FIELDS = (("temperature", "temperature"), ("topP", "top_p"), ("maxTokens", "max_tokens"))
# The token is a SigV4 presign, so it stops verifying when the credentials that signed it expire,
# whatever its own lifetime. Regenerating well inside any role session keeps it valid.
_TOKEN_REFRESH_SEC = 900.0
_HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)
_DROPPED_REQUEST_HEADERS = _HOP_BY_HOP | {"host", "authorization", "content-length"}
_DROPPED_RESPONSE_HEADERS = _HOP_BY_HOP | {"content-length"}


def sagemaker_mtrl(
    handler: Callable | None = None,
    *,
    model_id: str = "default",
    row: Callable[[dict], dict] | None = None,
    trusted_endpoint: str = SAGEMAKER_ENDPOINT,
) -> Callable:
    """Let a ``_config`` entrypoint also serve SageMaker MTRL rollouts.

    Apply it beneath ``@app.entrypoint``. The handler keeps its signature, including a ``context``
    parameter. On SageMaker requests it must return a dict with a finite scalar ``reward``, or a
    non-empty ``turn_rewards`` list of finite numbers, which is reported instead when present.
    ``status: "error"``, any other result, or an exception fails the trajectory with reward 0.

    Args:
        handler: a synchronous or ``async`` function returning a dict.
        model_id: the ``_config.model_id`` given to SageMaker rollouts.
        row: ``payload -> row`` for SageMaker requests; defaults to ``json.loads(payload["prompt"])``.
        trusted_endpoint: regular expression ``metadata["endpoint"]`` must fully match.
    """
    if handler is None:
        return functools.partial(sagemaker_mtrl, model_id=model_id, row=row, trusted_endpoint=trusted_endpoint)
    if inspect.isgeneratorfunction(handler) or inspect.isasyncgenfunction(handler):
        raise TypeError("sagemaker_mtrl requires a handler that returns a dict, not a generator")
    adapter = _MtrlAdapter(model_id=model_id, row=row or _prompt_row, trusted_endpoint=re.compile(trusted_endpoint))

    # `functools.wraps` keeps the handler's signature visible, so bedrock-agentcore still passes
    # `context` when the handler declares it; the wrappers forward whatever arguments arrive.
    if inspect.iscoroutinefunction(handler):

        @functools.wraps(handler)
        async def async_wrapper(payload: Any, *args: Any) -> Any:
            if not adapter.is_request(payload):
                return await handler(payload, *args)
            rollout = adapter.rollout(payload)
            try:
                result = await handler(rollout.start(), *args)
            except Exception as exc:
                rollout.report_failure(str(exc))
                raise
            finally:
                rollout.close()
            return rollout.report(result)

        return async_wrapper

    @functools.wraps(handler)
    def wrapper(payload: Any, *args: Any) -> Any:
        if not adapter.is_request(payload):
            return handler(payload, *args)
        rollout = adapter.rollout(payload)
        try:
            result = handler(rollout.start(), *args)
        except Exception as exc:
            rollout.report_failure(str(exc))
            raise
        finally:
            rollout.close()
        return rollout.report(result)

    return wrapper


def _prompt_row(payload: dict) -> dict:
    return json.loads(payload["prompt"])


@dataclass(frozen=True)
class _MtrlAdapter:
    """SageMaker MTRL's request shape and endpoint trust, for one decorated handler."""

    model_id: str
    row: Callable[[dict], dict]
    trusted_endpoint: re.Pattern

    def is_request(self, payload: Any) -> bool:
        # `_config` callers are excluded first, so a dataset row's own `metadata` cannot select this path.
        if not isinstance(payload, dict) or "_config" in payload:
            return False
        return isinstance(payload.get("prompt"), str) and isinstance(payload.get("metadata"), dict)

    def rollout(self, payload: dict) -> _MtrlRollout:
        """Accept one request; raises, before anything is reported, unless its endpoint is trusted.

        ``metadata["endpoint"]`` serves both inference and feedback. The SDK's feedback client
        prefers it over any other source, so both reach this host. Both carry a token signed with
        the Runtime's credentials, so an untrusted endpoint receives neither.
        """
        endpoint = payload["metadata"].get("endpoint")
        if not isinstance(endpoint, str) or not endpoint.strip():
            raise ValueError("SageMaker MTRL metadata names no endpoint")
        endpoint = endpoint.rstrip("/")
        if not self.trusted_endpoint.fullmatch(endpoint):
            raise ValueError(
                f"SageMaker MTRL endpoint {endpoint!r} does not match the trusted endpoint pattern "
                f"{self.trusted_endpoint.pattern!r}"
            )
        return _MtrlRollout(self, payload, endpoint)


class _MtrlRollout:
    """One SageMaker MTRL rollout: its translated payload, forwarder route, SDK context, and feedback.

    Uses only the SDK's public functions: ``set_rollout_context``, ``clear_rollout_context``,
    ``make_inference_headers``, ``generate_token``, and ``RolloutFeedbackClient``.
    """

    def __init__(self, adapter: _MtrlAdapter, payload: dict, endpoint: str) -> None:
        self._adapter = adapter
        self._payload = payload
        self._metadata = payload["metadata"]
        self._endpoint = endpoint
        # `None` lets `generate_token` fall back to AWS_REGION.
        self._region = self._metadata.get("region")
        self._key: str | None = None
        self._context_set = False

    def start(self) -> dict:
        """Return the ``_config`` payload, routed through the forwarder until ``close``."""
        from sagemaker.train.rft import make_inference_headers, set_rollout_context

        inference_params = self._payload.get("inferenceParams")
        # Agents that read the SDK's rollout context, e.g. through its framework adapters, keep working.
        set_rollout_context(self._metadata, inference_params)
        self._context_set = True
        row = self._adapter.row(self._payload)
        if not isinstance(row, dict):
            raise TypeError(f"SageMaker MTRL row must be a JSON object, got {type(row).__name__}")
        if "_config" in row:
            raise ValueError("SageMaker MTRL row must not contain reserved field '_config'")
        headers = make_inference_headers(self._metadata)
        if not _TRAJECTORY_HEADERS <= headers.keys():
            raise ValueError("SageMaker MTRL metadata must name both the job and the trajectory")
        _bearer(self._region)  # Fails here, not on the first model call, when no token can be made.
        sampling = sampling_params(inference_params)
        forwarder = _forwarder()
        self._key = forwarder.register(
            _Route(
                upstream=f"{self._endpoint}/v1",
                headers=headers,
                authorization=functools.partial(_bearer, self._region),
                transform=functools.partial(_with_sampling, sampling=sampling) if sampling else None,
            )
        )
        config = {"base_url": forwarder.base_url, "model_id": self._adapter.model_id, "api_key": self._key}
        return {**row, "_config": config}

    def close(self) -> None:
        if self._key is not None:
            _forwarder().unregister(self._key)
            self._key = None
        if self._context_set:
            from sagemaker.train.rft import clear_rollout_context

            clear_rollout_context()
            self._context_set = False

    def report(self, result: Any) -> Any:
        """Report the handler's result, and return what the app records.

        A feedback failure propagates, so the invocation record shows that SageMaker was not told.
        """
        if not isinstance(result, dict):
            error = f"entrypoint returned {type(result).__name__}, not a dict with a 'reward'"
            self.report_failure(error)
            raise TypeError(error)
        rewards, error = _rewards(result)
        if error is not None:
            self._feedback().report_error(error)
            return {**result, "status": "error", "error": error}
        feedback = self._feedback()
        feedback.complete_rollout("ready")
        feedback.update_reward(rewards)
        return result

    def report_failure(self, error: str) -> None:
        """Fail the trajectory with reward 0; a feedback failure is logged so ``error`` stays the cause."""
        try:
            self._feedback().report_error(error)
        except Exception:
            logger.exception("Failed to report SageMaker MTRL rollout failure")

    def _feedback(self):
        from sagemaker.train.rft import RolloutFeedbackClient

        return RolloutFeedbackClient(self._metadata)


def _rewards(result: dict) -> tuple[list[float] | None, str | None]:
    """``(rewards, None)`` to report, or ``(None, error)`` to fail the trajectory.

    ``turn_rewards`` is reported in place of ``reward``, so one result can carry the scalar ART's
    backends train on and the per-turn rewards SageMaker accepts.
    """
    if result.get("status") == "error":
        error = result.get("error")
        return None, error if isinstance(error, str) and error else "entrypoint returned status 'error'"
    if "turn_rewards" in result:
        turn_rewards = result["turn_rewards"]
        if isinstance(turn_rewards, list) and turn_rewards and all(map(_finite, turn_rewards)):
            return list(turn_rewards), None
        return None, (
            f"entrypoint returned invalid turn_rewards={turn_rewards!r}; expected a non-empty list of finite numbers"
        )
    if _finite(result.get("reward")):
        return [result["reward"]], None
    return None, f"entrypoint returned no finite scalar reward (reward={result.get('reward')!r})"


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def sampling_params(params: Any) -> dict[str, Any]:
    """OpenAI request fields for SageMaker's ``inferenceParams``."""
    if not isinstance(params, dict):
        return {}
    return {field: params[name] for name, field in _SAMPLING_FIELDS if params.get(name) is not None}


def _with_sampling(tail: str, body: bytes, *, sampling: dict[str, Any]) -> bytes:
    """Set ``sampling`` on an OpenAI completions request, overriding the handler's values."""
    if not tail.endswith("completions"):
        return body
    try:
        request = json.loads(body)
    except ValueError:
        return body
    if not isinstance(request, dict):
        return body
    for field, value in sampling.items():
        if field == "max_tokens" and "max_completion_tokens" in request:
            field = "max_completion_tokens"
        request[field] = value
    return json.dumps(request).encode()


_token_lock = threading.Lock()
_tokens: dict[str | None, tuple[float, str]] = {}


def _bearer(region: str | None) -> str:
    """``Authorization`` for SageMaker inference in ``region``, cached for ``_TOKEN_REFRESH_SEC``."""
    from sagemaker.core.token_generator import generate_token

    with _token_lock:
        cached = _tokens.get(region)
        if cached is None or time.monotonic() - cached[0] > _TOKEN_REFRESH_SEC:
            cached = _tokens[region] = (time.monotonic(), generate_token(region=region))
    return f"Bearer {cached[1]}"


@dataclass(frozen=True)
class _Route:
    """Where the forwarder sends one key's requests, and what it adds to them."""

    upstream: str
    headers: Mapping[str, str]
    #: Returns the upstream ``Authorization`` value; may block, so it runs off the serving loop.
    authorization: Callable[[], str]
    #: ``(path tail, body) -> body`` for each request, or ``None``.
    transform: Callable[[str, bytes], bytes] | None = None


class _Forwarder:
    """Forward ``/v1/*`` from a per-route key to that route's upstream, with its credentials.

    Listens on loopback only. The key in the bearer slot selects the route, as the rollout
    gateway's session id does; the upstream credential never reaches the caller.
    """

    def __init__(self) -> None:
        from aiohttp import web

        from agentcore_rl_toolkit.rollout_gateway.server import ThreadedGatewayServer

        # Written by handler threads and read on the serving loop; single dict operations are atomic.
        self._routes: dict[str, _Route] = {}
        self._session = None
        self.app = web.Application()
        self.app.router.add_route("*", "/v1/{tail:.*}", self._forward)
        self.app.on_startup.append(self._open)
        self.app.on_cleanup.append(self._close)
        self._server = ThreadedGatewayServer(self, host="127.0.0.1")
        self._server.start()

    @property
    def base_url(self) -> str:
        return self._server.base_url

    def register(self, route: _Route) -> str:
        key = uuid4().hex
        self._routes[key] = route
        return key

    def unregister(self, key: str) -> None:
        self._routes.pop(key, None)

    async def _open(self, app) -> None:
        import aiohttp

        # Encoded bodies pass through as sent; the caller's client decodes them.
        self._session = aiohttp.ClientSession(
            auto_decompress=False, timeout=aiohttp.ClientTimeout(total=None, sock_connect=30)
        )

    async def _close(self, app) -> None:
        await self._session.close()

    async def _forward(self, request):
        import asyncio

        from aiohttp import ClientError, web
        from multidict import CIMultiDict

        scheme, _, key = request.headers.get("Authorization", "").partition(" ")
        route = self._routes.get(key) if scheme.lower() == "bearer" else None
        if route is None:
            return web.json_response(
                {"error": {"message": "unknown rollout key", "type": "authentication_error"}}, status=401
            )
        tail = request.match_info["tail"]
        body = await request.read()
        if route.transform is not None and request.method == "POST":
            body = route.transform(tail, body)
        headers = CIMultiDict(
            (name, value) for name, value in request.headers.items() if name.lower() not in _DROPPED_REQUEST_HEADERS
        )
        headers.update(route.headers)
        headers["Authorization"] = await asyncio.to_thread(route.authorization)
        response = None
        try:
            async with self._session.request(
                request.method, f"{route.upstream}/{tail}", params=request.query, data=body, headers=headers
            ) as upstream:
                response = web.StreamResponse(
                    status=upstream.status,
                    reason=upstream.reason,
                    headers=CIMultiDict(
                        (name, value)
                        for name, value in upstream.headers.items()
                        if name.lower() not in _DROPPED_RESPONSE_HEADERS
                    ),
                )
                await response.prepare(request)
                async for chunk in upstream.content.iter_any():
                    await response.write(chunk)
                await response.write_eof()
                return response
        except ClientError as exc:
            if response is not None:
                raise  # Mid-stream: dropping the connection is the only signal left.
            return web.json_response({"error": {"message": f"upstream unreachable: {exc}"}}, status=502)


_forwarder_lock = threading.Lock()
_forwarder_instance: _Forwarder | None = None


def _forwarder() -> _Forwarder:
    global _forwarder_instance
    with _forwarder_lock:
        if _forwarder_instance is None:
            _forwarder_instance = _Forwarder()
        return _forwarder_instance
