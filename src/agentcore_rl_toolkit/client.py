"""Client for invoking agents and collecting rollouts via S3 polling."""

import asyncio
import json
import logging
import time
import traceback
import uuid
from concurrent.futures import CancelledError
from dataclasses import dataclass

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


def _format_exception(exc: BaseException) -> str:
    """Format an exception with its full traceback."""
    return "".join(traceback.format_exception(exc))


class ACRRateLimiter:
    """Unified rate limiter for all ACR API calls (invoke + stop).

    Since invoke_agent_runtime and stop_runtime_session share a single
    per-runtime-ARN rate limit on the ACR service, all calls must go
    through the same limiter to avoid throttling.

    Provides both sync and async interfaces. The async interface uses
    an asyncio.Lock to serialize timing checks without blocking the
    event loop during the sleep interval.
    """

    def __init__(self, tps_limit: int = 25):
        self.tps_limit = tps_limit
        self._min_interval = 1.0 / tps_limit
        self._last_call_time = 0.0
        # Async lock and its event loop (lazily created)
        self._async_lock = None
        self._async_lock_loop = None

    def _get_async_lock(self) -> asyncio.Lock:
        """Lazily create and return the async rate-limiting lock.

        Detects when the running event loop has changed (e.g., due to a new
        ``asyncio.run()`` call) and recreates the lock for the current loop.
        """
        loop = asyncio.get_running_loop()
        if self._async_lock is None or self._async_lock_loop is not loop:
            self._async_lock = asyncio.Lock()
            self._async_lock_loop = loop
        return self._async_lock

    def wait_sync(self):
        """Block until the next call is allowed under the TPS limit."""
        now = time.time()
        elapsed = now - self._last_call_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call_time = time.time()

    async def wait_async(self):
        """Async wait until the next call is allowed under the TPS limit.

        Uses a lock to serialize timing checks. The lock is held only during
        the timing check and sleep, so concurrent callers queue up properly.
        """
        async with self._get_async_lock():
            now = time.time()
            elapsed = now - self._last_call_time
            if elapsed < self._min_interval:
                await asyncio.sleep(self._min_interval - elapsed)
            self._last_call_time = time.time()


@dataclass
class BatchItem:
    """Result wrapper for batch execution, distinguishing success from error.

    Attributes:
        success: True if the request completed, False if it failed.
        result: The rollout result dict (populated when success=True).
        error: Error message (populated when success=False).
        index: Index of the payload in the original payloads list.
        elapsed: Time in seconds from request start to completion.
    """

    success: bool
    result: dict = None
    error: str = None
    index: int = None
    elapsed: float = None


class RolloutFuture:
    """Future representing an async rollout result, polled via S3 HEAD."""

    def __init__(
        self,
        s3_client,
        s3_bucket: str,
        result_key: str,
        initial_interval: float = 0.5,
        max_interval: float = 30.0,
        backoff_factor: float = 1.5,
        session_id: str = None,
        input_id: str = None,
        agentcore_client=None,
        agent_runtime_arn: str = None,
        rate_limiter: ACRRateLimiter = None,
    ):
        self.s3_client = s3_client
        self.s3_bucket = s3_bucket
        self.result_key = result_key
        self.session_id = session_id
        self.input_id = input_id
        self.agentcore_client = agentcore_client
        self.agent_runtime_arn = agent_runtime_arn
        self._rate_limiter = rate_limiter
        self._result = None
        self._done = False
        self._cancelled = False

        # Per-future backoff state
        self._poll_interval = initial_interval
        self._initial_interval = initial_interval
        self._max_interval = max_interval
        self._backoff_factor = backoff_factor
        self._last_poll_time = 0.0
        self._start_time = time.time()  # Track when future was created

    def done(self) -> bool:
        """Check if result is ready (non-blocking). Updates backoff state.

        Returns True if the future is in a terminal state: either the result
        is available in S3 or the future was cancelled.
        """
        if self._done or self._cancelled:
            return True
        try:
            self.s3_client.head_object(Bucket=self.s3_bucket, Key=self.result_key)
            self._done = True
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                # Update backoff state after each poll
                self._last_poll_time = time.time()
                self._poll_interval = min(self._poll_interval * self._backoff_factor, self._max_interval)
                return False
            raise

    def time_until_next_poll(self) -> float:
        """Returns seconds until this future should be polled again."""
        if self._done:
            return float("inf")
        elapsed = time.time() - self._last_poll_time
        return max(0, self._poll_interval - elapsed)

    def ready_to_poll(self) -> bool:
        """Returns True if enough time has passed since last poll."""
        return self.time_until_next_poll() <= 0

    def elapsed(self) -> float:
        """Returns seconds since this future was created."""
        return time.time() - self._start_time

    def cancel(self) -> bool:
        """Cancel the underlying ACR session (best-effort, rate-limited).

        Sets cancelled to True once called, even if the API call fails or
        the client/session_id are unavailable. Use ``cancelled`` to check
        whether cancellation was *attempted*, not whether the session was
        successfully stopped. Returns True only when the stop API call succeeds.
        """
        if self._cancelled:
            return False
        self._cancelled = True
        if not self.agentcore_client or not self.session_id:
            return False
        try:
            if self._rate_limiter:
                self._rate_limiter.wait_sync()
            self.agentcore_client.stop_runtime_session(
                agentRuntimeArn=self.agent_runtime_arn,
                runtimeSessionId=self.session_id,
            )
            logger.info(f"Stopped session {self.session_id[:8]}...")
            return True
        except Exception as e:
            logger.warning(f"Failed to stop session {self.session_id[:8]}...: {e}")
            return False

    async def cancel_async(self) -> bool:
        """Async version of ``cancel()`` with rate limiting.

        Uses the shared ACR rate limiter to avoid bursting stop calls
        that compete with invoke calls for the same service-side rate limit.
        """
        if self._cancelled:
            return False
        self._cancelled = True
        if not self.agentcore_client or not self.session_id:
            return False
        try:
            if self._rate_limiter:
                await self._rate_limiter.wait_async()
            await asyncio.to_thread(
                self.agentcore_client.stop_runtime_session,
                agentRuntimeArn=self.agent_runtime_arn,
                runtimeSessionId=self.session_id,
            )
            logger.info(f"Stopped session {self.session_id[:8]}...")
            return True
        except Exception as e:
            logger.warning(f"Failed to stop session {self.session_id[:8]}...: {e}")
            return False

    @property
    def cancelled(self) -> bool:
        """True if cancellation was attempted (may not have succeeded)."""
        return self._cancelled

    async def done_async(self) -> bool:
        """Check if result is ready (non-blocking, async). Updates backoff state.

        Like ``done()`` but wraps the S3 HEAD call in ``asyncio.to_thread()``
        so it doesn't block the event loop.
        """
        if self._done or self._cancelled:
            return True
        try:
            await asyncio.to_thread(self.s3_client.head_object, Bucket=self.s3_bucket, Key=self.result_key)
            self._done = True
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                self._last_poll_time = time.time()
                self._poll_interval = min(self._poll_interval * self._backoff_factor, self._max_interval)
                return False
            raise

    def _fetch_result(self) -> dict:
        """Fetch and parse the result from S3. Contains blocking I/O."""
        response = self.s3_client.get_object(Bucket=self.s3_bucket, Key=self.result_key)
        return json.loads(response["Body"].read())

    async def _async_poll(self) -> dict:
        """Poll until result is ready, using asyncio.sleep for backoff.

        Automatically cancels the ACR session once the result is fetched.
        """
        if self._result is not None:
            return self._result
        if self._cancelled:
            raise CancelledError("Future was cancelled")

        while True:
            if await self.done_async():
                self._result = await asyncio.to_thread(self._fetch_result)
                await self.cancel_async()
                return self._result
            await asyncio.sleep(self._poll_interval)

    def __await__(self):
        return self._async_poll().__await__()

    async def result_async(self, timeout: float = None) -> dict:
        """Async version of ``result()``.

        The underlying ACR session is automatically cancelled once the result
        is fetched, so callers don't need to manage session cleanup.

        Args:
            timeout: Max time to wait in seconds. If None, waits indefinitely.

        Returns:
            The rollout result dictionary from S3.

        Raises:
            CancelledError: If the future was cancelled before a result was available.
            TimeoutError: If timeout is reached before result is ready.
        """
        try:
            if timeout is not None:
                return await asyncio.wait_for(self._async_poll(), timeout=timeout)
            return await self
        except (TimeoutError, asyncio.TimeoutError):
            await self.cancel_async()
            raise

    def result(self, timeout: float = None) -> dict:
        """
        Block until result is ready, polling S3 HEAD with exponential backoff.

        Args:
            timeout: Max time to wait in seconds. If None, waits indefinitely
                until the result appears. For long-running tasks, consider
                setting a timeout to avoid infinite waits if the server fails
                to save the result.

        Returns:
            The rollout result dictionary from S3

        Raises:
            CancelledError: If the future was cancelled before a result was available.
            TimeoutError: If timeout is reached before result is ready.
        """
        if self._result is not None:
            return self._result
        if self._cancelled:
            raise CancelledError("Future was cancelled")

        start = time.time()

        while True:
            if self.done():
                self._result = self._fetch_result()
                self.cancel()
                return self._result

            if timeout and (time.time() - start) > timeout:
                self.cancel()
                raise TimeoutError(f"Result not ready after {timeout}s")

            # Use the per-future backoff interval
            time.sleep(self._poll_interval)


class RolloutClient:
    """Client for invoking agents and collecting rollouts with full lifecycle management.

    Provides both sync and async APIs. Sync methods (``invoke``, ``run_batch``)
    block the caller. Async methods (``invoke_async``, ``run_batch_async``) are
    suitable for use inside ``asyncio`` event loops.

    Note:
        This client is NOT thread-safe. Do not mix sync and async calls
        concurrently on the same instance. Use separate client instances for
        concurrent access from multiple threads.
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
        agent_runtime_arn: str,
        s3_bucket: str,
        exp_id: str,
        max_retry_attempts: int = 5,
        tps_limit: int = 25,
        max_pool_connections: int = 10,
        # Optional model inference config (for vLLM/SGLang servers)
        base_url: str = None,
        model_id: str = None,
        # Additional config passed through to _rollout (e.g., sampling params)
        **extra_config,
    ):
        """
        Initialize RolloutClient for invoking agents and collecting rollouts.

        Args:
            agent_runtime_arn: ARN of the ACR agent runtime (region is inferred from ARN)
            s3_bucket: S3 bucket for storing rollout results
            exp_id: Experiment ID for organizing results
            max_retry_attempts: Max retries for transient errors (default: 5)
            tps_limit: ACR invocation rate limit (default: 25)
            max_pool_connections: Max urllib3 connection pool size per boto3 client (default: 10).
                Set to at least the max number of concurrent requests to avoid
                "Connection pool is full, discarding connection" warnings.
            base_url: Optional vLLM/SGLang server URL
            model_id: Optional model ID for inference
            **extra_config: Additional config passed to _rollout (e.g., temperature, top_p)
        """
        self.agent_runtime_arn = agent_runtime_arn
        self.s3_bucket = s3_bucket
        self.exp_id = exp_id
        self.base_url = base_url
        self.model_id = model_id
        self.extra_config = extra_config
        self.tps_limit = tps_limit

        # Infer region from ARN (boto3 region must match resource region)
        self.region = self._parse_region_from_arn(agent_runtime_arn)

        # Configure boto3 with adaptive retry for 429/503
        config = Config(
            retries={"max_attempts": max_retry_attempts, "mode": "adaptive"},
            max_pool_connections=max_pool_connections,
        )
        self.agentcore_client = boto3.client("bedrock-agentcore", region_name=self.region, config=config)
        self.s3_client = boto3.client("s3", region_name=self.region, config=config)

        # Unified rate limiter for all ACR API calls (invoke + stop).
        # invoke_agent_runtime and stop_runtime_session share a single
        # per-runtime-ARN rate limit on the ACR service.
        self._rate_limiter = ACRRateLimiter(tps_limit)

    def _parse_response(self, response: dict) -> dict:
        """Parse ACR invocation response."""
        return json.loads(response["response"].read())

    def _build_full_payload(self, payload: dict, input_id: str, **overrides) -> dict:
        """Build the full payload with _rollout config.

        Merge order (last wins): required fields → client defaults → per-invocation overrides.
        """
        rollout_config = {
            "exp_id": self.exp_id,
            "input_id": input_id,
            "s3_bucket": self.s3_bucket,
            **self.extra_config,
        }
        if self.base_url:
            rollout_config["base_url"] = self.base_url
        if self.model_id:
            rollout_config["model_id"] = self.model_id
        rollout_config.update(overrides)
        return {**payload, "_rollout": rollout_config}

    def _rate_limited_invoke(self, payload: dict, session_id: str, input_id: str, **overrides) -> RolloutFuture:
        """Invoke with TPS rate limiting."""
        self._rate_limiter.wait_sync()

        full_payload = self._build_full_payload(payload, input_id, **overrides)

        # Invoke via boto3 with timing
        invoke_start = time.time()
        response = self.agentcore_client.invoke_agent_runtime(
            agentRuntimeArn=self.agent_runtime_arn,
            runtimeSessionId=session_id,
            payload=json.dumps(full_payload),
        )
        invoke_elapsed = time.time() - invoke_start
        logger.info(f"Invoked session {session_id[:8]}... in {invoke_elapsed:.1f}s")

        data = self._parse_response(response)
        return RolloutFuture(
            s3_client=self.s3_client,
            s3_bucket=data["s3_bucket"],
            result_key=data["result_key"],
            session_id=session_id,
            input_id=input_id,
            agentcore_client=self.agentcore_client,
            agent_runtime_arn=self.agent_runtime_arn,
            rate_limiter=self._rate_limiter,
        )

    async def _async_rate_limited_invoke(
        self, payload: dict, session_id: str, input_id: str, **overrides
    ) -> RolloutFuture:
        """Invoke with async TPS rate limiting.

        The rate limiter lock is held only during the timing check and released
        before the HTTP call, so cold starts on one request don't block
        submission of others.
        """
        await self._rate_limiter.wait_async()

        full_payload = self._build_full_payload(payload, input_id, **overrides)

        # Invoke via boto3 in a thread (includes response parsing which reads streaming body)
        def _invoke_and_parse():
            invoke_start = time.time()
            response = self.agentcore_client.invoke_agent_runtime(
                agentRuntimeArn=self.agent_runtime_arn,
                runtimeSessionId=session_id,
                payload=json.dumps(full_payload),
            )
            invoke_elapsed = time.time() - invoke_start
            logger.info(f"Invoked session {session_id[:8]}... in {invoke_elapsed:.1f}s")
            return self._parse_response(response)

        data = await asyncio.to_thread(_invoke_and_parse)
        return RolloutFuture(
            s3_client=self.s3_client,
            s3_bucket=data["s3_bucket"],
            result_key=data["result_key"],
            session_id=session_id,
            input_id=input_id,
            agentcore_client=self.agentcore_client,
            agent_runtime_arn=self.agent_runtime_arn,
            rate_limiter=self._rate_limiter,
        )

    def invoke(self, payload: dict, session_id: str = None, input_id: str = None, **rollout_overrides) -> RolloutFuture:
        """
        Single invocation, returns Future for the result.

        Args:
            payload: The payload to send to the agent
            session_id: Optional session ID (default: auto-generated UUID)
            input_id: Optional input ID (default: auto-generated UUID)
            **rollout_overrides: Per-invocation overrides merged into _rollout config
                (e.g., base_url, model_id, temperature). These take precedence over
                client-level defaults.

        Returns:
            RolloutFuture that can be awaited or polled for the result

        Usage:
            future = client.invoke({"prompt": "...", "answer": "42"})
            result = future.result(timeout=60)

            # With per-invocation overrides:
            future = client.invoke(payload, base_url="http://other-server", temperature=0.9)
        """
        session_id = session_id or str(uuid.uuid4())
        input_id = input_id or str(uuid.uuid4())
        return self._rate_limited_invoke(payload, session_id, input_id, **rollout_overrides)

    async def invoke_async(
        self, payload: dict, session_id: str = None, input_id: str = None, **rollout_overrides
    ) -> RolloutFuture:
        """Async version of ``invoke()``. Returns a ``RolloutFuture``.

        Args:
            payload: The payload to send to the agent
            session_id: Optional session ID (default: auto-generated UUID)
            input_id: Optional input ID (default: auto-generated UUID)
            **rollout_overrides: Per-invocation overrides merged into _rollout config
                (e.g., base_url, model_id, temperature). These take precedence over
                client-level defaults.

        Returns:
            RolloutFuture that can be awaited or polled for the result

        Usage::

            future = await client.invoke_async({"prompt": "...", "answer": "42"})
            result = await future.result_async(timeout=60)
            # or simply: result = await future

            # With per-invocation overrides:
            future = await client.invoke_async(payload, base_url="http://other-server")
        """
        session_id = session_id or str(uuid.uuid4())
        input_id = input_id or str(uuid.uuid4())
        return await self._async_rate_limited_invoke(payload, session_id, input_id, **rollout_overrides)

    def run_batch(
        self,
        payloads: list[dict],
        max_concurrent_sessions: int,
        timeout: float = 900.0,
        initial_interval: float = 0.5,
        max_interval: float = 30.0,
        backoff_factor: float = 1.5,
        log_interval: float = 30.0,
    ) -> "BatchResult":
        """
        Run batch with full lifecycle management.

        Handles:
        - TPS rate limiting (default 25/sec)
        - Session concurrency limiting
        - Automatic completion polling via S3 HEAD with exponential backoff
        - Yielding results as they complete
        - Per-request timeout
        - Periodic status logging

        Note:
            Results are yielded in completion order, NOT input order. This is more
            efficient as it doesn't require buffering. Use item.index to
            correlate results with inputs.

        Args:
            payloads: List of payloads to process
            max_concurrent_sessions: Max ACR sessions to run concurrently. Set based
                on your ACR session quota and model API quota, etc.
            timeout: Max seconds to wait for each request (default 900s / 15 min).
                Requests exceeding this yield a BatchItem with success=False.
            initial_interval: Starting poll interval (default 0.5s)
            max_interval: Cap on poll interval (default 30s)
            backoff_factor: Multiply interval by this each poll (default 1.5x)
            log_interval: Seconds between status log messages (default 30s)

        Returns:
            BatchResult iterator that yields BatchItem for each payload

        Usage:
            for item in client.run_batch(payloads, max_concurrent_sessions=10):
                if item.success:
                    process(item.result)
                else:
                    log.warning(f"Payload {item.index} failed: {item.error}")
        """
        return BatchResult(
            client=self,
            payloads=payloads,
            max_concurrent=max_concurrent_sessions,
            timeout=timeout,
            initial_interval=initial_interval,
            max_interval=max_interval,
            backoff_factor=backoff_factor,
            log_interval=log_interval,
        )

    def run_batch_async(
        self,
        payloads: list[dict],
        max_concurrent_sessions: int,
        timeout: float = 900.0,
        initial_interval: float = 0.5,
        max_interval: float = 30.0,
        backoff_factor: float = 1.5,
        log_interval: float = 30.0,
    ) -> "AsyncBatchResult":
        """Async version of ``run_batch()``. Returns an async iterator.

        Same semantics as ``run_batch`` but submissions and polling use
        ``asyncio`` so cold starts on one request don't block others.

        Usage::

            async for item in client.run_batch_async(payloads, max_concurrent_sessions=10):
                if item.success:
                    process(item.result)
                else:
                    log.warning(f"Payload {item.index} failed: {item.error}")
        """
        return AsyncBatchResult(
            client=self,
            payloads=payloads,
            max_concurrent=max_concurrent_sessions,
            timeout=timeout,
            initial_interval=initial_interval,
            max_interval=max_interval,
            backoff_factor=backoff_factor,
            log_interval=log_interval,
        )


class BatchResult:
    """Iterator that manages batch execution lifecycle with adaptive polling."""

    def __init__(
        self,
        client: RolloutClient,
        payloads: list[dict],
        max_concurrent: int,
        timeout: float = 900.0,
        initial_interval: float = 0.5,
        max_interval: float = 30.0,
        backoff_factor: float = 1.5,
        log_interval: float = 30.0,
    ):
        self.client = client
        self.payloads = list(payloads)
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.initial_interval = initial_interval
        self.max_interval = max_interval
        self.backoff_factor = backoff_factor
        self.log_interval = log_interval

    def __iter__(self):
        """Yield BatchItem as sessions complete, with per-future exponential backoff.

        Yields BatchItem for each payload, with success=True for completed results
        and success=False for errors. This allows batch processing to continue
        even when some requests fail.
        """
        pending_payloads = list(enumerate(self.payloads))  # (index, payload)
        active_futures: dict[str, tuple[int, RolloutFuture]] = {}  # key -> (index, future)
        last_status_log = time.time()

        while pending_payloads or active_futures:
            # Start new sessions up to max_concurrent (respects TPS via _rate_limited_invoke)
            while pending_payloads and len(active_futures) < self.max_concurrent:
                idx, payload = pending_payloads.pop(0)
                session_id = str(uuid.uuid4())
                input_id = str(uuid.uuid4())
                try:
                    future = self.client._rate_limited_invoke(payload, session_id, input_id)
                    # Override future's backoff settings
                    future._poll_interval = self.initial_interval
                    future._initial_interval = self.initial_interval
                    future._max_interval = self.max_interval
                    future._backoff_factor = self.backoff_factor
                    active_futures[future.result_key] = (idx, future)
                except Exception as e:
                    yield BatchItem(success=False, error=_format_exception(e), index=idx, elapsed=0.0)

            # Poll futures that are ready (per-future backoff) and check for timeouts
            completed_keys = []
            for key, (idx, future) in active_futures.items():
                # Check for timeout first
                if future.elapsed() > self.timeout:
                    completed_keys.append(key)
                    future.cancel()
                    yield BatchItem(
                        success=False,
                        error=f"Timeout after {self.timeout}s",
                        index=idx,
                        elapsed=future.elapsed(),
                    )
                elif future.ready_to_poll() and future.done():
                    completed_keys.append(key)
                    try:
                        result = future.result()
                        yield BatchItem(success=True, result=result, index=idx, elapsed=future.elapsed())
                    except Exception as e:
                        yield BatchItem(success=False, error=_format_exception(e), index=idx, elapsed=future.elapsed())
                    finally:
                        future.cancel()

            # Remove completed from active
            for key in completed_keys:
                del active_futures[key]

            # Log status periodically
            if active_futures and (time.time() - last_status_log) >= self.log_interval:
                elapsed_times = [f.elapsed() for _, f in active_futures.values()]
                min_elapsed = min(elapsed_times)
                max_elapsed = max(elapsed_times)
                logger.info(
                    f"[Status] {len(active_futures)} active, {len(pending_payloads)} pending, "
                    f"elapsed: {min_elapsed:.0f}s-{max_elapsed:.0f}s"
                )
                last_status_log = time.time()

            # Sleep until next poll or timeout, whichever comes first
            if active_futures and not completed_keys:
                min_wait = min(f.time_until_next_poll() for _, f in active_futures.values())
                if self.timeout:
                    min_timeout_wait = min(self.timeout - f.elapsed() for _, f in active_futures.values())
                    min_wait = min(min_wait, max(0, min_timeout_wait))
                if min_wait > 0:
                    time.sleep(min_wait)


class AsyncBatchResult:
    """Async iterator that manages batch execution lifecycle with adaptive polling.

    Submissions are dispatched as concurrent tasks so that cold starts on one
    request don't block submission of others.
    """

    def __init__(
        self,
        client: RolloutClient,
        payloads: list[dict],
        max_concurrent: int,
        timeout: float = 900.0,
        initial_interval: float = 0.5,
        max_interval: float = 30.0,
        backoff_factor: float = 1.5,
        log_interval: float = 30.0,
    ):
        self.client = client
        self.payloads = list(payloads)
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.initial_interval = initial_interval
        self.max_interval = max_interval
        self.backoff_factor = backoff_factor
        self.log_interval = log_interval

    async def _run(self):
        """Yield ``BatchItem`` as sessions complete."""
        pending_payloads = list(enumerate(self.payloads))  # (index, payload)
        # Tasks that are still submitting (HTTP call in progress)
        submitting: dict[asyncio.Task, tuple[int, float]] = {}  # task -> (index, start_time)
        # Futures whose submission completed and are now polling S3
        active_futures: dict[str, tuple[int, RolloutFuture]] = {}  # key -> (index, future)
        last_status_log = time.time()

        while pending_payloads or submitting or active_futures:
            # Fire new submissions as concurrent tasks
            while pending_payloads and (len(submitting) + len(active_futures)) < self.max_concurrent:
                idx, payload = pending_payloads.pop(0)
                session_id = str(uuid.uuid4())
                input_id = str(uuid.uuid4())
                task = asyncio.create_task(self.client._async_rate_limited_invoke(payload, session_id, input_id))
                submitting[task] = (idx, time.time())

            # Collect completed submissions → move to active_futures
            done_tasks = [t for t in submitting if t.done()]
            for task in done_tasks:
                idx, start = submitting.pop(task)
                try:
                    future = task.result()
                    future._poll_interval = self.initial_interval
                    future._initial_interval = self.initial_interval
                    future._max_interval = self.max_interval
                    future._backoff_factor = self.backoff_factor
                    active_futures[future.result_key] = (idx, future)
                except Exception as e:
                    yield BatchItem(success=False, error=_format_exception(e), index=idx, elapsed=time.time() - start)

            # Poll active futures that are ready and check for timeouts
            completed_keys = []
            for key, (idx, future) in active_futures.items():
                if future.elapsed() > self.timeout:
                    completed_keys.append(key)
                    await future.cancel_async()
                    yield BatchItem(
                        success=False,
                        error=f"Timeout after {self.timeout}s",
                        index=idx,
                        elapsed=future.elapsed(),
                    )
                elif future.ready_to_poll() and await future.done_async():
                    completed_keys.append(key)
                    try:
                        result = await future.result_async()
                        yield BatchItem(success=True, result=result, index=idx, elapsed=future.elapsed())
                    except Exception as e:
                        yield BatchItem(success=False, error=_format_exception(e), index=idx, elapsed=future.elapsed())
                    finally:
                        await future.cancel_async()

            for key in completed_keys:
                del active_futures[key]

            # Log status periodically
            if active_futures and (time.time() - last_status_log) >= self.log_interval:
                elapsed_times = [f.elapsed() for _, f in active_futures.values()]
                min_elapsed = min(elapsed_times)
                max_elapsed = max(elapsed_times)
                logger.info(
                    f"[Status] {len(active_futures)} active, {len(submitting)} submitting, "
                    f"{len(pending_payloads)} pending, elapsed: {min_elapsed:.0f}s-{max_elapsed:.0f}s"
                )
                last_status_log = time.time()

            # Sleep until next event
            if (active_futures or submitting) and not completed_keys and not done_tasks:
                min_wait = float("inf")
                if active_futures:
                    min_wait = min(f.time_until_next_poll() for _, f in active_futures.values())
                    if self.timeout:
                        min_timeout_wait = min(self.timeout - f.elapsed() for _, f in active_futures.values())
                        min_wait = min(min_wait, max(0, min_timeout_wait))
                # If only submitting tasks are pending, use a short sleep
                if submitting and min_wait == float("inf"):
                    min_wait = 0.1
                if min_wait > 0 and min_wait != float("inf"):
                    await asyncio.sleep(min_wait)

    def __aiter__(self):
        return self._run().__aiter__()
