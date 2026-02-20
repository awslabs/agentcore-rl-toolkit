"""Client for invoking agents and collecting rollouts via S3 polling."""

import json
import time
import uuid
from dataclasses import dataclass

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError


@dataclass
class BatchItem:
    """Result wrapper for batch execution, distinguishing success from error.

    Attributes:
        success: True if the request completed, False if it failed.
        result: The rollout result dict (populated when success=True).
        error: Error message (populated when success=False).
        index: Index of the payload in the original payloads list.
    """

    success: bool
    result: dict = None
    error: str = None
    index: int = None


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
    ):
        self.s3_client = s3_client
        self.s3_bucket = s3_bucket
        self.result_key = result_key
        self._result = None
        self._done = False

        # Per-future backoff state
        self._poll_interval = initial_interval
        self._initial_interval = initial_interval
        self._max_interval = max_interval
        self._backoff_factor = backoff_factor
        self._last_poll_time = 0.0

    def done(self) -> bool:
        """Check if result is ready (non-blocking). Updates backoff state."""
        if self._done:
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
            TimeoutError: If timeout is reached before result is ready.
        """
        if self._result is not None:
            return self._result

        start = time.time()

        while True:
            if self.done():
                response = self.s3_client.get_object(Bucket=self.s3_bucket, Key=self.result_key)
                self._result = json.loads(response["Body"].read())
                return self._result

            if timeout and (time.time() - start) > timeout:
                raise TimeoutError(f"Result not ready after {timeout}s")

            # Use the per-future backoff interval
            time.sleep(self._poll_interval)


class RolloutClient:
    """Client for invoking agents and collecting rollouts with full lifecycle management.

    Note:
        This client is NOT thread-safe. Use separate client instances for
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
        # Optional model inference config (for vLLM/SGLang servers)
        base_url: str = None,
        model_id: str = None,
        # Additional config passed through to _training (e.g., sampling params)
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
            base_url: Optional vLLM/SGLang server URL
            model_id: Optional model ID for inference
            **extra_config: Additional config passed to _training (e.g., temperature, top_p)
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
        config = Config(retries={"max_attempts": max_retry_attempts, "mode": "adaptive"})
        self.agentcore_client = boto3.client("bedrock-agentcore", region_name=self.region, config=config)
        self.s3_client = boto3.client("s3", region_name=self.region, config=config)

        # Rate limiting state
        self._last_invoke_time = 0.0
        self._min_invoke_interval = 1.0 / tps_limit

    def _parse_response(self, response: dict) -> dict:
        """Parse ACR invocation response."""
        return json.loads(response["response"].read())

    def _rate_limited_invoke(self, payload: dict, session_id: str, input_id: str) -> RolloutFuture:
        """Invoke with TPS rate limiting."""
        # Enforce TPS limit
        now = time.time()
        elapsed = now - self._last_invoke_time
        if elapsed < self._min_invoke_interval:
            time.sleep(self._min_invoke_interval - elapsed)
        self._last_invoke_time = time.time()

        # Build rollout config
        rollout_config = {
            "exp_id": self.exp_id,
            "session_id": session_id,
            "input_id": input_id,
            "s3_bucket": self.s3_bucket,
            **self.extra_config,
        }
        if self.base_url:
            rollout_config["base_url"] = self.base_url
        if self.model_id:
            rollout_config["model_id"] = self.model_id

        full_payload = {**payload, "_training": rollout_config}

        # Invoke via boto3
        response = self.agentcore_client.invoke_agent_runtime(
            agentRuntimeArn=self.agent_runtime_arn,
            runtimeSessionId=session_id,
            payload=json.dumps(full_payload),
        )

        data = self._parse_response(response)
        return RolloutFuture(
            s3_client=self.s3_client,
            s3_bucket=data["s3_bucket"],
            result_key=data["result_key"],
        )

    def invoke(self, payload: dict, session_id: str = None, input_id: str = None) -> RolloutFuture:
        """
        Single invocation, returns Future for the result.

        Args:
            payload: The payload to send to the agent
            session_id: Optional session ID (default: auto-generated UUID)
            input_id: Optional input ID (default: auto-generated UUID)

        Returns:
            RolloutFuture that can be awaited or polled for the result

        Usage:
            future = client.invoke({"prompt": "...", "answer": "42"})
            result = future.result(timeout=60)
        """
        session_id = session_id or str(uuid.uuid4())
        input_id = input_id or str(uuid.uuid4())
        return self._rate_limited_invoke(payload, session_id, input_id)

    def run_batch(
        self,
        payloads: list[dict],
        max_concurrent_sessions: int,
        initial_interval: float = 0.5,
        max_interval: float = 30.0,
        backoff_factor: float = 1.5,
    ) -> "BatchResult":
        """
        Run batch with full lifecycle management.

        Handles:
        - TPS rate limiting (default 25/sec)
        - Session concurrency limiting
        - Automatic completion polling via S3 HEAD with exponential backoff
        - Yielding results as they complete

        Note:
            Results are yielded in completion order, NOT input order. This is more
            efficient as it doesn't require buffering. Use item.index to
            correlate results with inputs.

        Args:
            payloads: List of payloads to process
            max_concurrent_sessions: Max ACR sessions to run concurrently. Set based
                on your ACR session quota and model API quota, etc.
            initial_interval: Starting poll interval (default 0.5s)
            max_interval: Cap on poll interval (default 30s)
            backoff_factor: Multiply interval by this each poll (default 1.5x)

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
            initial_interval=initial_interval,
            max_interval=max_interval,
            backoff_factor=backoff_factor,
        )


class BatchResult:
    """Iterator that manages batch execution lifecycle with adaptive polling."""

    def __init__(
        self,
        client: RolloutClient,
        payloads: list[dict],
        max_concurrent: int,
        initial_interval: float = 0.5,
        max_interval: float = 30.0,
        backoff_factor: float = 1.5,
    ):
        self.client = client
        self.payloads = list(payloads)
        self.max_concurrent = max_concurrent
        self.initial_interval = initial_interval
        self.max_interval = max_interval
        self.backoff_factor = backoff_factor

    def __iter__(self):
        """Yield BatchItem as sessions complete, with per-future exponential backoff.

        Yields BatchItem for each payload, with success=True for completed results
        and success=False for errors. This allows batch processing to continue
        even when some requests fail.
        """
        pending_payloads = list(enumerate(self.payloads))  # (index, payload)
        active_futures: dict[str, tuple[int, RolloutFuture]] = {}  # key -> (index, future)

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
                    yield BatchItem(success=False, error=str(e), index=idx)

            # Poll futures that are ready (per-future backoff)
            completed_keys = []
            for key, (idx, future) in active_futures.items():
                if future.ready_to_poll() and future.done():
                    completed_keys.append(key)
                    try:
                        result = future.result()
                        yield BatchItem(success=True, result=result, index=idx)
                    except Exception as e:
                        yield BatchItem(success=False, error=str(e), index=idx)

            # Remove completed from active
            for key in completed_keys:
                del active_futures[key]

            # Sleep until next future is ready to poll
            if active_futures and not completed_keys:
                min_wait = min(f.time_until_next_poll() for _, f in active_futures.values())
                if min_wait > 0:
                    time.sleep(min_wait)
