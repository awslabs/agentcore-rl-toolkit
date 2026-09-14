"""A rollout session that drives an ``AgentCoreRLApp`` agent through :class:`RolloutClient`.

:mod:`.agentcore_http_session` and :mod:`.docker_session` speak the four-POST :mod:`.wire`
protocol to an agent server built for training. This session instead wraps the agent
contract that already exists in production: an ``AgentCoreRLApp`` with
``@app.rollout_entrypoint``, invoked once fire-and-forget, whose result dict lands in S3
for a :class:`RolloutFuture` to poll. An agent already deployed for batch evaluation
therefore trains with no container-side change at all -- same image, same payload, same
result contract.

What that contract costs, all accepted deliberately:

* **The setup/run split is vacuous.** ``invoke_async`` both creates the ACR session and
  starts the rollout, so ``setup`` provisions nothing, ``container_setup_timeout`` bounds
  nothing, and cold start plus task setup are charged to ``agent_run_timeout``.
* **Failure detection is coarse.** There is no status channel, so a container that dies
  without the SDK's error path running (OOM, process death) is noticed only when
  ``agent_run_timeout`` expires. Exceptions raised inside the handler still come back
  promptly, as ``{"status_code": 500, "stop_reason": ...}`` through S3.
* **Reward and metrics are conventions of the result dict** (``{"rewards": ...}`` plus an
  optional ``metrics`` dict) rather than typed dump fields.
* **Completion is polled.** ``RolloutFuture`` HEADs S3 with per-future exponential
  backoff, so a finished rollout is noticed up to its ``max_interval`` (30s) late.
"""

import logging
from typing import Any

from agentcore_rl_toolkit.aws_tools.persistent_dict import PersistentDict
from agentcore_rl_toolkit.client import RolloutClient, RolloutFuture
from agentcore_rl_toolkit.rollout_session.errors import RolloutContractError
from agentcore_rl_toolkit.rollout_session.lifecycle import RolloutSession, require_task_id
from agentcore_rl_toolkit.rollout_session.wire import RolloutDumpResponse

logger = logging.getLogger(__file__)

# ACR's runtimeSessionId floor. The service rejects a shorter id, so a harness that
# numbers its sessions too tersely should hear about it at construction rather than one
# rollout later, as a botocore ValidationException.
MIN_ACR_SESSION_ID_LEN = 33

# ACR throttling is owned by run_rollout_with_bounds -- bounds.session_rate_limiter is a
# cluster-wide named actor, which is the only place a per-ARN TPS cap can actually be
# enforced: the client's limiter is per process, so N agent-loop workers each holding a 25
# TPS budget admit 25N. Effectively-infinite rather than a disable flag, because
# ACRRateLimiter has no off switch; a `1/tps` interval of a nanosecond never sleeps.
#
# What that leaves unthrottled is the stop call in teardown, which shares ACR's per-ARN
# budget with invoke but happens outside the bounded region -- so size
# session_rate_limiter for roughly two calls per rollout. The client's boto3 config
# (adaptive retries) is the backstop if that is ever wrong.
UNTHROTTLED_TPS = 10**9

# Sessions are constructed one per trajectory, so same-config sessions MUST share one
# client: the client owns the boto3 clients and their connection pools. Keyed by config
# rather than a process singleton so distinct runtimes get distinct clients.
# RolloutClient is not thread-safe; as in `backends/verl/agent_loop.py`, this is only ever
# touched from the agent-loop worker's asyncio thread.
_CLIENTS: dict[tuple, RolloutClient] = {}


def get_or_create_rollout_client(
    *,
    agentcore_runtime_arn: str,
    rollout_output_s3: str,
    experiment_name: str,
    max_pool_connections: int = 100,
) -> RolloutClient:
    """The process-wide :class:`RolloutClient` for this config, created on first use.

    Takes the config's canonical names; ``RolloutClient``'s own ``s3_bucket`` / ``exp_id``
    pair is derived from them by :func:`result_location`.
    """
    bucket, exp_id = result_location(rollout_output_s3, experiment_name)
    key = (agentcore_runtime_arn, bucket, exp_id, max_pool_connections)
    client = _CLIENTS.get(key)
    if client is None:
        client = _CLIENTS[key] = RolloutClient(
            agent_runtime_arn=agentcore_runtime_arn,
            s3_bucket=bucket,
            exp_id=exp_id,
            tps_limit=UNTHROTTLED_TPS,
            max_pool_connections=max_pool_connections,
        )
    return client


def result_location(rollout_output_s3: str, experiment_name: str) -> tuple[str, str]:
    """``(bucket, exp_id)`` for the agent's result objects, from an S3 location.

    The agent SDK writes one object per rollout at ``{exp_id}/{input_id}/{session_id}.json``
    in a single bucket, with no notion of a prefix. So a prefix in ``rollout_output_s3`` is
    honoured by folding it into ``exp_id`` -- ``s3://bucket/runs`` with experiment ``exp-1``
    puts results under ``s3://bucket/runs/exp-1/<task_id>/<session>.json`` -- rather than
    being silently dropped at the bucket root. A bare bucket name works too.
    """
    location = rollout_output_s3.strip()
    if location.startswith("s3://"):
        location = location[len("s3://") :]
    bucket, _, prefix = location.strip("/").partition("/")
    if not bucket:
        raise ValueError(
            f"rollout_output_s3={rollout_output_s3!r} names no bucket. Give a bucket name or "
            "an `s3://bucket[/prefix]` URI; agent results are written under it."
        )
    if not experiment_name or not experiment_name.strip():
        raise ValueError(
            f"experiment_name={experiment_name!r} is empty. It prefixes this run's agent "
            "results, so re-runs would otherwise be indistinguishable in the bucket."
        )
    experiment_name = experiment_name.strip()
    return bucket, f"{prefix}/{experiment_name}" if prefix else experiment_name


def reset_client_cache() -> None:
    """Forget every cached client. For tests; each client holds boto3 clients."""
    _CLIENTS.clear()


class AgentCoreS3Session(RolloutSession):
    """A rollout session backed by one ACR invocation of an ``AgentCoreRLApp`` agent."""

    def __init__(
        self,
        session_id: str,
        session_state: PersistentDict,
        *,
        client: RolloutClient,
    ):
        if len(session_id) < MIN_ACR_SESSION_ID_LEN:
            raise ValueError(
                f"session_id={session_id!r} is {len(session_id)} characters; ACR requires a "
                f"runtimeSessionId of at least {MIN_ACR_SESSION_ID_LEN}. This id is used as "
                "the ACR session id and as the gateway's capture key, so it cannot be padded "
                "on one side only -- lengthen the harness's id instead."
            )
        self.session_id = session_id
        self.session_state = session_state
        self._client = client
        # Set once the invoke returns; `None` means either "not invoked yet" or "already
        # shut down", and both make shutdown a no-op.
        self._future: RolloutFuture | None = None

    async def __aenter__(self) -> "AgentCoreS3Session":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.shutdown()

    async def setup(self, task: dict) -> None:
        """Provision nothing: :meth:`run`'s invoke creates the ACR session itself.

        Only records which runtime and result bucket this rollout used, so the session
        record identifies where to look when it fails.
        """
        await self.session_state.update(
            {
                "runtime_arn": self._client.agent_runtime_arn,
                "result_s3_bucket": self._client.s3_bucket,
                # The client's exp_id: the experiment name, under the configured prefix.
                "result_s3_prefix": self._client.exp_id,
            }
        )

    async def run(self, task: dict) -> RolloutDumpResponse:
        llm = _require_llm(task)
        # The session id doubles as the ACR runtimeSessionId and as the agent's api_key,
        # which is the slot the rollout gateway reads as its capture session -- so the
        # trajectory the trainer drains is the one this container generated.
        future = await self._client.invoke_async(
            _require_payload(task),
            session_id=self.session_id,
            input_id=require_task_id(task),
            base_url=llm["base_url"],
            # LiteLLM-shaped `openai/<name>`; the agent's OpenAI client wants the bare name.
            model_id=str(llm["model"]).split("/")[-1],
            api_key=llm["api_key"],
        )
        self._future = future
        await self.session_state.set("result_key", future.result_key)
        # `task["sampling_params"]` is deliberately not forwarded: the gateway applies the
        # session's own sampling defaults over whatever the container asks for, so a second
        # copy on the invoke payload would only be able to disagree.

        # No timeout argument: run_rollout_with_bounds already wraps this call in
        # bounds.agent_run_timeout and shuts the session down on the way out.
        result = await future
        return to_dump(result)

    async def shutdown(self) -> None:
        """Stop the ACR runtime session. Idempotent, and safe before or after :meth:`run`.

        ``RolloutFuture.cancel_async`` is itself idempotent, so this is a no-op when the
        result fetch already stopped the session. The one uncovered case is an invoke that
        raised *after* ACR accepted it (a malformed immediate response): no future exists,
        so that session is left to ACR's idle reaper.
        """
        future, self._future = self._future, None
        if future is not None:
            await future.cancel_async()


def to_dump(result: dict) -> RolloutDumpResponse:
    """One agent result dict, as the dump the rollout lifecycle returns.

    A result that reports a non-200 ``status_code`` becomes the dump's ``exception``, and
    reward extraction is skipped -- a failed handler's ``rewards`` field, if any, describes
    nothing. A successful result with no reward is a contract violation of this session's
    dump (``failure_reason`` reports it), not a zero.
    """
    exception = status_error(result)
    return RolloutDumpResponse(
        metrics=numeric_metrics(result.get("metrics")),
        task_output=result,
        reward=None if exception is not None else extract_agent_reward(result),
        exception=exception,
    )


def status_error(result: dict) -> str | None:
    """The agent-reported failure in a result dict, or None if it reports success.

    ``@rollout_entrypoint`` saves a result either way; a handler that raised is saved with
    a non-200 ``status_code`` and the exception in ``stop_reason``.
    """
    status_code = result.get("status_code")
    if status_code is None or status_code == 200:
        return None
    return f"agent returned status_code={status_code}: {result.get('stop_reason', 'unknown')}"


def numeric_metrics(metrics: Any) -> dict[str, float]:
    """The float-valued entries of an agent's ``metrics`` dict, dropping the rest.

    The dump's metrics are scalars the trainer reduces across a batch, while the agent's
    result may carry anything JSON holds; non-numeric entries survive in ``task_output``.
    """
    if not isinstance(metrics, dict):
        return {}
    numeric: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, bool | int | float):
            numeric[str(key)] = float(value)
    return numeric


def extract_agent_reward(result: dict) -> float | None:
    """The agent-reported reward from a session result (the ``{"rewards": ...}``
    convention of ``@rollout_entrypoint`` apps: scalar, or last element of a list), or
    ``None`` if the agent didn't report one.

    A non-numeric value raises :class:`RolloutContractError` rather than scoring 0.0:
    broken reward code is broken on every rollout, and zeros would flatten every GRPO
    group's advantages instead. The agent loop lets that one out (it absorbs ordinary
    rollout failures, not contract violations), which stops the run.

    A near-copy of ``_extract_agent_reward`` in ``backends/verl/agent_loop.py``, which
    cannot be imported here: this package must stay verl-free. Change both together until
    that loop is retired.
    """
    rewards = result.get("rewards")
    if rewards is None:
        return None
    if isinstance(rewards, list) and not rewards:
        return None  # an empty list reports no reward
    value = rewards[-1] if isinstance(rewards, list) else rewards
    try:
        return float(value)
    except (TypeError, ValueError) as e:
        raise RolloutContractError(
            f"The agent returned a non-numeric built-in reward: rewards={rewards!r} ({e}). "
            "It must be a float, or a list of floats whose last element is the reward."
        ) from e


def _require_payload(task: dict) -> dict:
    """The agent's invoke payload: the row's ``payload`` column, forwarded verbatim.

    The single contract, with no field selection and no forward-everything fallback -- the
    task namespace is shared with the trainer's plumbing fields, and trainer-shaped columns
    (chat-format prompts, tensors) are not agent-shaped.
    """
    payload = task.get("payload")
    if isinstance(payload, dict):
        return payload
    raise RolloutContractError(
        "Cannot build the agent invoke payload: the task has no `payload` dict "
        f"(got {type(task.get('payload')).__name__}). Author dataset rows with a `payload` "
        "column holding the agent's exact invoke payload; the trainer forwards it unchanged."
    )


def _require_llm(task: dict) -> dict:
    """The task's ``llm`` block, with the fields this session reads present.

    ``api_key`` is checked as carefully as the address: it is the gateway's capture key, and
    a task missing it would run a whole rollout against the trainer's engine while capturing
    no trajectory at all.
    """
    llm = task.get("llm")
    missing = [k for k in ("base_url", "model", "api_key") if not isinstance(llm, dict) or not llm.get(k)]
    if missing:
        raise RolloutContractError(
            f"The task's `llm` block is missing {missing} (got {llm!r}). The harness builds it "
            "per rollout: `base_url` and `model` address the trainer's inference endpoint, and "
            "`api_key` is the capture session id the rollout gateway keys trajectories by."
        )
    return llm  # type: ignore[return-value]
