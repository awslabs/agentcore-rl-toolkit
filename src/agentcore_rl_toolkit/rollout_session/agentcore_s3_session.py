"""A rollout session that drives an ``AgentCoreRLApp`` agent through :class:`RolloutClient`.

:mod:`.agentcore_http_session` and :mod:`.docker_session` speak the four-POST :mod:`.wire`
protocol to an agent server built for training. This session instead wraps the agent
contract that already exists in production: an ``AgentCoreRLApp`` with
``@app.rollout_entrypoint``, invoked once fire-and-forget, whose result dict lands in S3
for a :class:`RolloutFuture` to poll. An agent already deployed for batch evaluation
therefore trains with no container-side change at all -- same image, same payload, same
result contract.

What that contract costs, all accepted deliberately:

* **The setup/run split is thin.** ``invoke_async`` would create the ACR session itself, so
  all ``setup`` can do is start that session early (a warm-up command, under
  ``container_setup_timeout``) to keep the microVM cold start out of ``agent_run_timeout``.
  Preparing the *task* inside the container has no channel of its own here: whatever the
  handler does before the agent runs is charged to ``agent_run_timeout``. That warm-up is
  also where this backend feels container readiness -- see the retry in
  :func:`start_agentcore_session` -- and a ``setup`` that fails regardless still stops its
  session rather than leaving a warm microVM to the idle reaper.
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

from agentcore_rl_toolkit.aws_tools.agentcore_tools import (
    region_of,
    shared_agentcore_client,
    start_agentcore_session,
    stop_agentcore_microvm_session,
)
from agentcore_rl_toolkit.aws_tools.persistent_dict import PersistentDict, measure_span_persistent
from agentcore_rl_toolkit.client import RolloutClient, RolloutFuture
from agentcore_rl_toolkit.rollout_session.errors import RolloutContractError
from agentcore_rl_toolkit.rollout_session.lifecycle import RolloutSession
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
    puts results under ``s3://bucket/runs/exp-1/<group_id>/<session>.json`` -- rather than
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
        # shut down", and shutdown stops the session directly in the first case.
        self._future: RolloutFuture | None = None
        # Whether an ACR session may exist for this rollout, which is what shutdown has to
        # act on when there is no future to ride.
        self._started = False

    async def __aenter__(self) -> "AgentCoreS3Session":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.shutdown()

    async def _acr(self) -> Any:
        """The shared data-plane client this session's own ACR calls ride on.

        Not ``RolloutClient``'s own boto3 client: that one is synchronous, so hundreds of
        concurrent setups would queue on the default thread pool. This one is per process,
        not per rollout.
        """
        return await shared_agentcore_client(region_of(self._client.agent_runtime_arn))

    async def setup(self, task: dict) -> None:
        """Start the ACR session, and record where this rollout ran.

        :meth:`run`'s invoke would create the session itself, but then the microVM cold
        start would be charged to ``agent_run_timeout``; starting it here puts it under
        ``container_setup_timeout``, where the other backends' provisioning lives. The
        recorded coordinates identify where to look when a rollout fails.
        """
        runtime_arn = self._client.agent_runtime_arn
        await self.session_state.update(
            {
                "runtime_arn": runtime_arn,
                "result_s3_bucket": self._client.s3_bucket,
                # The client's exp_id: the experiment name, under the configured prefix.
                "result_s3_prefix": self._client.exp_id,
            }
        )

        async with measure_span_persistent("agentcore_setup", self.session_state):
            client = await self._acr()
            # Marked before the call, not after: a warm-up that fails is precisely the case
            # where the microVM usually *does* exist -- the platform's command dispatch loses
            # a race with the container's start-up -- and one nobody stops holds a session
            # slot until ACR's idle reaper eventually takes it.
            self._started = True
            await start_agentcore_session(runtime_arn, self.session_id, client)

    async def run(self, task: dict) -> RolloutDumpResponse:
        llm = _require_llm(task)
        # An invoke creates the session too if setup did not, so from here on there may be one
        # to stop even if no future comes back.
        self._started = True
        future = await self._client.invoke_async(
            _require_payload(task),
            session_id=self.session_id,
            input_id=_require_group_id(task),
            base_url=llm["base_url"],
            # LiteLLM-shaped `openai/<name>`; the agent's OpenAI client wants the bare name.
            model_id=str(llm["model"]).split("/")[-1],
            api_key=llm["api_key"],
        )
        self._future = future
        await self.session_state.set("result_key", future.result_key)
        result = await future
        return to_dump(result)

    async def shutdown(self) -> None:
        """Stop the ACR runtime session. Idempotent, and safe before or after :meth:`run`.

        Two paths reach the same session. When :meth:`run` got a future, the stop rides it:
        ``RolloutFuture.cancel_async`` is itself idempotent, so this is a no-op once the
        result fetch has stopped the session. When no future exists -- a :meth:`setup` that
        raised, or an invoke that raised after ACR accepted it -- the stop is made directly
        against the runtime instead of leaving a warm microVM to ACR's idle reaper. That one
        is best-effort by construction (see :func:`stop_agentcore_microvm_session`): teardown
        must not replace the error that ended the rollout.
        """
        future, self._future = self._future, None
        started, self._started = self._started, False
        if future is not None:
            await future.cancel_async()
        elif started:
            await self._stop_started_session()

    async def _stop_started_session(self) -> None:
        """Stop this rollout's ACR session with no future to ride. Never raises."""
        try:
            client = await self._acr()
        except Exception as e:
            # Only reachable if the shared client cannot be opened at all, which setup
            # already did once. Swallowed for the same reason the stop itself is.
            logger.warning("no data-plane client to stop session %s with: %s", self.session_id, e)
            return
        await stop_agentcore_microvm_session(self._client.agent_runtime_arn, self.session_id, client)


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


def _require_group_id(task: dict) -> str:
    """The task's rollout-group id, which this session keys its result objects by.

    It becomes the ``input_id`` segment of ``{exp_id}/{input_id}/{session_id}.json``, so the
    rollouts sharing a group id land in one S3 prefix. The name is the harness's contract,
    not any one trainer's: verl's agent loop maps its own ``uid`` onto it, and an evaluator
    with no notion of groups can use the task's own id, one rollout per group.

    Deliberately no fallback: the row index keys results by position, which shuffles between
    runs, and the session id is unique per rollout, so either one scatters a group's results
    instead of grouping them.

    A :class:`RolloutContractError` rather than the subscript's ``KeyError`` because no
    rollout of a run whose tasks lack it can succeed. Nothing treats the marker type
    specially yet -- the agent loop logs it with a traceback and records it in the session's
    S3 output like any other rollout exception, which is enough to diagnose it.
    """
    group_id = task.get("group_id")
    if group_id is None or (isinstance(group_id, str) and not group_id.strip()):
        raise RolloutContractError(
            f"The task has no usable `group_id` (got {group_id!r}). It names the group of "
            "rollouts this one belongs to, and this session stores each agent result under "
            "it, so there is nothing to key this rollout's result object by."
        )
    return str(group_id)


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
