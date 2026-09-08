"""Batch evaluation harness for container-based agents.

This module is the *machinery*: the endpoints, the :class:`EvalConfig` shape and the
driver that runs one such config. What to actually run -- the dataset paths and the
config grid -- lives in this recipe's ``evaluate.py``, which is what you invoke; this
file holds nothing run-specific and reads no config of its own, since :func:`run_eval`
takes the resolved values as an ``env`` mapping.

It lives in the recipe rather than beside the sessions it drives, unlike
``rollout_session.lifecycle``: what is shared is the *session*, and this is one way to
drive a batch of them -- the toolkit has another with a different session model
(``agentcore_rl_toolkit.client``'s ``run_batch``), and this recipe is the only consumer
of this one. It is the eval-only counterpart to training-time rollout, running a
dataset of tasks through the same session lifecycle -- ``setup(task)`` ->
``run(task) -> RolloutDumpResponse`` -> ``shutdown()`` -- but driving
:class:`RolloutSession` directly, with no trainer and no Ray.

The point is measurement, not training: per rollout we record the reward and the generic
scalar ``metrics`` off the returned ``RolloutDumpResponse`` (plus the session's timing
spans) and persist them -- metrics to DynamoDB, a full dump to S3 -- rather than feeding
tokens back to a trainer. Those per-rollout writes are the run's record; the report file
:func:`run_eval` leaves behind is a snapshot of them, built by :mod:`rollout_report` so
the same numbers can be recomputed from the table for any past run.

Two things are parameters, not fixed:

* the **dataset** -- any parquet whose rows are tasks the agent server accepts;
* the **inference endpoint** -- one of three :class:`Endpoint` flavors, differing only
  in the ``task["llm"]`` seam:

  - :class:`NullEndpoint` -- no inference at all, for agent types that never call an
    LLM (``oracle`` applies the gold patch, ``noop`` grades the untouched repo), so
    those runs still get the whole harness with the inference seam removed.
  - :class:`BedrockEndpoint` -- the agent talks straight to Bedrock (OpenAI-compatible
    chat), with a fresh bearer token minted per rollout off a long-lived STS session
    (see :func:`bedrock_token`, :class:`LateBoundLlmSession`). The light path: reward +
    latencies, **no token capture**, because Bedrock is chat-only and returns neither
    token-id prompts nor output token ids for a token-in/token-out gateway to front.
  - :class:`VllmGatewayEndpoint` -- stands up a standalone rollout **gateway** (the same
    capture layer training uses, wired directly rather than through a Ray host) over a
    specified vLLM server. The agent points at the gateway (``api_key = session_id``
    routes the trajectory) and each rollout drains into ``TraceRecord``s -- token ids,
    loss mask, logprobs -- saved to S3 beside the dump. The heavy path.

Each rollout goes through :func:`run_rollout_with_bounds` rather than resequencing the
bounded lifecycle here. Its bounds arrive as one ``ContainerBounds`` of interfaces, so
this driver fills it with process-local implementations (:func:`local_bounds`) and needs
no cross-process coordination, where a trainer fills it with Ray actors.

Each task is replayed ``n`` times (``n_idx`` 0..n-1) so pass@k and reward/latency
variance are reportable; a group is one task and its ``n`` samples.

Beyond the ``env`` mapping, a run needs ambient AWS creds for Bedrock token minting and
-- for a gateway endpoint -- a reachable vLLM server plus the model's HF tokenizer.
"""

import asyncio
import dataclasses
import datetime as dt
import json
import logging
import os
from typing import Callable, Protocol, runtime_checkable
from uuid import uuid4

# This recipe's own module, imported as a top-level name: it is a sibling of the
# ``./evaluate.py`` entrypoint that drives this harness, and ``conftest.py`` puts the
# same directory on ``sys.path`` so the tests resolve it identically.
from rollout_report import summarize

from agentcore_rl_toolkit.aws_tools.boto3_tools import LongLivedCredentials
from agentcore_rl_toolkit.aws_tools.ec2_monitor import DEFAULT_POLL_INTERVAL, EC2Monitor
from agentcore_rl_toolkit.aws_tools.persistent_dict import DynamoDBPersister, PersistentDict
from agentcore_rl_toolkit.aws_tools.s3_tools import upload_object
from agentcore_rl_toolkit.concurrency.priority_assigner import LocalPriorityAssigner
from agentcore_rl_toolkit.concurrency.priority_semaphore import LocalPrioritySemaphore
from agentcore_rl_toolkit.concurrency.rate_limiter import ACRRateLimiter

# BaseTrace/TraceRecord are torch-free and aiohttp-free; the heavy gateway pieces
# (RolloutGateway, ThreadedGatewayServer, VllmHttpBackend) and transformers are
# imported lazily in VllmGatewayEndpoint.start() so a Bedrock-only run stays light.
from agentcore_rl_toolkit.rollout_gateway import BaseTrace, TraceRecord
from agentcore_rl_toolkit.rollout_session.agentcore_session import AgentCoreSession
from agentcore_rl_toolkit.rollout_session.exception_utils import exception_to_string
from agentcore_rl_toolkit.rollout_session.lifecycle import (
    ContainerBounds,
    RolloutSession,
    run_rollout_with_bounds,
)
from agentcore_rl_toolkit.rollout_session.wire import RolloutDumpResponse

logger = logging.getLogger(__name__)

# Every session this harness creates is named with this prefix, which is both how eval
# sessions are recognizable in the session table (training runs use another) and what
# narrows the EC2 monitor's instance scan to this run's sessions.
SESSION_PREFIX = "eval_"


# --- endpoints: the one seam that differs from a real rollout -----------------


@runtime_checkable
class Endpoint(Protocol):
    """How the agent reaches inference, and whether the run captures tokens.

    The lifecycle mirrors the gateway's: ``start`` once per experiment,
    ``open_session``/``finish_session`` per rollout, ``build_task_llm`` produces
    the ``task["llm"]`` client config the agent server consumes (``None`` when the
    endpoint has no inference, in which case the task carries no ``llm`` key).
    """

    model: str

    def label(self) -> str:
        ...

    def captures_tokens(self) -> bool:
        ...

    async def start(self) -> None:
        ...

    async def stop(self) -> None:
        ...

    def build_task_llm(self, session_id: str, temperature: float, top_p: float) -> dict | None:
        ...

    def open_session(self, session_id: str, sampling_params: dict) -> None:
        ...

    async def finish_session(self, session_id: str) -> list[TraceRecord]:
        ...


@dataclasses.dataclass
class NullEndpoint:
    """No inference: the container runs the agent with no ``task["llm"]`` at all.

    The endpoint for agent types that need no model -- ``oracle`` (applies the reference
    patch and grades it), ``noop`` (grades the repo as-is) and other no-LLM baselines.
    Every seam is a no-op and ``build_task_llm`` returns ``None``, so no ``llm`` key
    reaches the agent server and a session that did read it would fail loudly rather
    than silently talk to a stray model. ``model`` is a label only: it names the run and
    the session item, where a real endpoint records a model id.
    """

    model: str = "none"
    kind: str = "null"

    def label(self) -> str:
        return "null"

    def captures_tokens(self) -> bool:
        return False

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    def build_task_llm(self, session_id: str, temperature: float, top_p: float) -> None:
        return None

    def open_session(self, session_id: str, sampling_params: dict) -> None:
        pass

    async def finish_session(self, session_id: str) -> list[TraceRecord]:
        return []


# --- Bedrock bearer tokens ----------------------------------------------------
#
# A Bedrock bearer token is a SigV4-presigned URL, so it inherits the awkward property
# of one: signed with temporary credentials, it dies when those credentials do, whatever
# its X-Amz-Expires claims. A token's real lifetime is its signing credentials' remaining
# lifetime -- hence LongLivedCredentials rather than whatever the ambient chain holds.
#
# That is also what makes signing per rollout affordable: `provide_token` builds a fresh
# botocore Session and walks the whole credential chain on every call, but the signing
# itself is local HMAC, so handing it resolved credentials removes the chain walk -- the
# part that gets rate limited when a batch opens hundreds of containers in bursts.
#
# *When* a token is minted is the other half of the problem, and is not solved here --
# see :class:`LateBoundLlmSession`.
_bedrock_credentials = LongLivedCredentials(role_session_name="batch-agent-eval")


def bedrock_token(region: str) -> str:
    """Mint a Bedrock bearer token for ``region`` off the shared long-lived session.

    Call this from inside the bounded run, not when the rollout is queued: the
    token's life starts here. Warns if the signing credentials cannot cover the
    session's floor, which only happens in the ambient fallback.
    """
    from aws_bedrock_token_generator import provide_token

    _bedrock_credentials.load()
    remaining = _bedrock_credentials.expires_in()
    if remaining is not None and remaining < _bedrock_credentials.min_lifetime_seconds:
        logger.warning(
            "bedrock token for %s is signed with credentials expiring in %.0fs; "
            "a rollout outliving that will fail with an expired token",
            region,
            remaining,
        )

    # Cap the token's advertised expiry at the credentials' own so X-Amz-Expires does
    # not promise a lifetime the signature cannot back -- a token that outlives its
    # credentials then fails as plainly expired rather than as an opaque auth error.
    max_expiry = _bedrock_credentials.duration_seconds
    expiry_seconds = max_expiry if remaining is None else remaining
    expiry_seconds = max(1, min(expiry_seconds, max_expiry))
    return provide_token(
        region=region,
        aws_credentials_provider=_bedrock_credentials,
        expiry=dt.timedelta(seconds=expiry_seconds),
    )


@dataclasses.dataclass
class BedrockEndpoint:
    """The agent talks straight to Bedrock -- no gateway, no token capture.

    ``model`` is the LiteLLM model id (e.g. ``openai/qwen.qwen3-coder-30b-a3b-instruct``);
    the ``openai/`` provider prefix routes LiteLLM at ``base_url``. Every rollout
    mints its own bearer token via :func:`bedrock_token`; because
    :class:`LateBoundLlmSession` calls ``build_task_llm`` from inside the bounded
    run, no container is handed a token older than its own rollout.
    """

    model: str
    region: str
    kind: str = "bedrock"

    def label(self) -> str:
        return f"bedrock-{self.region}"

    def captures_tokens(self) -> bool:
        return False

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    def build_task_llm(self, session_id: str, temperature: float, top_p: float) -> dict:
        return dict(
            model=self.model,
            base_url=f"https://bedrock-mantle.{self.region}.api.aws/v1",
            api_key=bedrock_token(self.region),
            temperature=temperature,
            top_p=top_p,
        )

    def open_session(self, session_id: str, sampling_params: dict) -> None:
        pass

    async def finish_session(self, session_id: str) -> list[TraceRecord]:
        return []


@dataclasses.dataclass
class VllmGatewayEndpoint:
    """The agent talks to a rollout gateway fronting a specified vLLM server.

    ``start`` assembles a ``RolloutGateway`` (tokenizer + renderer + a
    :class:`VllmHttpBackend` pointed at ``vllm_url``) and serves it on a background
    thread via ``ThreadedGatewayServer`` -- the same capture layer training uses, but
    wired directly here (no verl/Ray). The gateway is a per-experiment singleton
    shared by every concurrent rollout.

    Per rollout: ``open_session`` registers the sid, the agent's OpenAI client
    (``api_key = session_id``) drives the gateway, and ``finish_session`` drains the
    trajectory tree into ``TraceRecord``s. ``vllm_url`` is the base the backend POSTs
    ``{vllm_url}/inference/v1/generate`` to; ``tokenizer_path`` is the HF tokenizer
    the gateway renders and derenders with (must match the served model).
    """

    model: str
    vllm_url: str
    tokenizer_path: str
    max_context_tokens: int = 32768
    max_new_tokens: int = 16000
    host: str = "127.0.0.1"
    kind: str = "vllm"

    def label(self) -> str:
        return "vllm"

    def captures_tokens(self) -> bool:
        return True

    async def start(self) -> None:
        if getattr(self, "_server", None) is not None:
            return
        from transformers import AutoTokenizer

        from agentcore_rl_toolkit.rollout_gateway import (
            HfTemplateRenderer,
            RolloutGateway,
            ThreadedGatewayServer,
        )
        from agentcore_rl_toolkit.rollout_gateway.sampling_backends.vllm_http import VllmHttpBackend

        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        gateway = RolloutGateway(
            backend=VllmHttpBackend(self.vllm_url),
            renderer=HfTemplateRenderer(tokenizer),
            tokenizer=tokenizer,
            adapters=["openai"],
            max_turns_per_sid=1000,
            fork_threshold_tokens=0,
        )
        server = ThreadedGatewayServer(gateway, host=self.host, port=0)
        server.start()
        self._gateway = gateway
        self._server = server
        logger.info("vllm gateway serving at %s (upstream=%s)", server.base_url, self.vllm_url)

    async def stop(self) -> None:
        server = getattr(self, "_server", None)
        if server is not None:
            server.shutdown()
        self._server = None
        self._gateway = None

    def build_task_llm(self, session_id: str, temperature: float, top_p: float) -> dict:
        # The session id rides in the api-key slot, as on the training side, so the
        # gateway keys token capture off it. base_url already ends in /v1, so the
        # OpenAI client hits /v1/chat/completions on the gateway.
        return dict(
            model=f"openai/{self.model.split('/')[-1]}",
            base_url=self._server.base_url,
            api_key=session_id,
            temperature=temperature,
            top_p=top_p,
            num_retries=0,
            timeout=120,
        )

    def open_session(self, session_id: str, sampling_params: dict) -> None:
        defaults: dict = {"max_new_tokens": self.max_new_tokens}
        for key in ("temperature", "top_p", "top_k"):
            if key in sampling_params:
                defaults[key] = sampling_params[key]
        self._gateway.create_session(
            session_id,
            sampling_defaults=defaults,
            max_context_tokens=self.max_context_tokens,
        )

    async def finish_session(self, session_id: str) -> list[TraceRecord]:
        records = await self._gateway.finish_session(
            session_id, base_sample=BaseTrace(rollout_id=session_id), reward=0.0
        )
        # An empty-token record is a turn that produced nothing trainable; drop it,
        # as the training side does.
        return [r for r in records if r.token_ids]


@dataclasses.dataclass
class EvalConfig:
    """One evaluation run: a (dataset slice, endpoint, sampling, concurrency) point.

    ``dataset`` is a parquet path whose rows are tasks -- the same schema training
    reads. ``num_tasks`` caps how many rows are used (None = all); ``n`` is the
    samples-per-task for pass@k / variance.

    The concurrency/rate/timeout fields are the eval-side counterparts of the bounds
    training feeds into :func:`run_rollout_with_bounds`, here bounded process-locally.

    ``task_kwargs`` is the same seam training has: extra keys merged into every task
    the agent server receives (the ``agent`` dispatch key, the
    ``docker_image_namespace`` its task images are pulled from, ...). The harness
    passes it through without interpreting it, so which keys a given agent needs is a
    property of this config, not of the driver.
    """

    experiment_name: str
    endpoint: Endpoint
    dataset: str
    num_tasks: int | None
    n: int
    concurrency: int  # max in-flight containers
    session_create_rate: float  # container/session creation req/sec ceiling
    timeout: int  # per-rollout agent_run wall-clock budget (seconds)
    # Where this run's report is written, as `{report_dir}/{experiment_name}.json`.
    # Deliberately without a default: the report is an artifact of the recipe being
    # evaluated, so only the driver knows where it belongs. Pass an absolute path if it
    # must mean one particular directory -- it is resolved when the run finishes, so a
    # relative one answers to whatever cwd the driver happened to have.
    report_dir: str
    # max concurrent agent runs (None = `concurrency`, i.e. non-binding: every
    # container that exists may run). Lower it to hold containers warm while
    # throttling how many talk to inference at once, as training does.
    rollout_concurrency: int | None = None
    container_setup_timeout: int = 1200  # per-rollout container provisioning budget
    temperature: float = 1.0
    top_p: float = 1.0
    task_kwargs: dict = dataclasses.field(default_factory=dict)
    # seconds between EC2 monitor polls; 0 disables it.
    ec2_monitor_poll_interval: float = DEFAULT_POLL_INTERVAL


def task_index(task_row: dict) -> int:
    """The dataset's own index for a task row, resolved as verl resolves it.

    verl's ``RLHFDataset`` promotes ``extra_info["index"]`` to a top-level ``index``
    field on the batch, which the training side reads as the task id
    (``task_id = str(task["index"])``). Reading it the same way here means an eval task
    id names the same dataset row a training task id does, instead of a number that
    shifts with how the slice was cut. A row that already carries a top-level ``index``
    wins.
    """
    if "index" in task_row:
        return task_row["index"]
    return (task_row.get("extra_info") or {}).get("index", 0)


def token_stats(records: list[TraceRecord]) -> dict:
    """Token counts for one trajectory, from its captured trace records.

    A trajectory is captured as one record per trainable turn, each a full
    prompt+response token sequence whose loss mask marks the LLM-generated span.
    Over the whole trajectory that gives three different numbers, all worth having:

    * ``num_tokens`` -- every token in every record. Records overlap (a turn's
      prompt is the previous turns replayed), so this is the trainer's cost of the
      trajectory, not the length of the agent's conversation.
    * ``num_generated_tokens`` -- the loss-masked tokens only, i.e. what the LLM
      actually produced and what the policy update trains on.
    * ``max_record_tokens`` -- the longest single record, which is the sequence
      that has to fit the context window; it, not the sum, is what bumps against
      ``max_context_tokens``.

    Plus ``num_records``, the number of trainable turns.
    """
    token_counts = [len(record.token_ids) for record in records]
    return {
        "num_records": len(records),
        "num_tokens": sum(token_counts),
        "num_generated_tokens": sum(sum(record.loss_mask) for record in records),
        "max_record_tokens": max(token_counts, default=0),
    }


def _trace_to_dict(record: TraceRecord) -> dict:
    """JSON-safe view of a TraceRecord (token ids / loss mask / logprobs)."""
    return {
        "rollout_id": record.rollout_id,
        "token_ids": record.token_ids,
        "loss_mask": record.loss_mask,
        "logprobs": record.logprobs,
        "reward": record.reward,
        "response_length": record.response_length,
        "response": record.response,
        "metadata": record.metadata,
        "status": record.status.value,
    }


def local_bounds(config: EvalConfig) -> ContainerBounds:
    """Fill :class:`ContainerBounds` with the process-local implementations.

    The bounds are all interfaces, so where training hands the bounded run Ray actors,
    this driver hands it these. Built once per experiment and shared by every rollout,
    which is what makes the semaphores cap anything.
    """
    return ContainerBounds(
        container_semaphore=LocalPrioritySemaphore(config.concurrency),
        rollout_semaphore=LocalPrioritySemaphore(config.rollout_concurrency or config.concurrency),
        container_priority_assigner=LocalPriorityAssigner(),
        rollout_priority_assigner=LocalPriorityAssigner(),
        session_rate_limiter=ACRRateLimiter(config.session_create_rate),
        container_setup_timeout=config.container_setup_timeout,
        agent_run_timeout=config.timeout,
    )


class LateBoundLlmSession:
    """Wraps a session so ``task["llm"]`` is built inside the bounded run, not before it.

    Every rollout coroutine is created up front, so the wait for a container slot
    happens *inside* :func:`run_one` -- which means anything perishable prepared before
    that wait rots while its rollout sits in the queue. A Bedrock bearer token built at
    dispatch is expired long before the container it was minted for exists.

    Rather than add a second gate outside :func:`run_rollout_with_bounds`, this defers
    the perishable step to the one place already behind that function's gates. ``run``
    is the tightest such point and it is not too late: the agent server reads
    ``task_input["llm"]`` only from the rollout-start payload that ``run`` sends, never
    from the setup payload. By then the container slot, the session-creation throttle,
    ``setup`` and the rollout slot are all behind us, so the token's age at the agent's
    last LLM call is bounded by ``agent_run_timeout`` -- a config value -- rather than
    by how long the queue was.

    Delegates the rest of the :class:`RolloutSession` protocol untouched, so the
    wrapped session keeps owning the container lifecycle.
    """

    def __init__(self, session: RolloutSession, build_llm: Callable[[], dict | None]) -> None:
        self._session = session
        self._build_llm = build_llm

    async def __aenter__(self) -> "LateBoundLlmSession":
        await self._session.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self._session.__aexit__(exc_type, exc, tb)

    async def setup(self, task: dict) -> None:
        await self._session.setup(task)

    async def run(self, task: dict) -> RolloutDumpResponse:
        # The token's clock starts here, one step before the payload carrying it
        # goes to the container. A ``None`` config means the endpoint has no
        # inference, and the key must stay absent rather than be set to None.
        llm = self._build_llm()
        if llm is not None:
            task["llm"] = llm
        return await self._session.run(task)

    async def shutdown(self) -> None:
        await self._session.shutdown()


async def run_one(
    config: EvalConfig,
    runtime_arn: str,
    capacity_provider_arn: str,
    experiment_start_at: str,
    task_row: dict,
    n_idx: int,
    bounds: ContainerBounds,
    s3_prefix: str,
    session_table: str,
    storage_region: str,
) -> dict:
    """Drive one rollout (one sample of one task) end to end.

    The bounded lifecycle -- container slot, creation-rate throttle, timed ``setup``,
    rollout slot, timed ``run``, teardown -- is :func:`run_rollout_with_bounds`, the
    same function training calls, so both share one definition of how a container
    rollout is sequenced and bounded. This wrapper only supplies what is eval-specific:
    the session state, the endpoint's capture session, the token-trace drain, and the
    S3 dump. One thing it deliberately does *not* supply is the ``task["llm"]`` config:
    that is perishable and this coroutine is created long before it runs, so
    :class:`LateBoundLlmSession` builds it inside the bounded run instead.

    The session state is a :class:`PersistentDict` that persists itself to DynamoDB on
    every mutation (here and inside the bounded run), so an in-flight rollout is
    observable as it progresses. Returns that meta as a plain dict -- reward, metrics
    and token counts folded in, so the report reduces one flat row per rollout. Never
    raises: a failed rollout is recorded with ``aborted=True`` and a null reward so it
    still counts in pass@k denominators.
    """
    session_id = SESSION_PREFIX + uuid4().hex
    endpoint = config.endpoint
    index = task_index(task_row)
    task_id = str(index)

    # meta is a PersistentDict over this session's item in the session table -- the
    # same table training writes to, here named out of the caller's ``env`` mapping.
    # Every mutation below persists itself, so DynamoDB tracks a rollout live.
    meta = PersistentDict(
        {
            "session_id": session_id,
            "experiment_name": config.experiment_name,
            "experiment_start_at": experiment_start_at,
            "experiment_start_at_session_id": f"{experiment_start_at}:{session_id}",
            "harness": "batch_agent_eval",
            "task_id": task_id,
            "instance_id": task_row.get("instance_id"),
            "n_idx": n_idx,
            "model": endpoint.model,
            "endpoint": endpoint.label(),
            "temperature": config.temperature,
            "top_p": config.top_p,
            "eval_start_at": dt.datetime.now(),
        },
        persister=DynamoDBPersister(
            session_table,
            {"session_id": session_id},
            region_name=storage_region,
        ),
    )

    sampling_params = dict(temperature=config.temperature, top_p=config.top_p)
    task = dict(task_row)
    task["sampling_params"] = sampling_params
    task.update(config.task_kwargs)
    # No task["llm"] yet: it is perishable, and this coroutine is created long before
    # it runs. LateBoundLlmSession fills it in once the bounded run reaches `run`.

    # A fresh session per rollout, as on the training side. It shares this rollout's
    # `meta` as its session_state, so the session's own keys (runtime_arn, its timing
    # spans) land in the same session item.
    session = LateBoundLlmSession(
        AgentCoreSession(
            session_id,
            meta,
            runtime_arn=runtime_arn,
            capacity_provider_arn=capacity_provider_arn,
        ),
        lambda: endpoint.build_task_llm(session_id, config.temperature, config.top_p),
    )

    # The initial state is adopted, not written, by the constructor; persist it so
    # the session shows up before the rollout starts.
    await meta.persist()

    rollout: RolloutDumpResponse | None = None
    records: list[TraceRecord] = []
    exception: str | None = None
    try:
        # Register the capture session before the agent can make any LLM call
        # (no-op for the Bedrock endpoint), as the training side does.
        endpoint.open_session(session_id, sampling_params)
        rollout = await run_rollout_with_bounds(
            meta,
            bounds,
            # A group is one task and its n samples, so the task id is the priority
            # key: all n samples of a task queue at one priority, in task arrival
            # order (training adds the step to the same key).
            task_id,
            session,
            task,
        )
        # A dump comes back for a rollout that failed inside the container too, so
        # the response is asked rather than assumed: its metrics are kept either way,
        # while an unsuccessful one counts as aborted and its reward is not read.
        # Only a failure on this side (timeout, transport) raises.
        exception = rollout.failure_reason()
        if exception is not None:
            logger.error(
                "rollout %s (instance=%s n_idx=%s) failed in container: %s",
                session_id,
                meta["instance_id"],
                n_idx,
                exception,
            )
        await meta.set("aborted", exception is not None)
    except Exception as error:
        logger.error("rollout %s (instance=%s n_idx=%s) failed: %r", session_id, meta["instance_id"], n_idx, error)
        exception = exception_to_string(error)
        await meta.set("aborted", True)
    finally:
        # Drain the gateway session (frees its trajectory tree and cancels any
        # in-flight turn). The container is already gone: the bounded run tears it
        # down on the way out of its `async with session`, on success or failure.
        try:
            records = await endpoint.finish_session(session_id)
        except Exception as error:
            logger.error("finish_session %s failed: %r", session_id, error)

    # Reward and resolved are the only fields named downstream (pass@k needs them);
    # the dump's metrics dict rides along under its own key names. Reward comes off a
    # successful rollout only -- an aborted one reports null, which is what keeps it
    # in the pass@k denominator without counting as a pass -- while the metrics come
    # off whatever dump exists, so a failed attempt's timings are not lost.
    reward = rollout.reward if rollout is not None and rollout.is_successful() else None
    metrics = dict(rollout.metrics) if rollout is not None else {}
    resolved = bool(reward) if reward is not None else False

    # Token counts exist only where the trajectory was captured (gateway path);
    # None, not zeros, for an endpoint that captures nothing.
    tokens = token_stats(records) if endpoint.captures_tokens() else None

    rollout_summary = {"reward": reward, "resolved": resolved, **metrics}
    if tokens is not None:
        rollout_summary.update(tokens)

    # One persisted update for everything known once the rollout has settled.
    await meta.update(
        {
            "eval_end_at": dt.datetime.now(),
            **rollout_summary,
        }
    )

    s3_uri = f"{s3_prefix}/{session_id}"
    dump = {
        "meta": meta.snapshot(),
        "task": task,
        "rollout_dump_response": rollout.model_dump(mode="json") if rollout is not None else None,
        "trace_records": [_trace_to_dict(r) for r in records],
        "exception": exception,
    }
    await upload_object(
        s3_uri=s3_uri,
        region_name=storage_region,
        data=json.dumps(dump, default=str).encode(),
    )

    await meta.set("output_s3_uri", s3_uri)

    # The row IS the session meta: reward/resolved, the session's metrics, the token
    # counts and every lifecycle timing span were folded into it above, flat, so the
    # report can reduce whatever is numeric without knowing any of their names.
    return dict(meta)


def build_report(config: EvalConfig, experiment_start_at: str, rows: list[dict]) -> dict:
    """This run's rows as a report: the shared summary, plus what produced it.

    The arithmetic is :func:`~rollout_report.summarize`, deliberately not this
    module's: the same rows are in the session table, so the same summary can be
    recomputed from there for any past run, and a second implementation here is how
    the two would come to disagree. What this adds is what only the driver knows --
    the config the run was launched with -- and the rows themselves, so a report stays
    a self-contained artifact even though it is no longer a source of truth.

    ``k`` is passed explicitly rather than observed from the rows so that a run cut
    short still reports pass@n for the ``n`` it was asked for.
    """
    return {
        "config": dataclasses.asdict(config),
        "experiment_start_at": experiment_start_at,
        **summarize(rows, k=config.n),
        "rollouts": rows,
    }


async def start_eval_ec2_monitor(config: EvalConfig, env: dict) -> EC2Monitor | None:
    """Start the EC2 -> session instance-id poller for this eval, or ``None``.

    The eval-side counterpart of the trainer entrypoint's monitor startup:
    same gating (needs a capacity provider, a session table and a positive poll
    interval), same background-task hosting on the caller's loop. ``None`` means
    the run simply goes unstamped -- nothing downstream depends on the monitor.
    """
    table = env.get("agent_dynamodb_table")
    capacity_provider_arn = env.get("agentcore_capacity_provider_arn")
    if config.ec2_monitor_poll_interval <= 0 or not table or not capacity_provider_arn:
        logger.info("ec2 monitor disabled for experiment=%s", config.experiment_name)
        return None

    monitor = EC2Monitor(
        capacity_provider_arn,
        table,
        session_prefix=SESSION_PREFIX,
        poll_interval=config.ec2_monitor_poll_interval,
        region_name=env["aws_region"],
    )
    await monitor.start()
    return monitor


async def run_eval(config: EvalConfig, env: dict) -> dict:
    """Run one EvalConfig: start the endpoint and monitor, fan out the samples, aggregate.

    The AgentCore ARNs, the session table and the output bucket come from ``env``;
    anything the *task* needs rides in ``config.task_kwargs`` instead.

    Two things live for the whole eval rather than per rollout: the inference
    endpoint and the :class:`EC2Monitor`.
    """
    # The only two uses of polars and tqdm here, both belonging to this driver rather
    # than to the pieces other callers import. Imported inside the function for the
    # same reason transformers and the Bedrock token generator are further down: it
    # keeps importing this module a base-install operation, while these two ship in the
    # ``swe-agent`` dependency group alongside the scripts that call ``run_eval``.
    import polars as pl
    from tqdm import tqdm

    runtime_arn = env["agentcore_runtime_arn"]
    capacity_provider_arn = env["agentcore_capacity_provider_arn"]
    # required here, unlike in the monitor: every rollout's session state is written
    # through it, so an eval without a table would report nothing to look at later.
    session_table = env["agent_dynamodb_table"]
    # The region the session table and the dump bucket live in -- ``env``'s single
    # ``aws_region``, so eval writes land beside training's.
    storage_region = env["aws_region"]
    experiment_start_at = dt.datetime.now().isoformat()
    s3_prefix = f"{env['rollout_output_s3']}/{experiment_start_at}"

    frame = pl.read_parquet(config.dataset)
    if config.num_tasks is not None:
        frame = frame.head(config.num_tasks)
    task_rows = frame.to_dicts()
    assert task_rows, f"dataset produced no tasks: {config.dataset}"

    bounds = local_bounds(config)

    logger.info(
        "experiment=%s endpoint=%s model=%s tasks=%d n=%d concurrency=%d capture=%s -> %d rollouts",
        config.experiment_name,
        config.endpoint.label(),
        config.endpoint.model,
        len(task_rows),
        config.n,
        config.concurrency,
        config.endpoint.captures_tokens(),
        len(task_rows) * config.n,
    )

    await config.endpoint.start()
    monitor = await start_eval_ec2_monitor(config, env)

    total = len(task_rows) * config.n
    bar = tqdm(total=total, desc=config.experiment_name, unit="rollout")
    resolved = 0
    aborted = 0

    async def _tracked(coro):
        """Run one rollout and tick the progress bar as it settles."""
        nonlocal resolved, aborted
        row = await coro
        resolved += bool(row["resolved"])
        aborted += bool(row["aborted"])
        bar.set_postfix(resolved=resolved, aborted=aborted, refresh=False)
        bar.update(1)
        return row

    try:
        coros = [
            _tracked(
                run_one(
                    config,
                    runtime_arn,
                    capacity_provider_arn,
                    experiment_start_at,
                    task_row,
                    n_idx,
                    bounds,
                    s3_prefix,
                    session_table,
                    storage_region,
                )
            )
            for task_row in task_rows
            for n_idx in range(config.n)
        ]
        rows = await asyncio.gather(*coros)
    finally:
        bar.close()
        await config.endpoint.stop()
        # The monitor owns the only reference to its polling task, so it has to be
        # stopped here or the task outlives the eval it was stamping for.
        if monitor is not None:
            await monitor.stop()

    report = build_report(config, experiment_start_at, rows)

    # Written once, at the end, and nothing reads it as an authority: every row in it
    # was already persisted to the session table as the rollout produced it, and can be
    # summarized again from there, so a run whose driver dies here loses a convenience
    # rather than its results.
    os.makedirs(config.report_dir, exist_ok=True)
    report_path = os.path.join(config.report_dir, f"{config.experiment_name}.json")
    with open(report_path, "w") as handle:
        json.dump(report, handle, indent=2, default=str)

    logger.info(
        "experiment=%s done: pass@%d=%.3f mean_reward=%s report=%s",
        config.experiment_name,
        config.n,
        report["reward"]["pass_at_k"] or 0.0,
        report["reward"]["mean_reward"],
        report_path,
    )
    return report
