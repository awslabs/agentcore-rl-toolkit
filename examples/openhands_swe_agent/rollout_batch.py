"""Batch evaluation harness for container-based agents.

Holds the machinery only: the inference :class:`Endpoint` flavors, the :class:`EvalConfig`
shape and :func:`run_eval`, which drives one config's rollouts through the same bounded
session lifecycle training uses. The datasets and config grid live in ``evaluate.py``.
"""

import asyncio
import dataclasses
import datetime as dt
import json
import logging
import os
from typing import Callable, Protocol, runtime_checkable
from uuid import uuid4

# Sibling module of the ``./evaluate.py`` entrypoint; ``conftest.py`` puts the same
# directory on ``sys.path`` so tests resolve it identically.
from rollout_report import summarize

from agentcore_rl_toolkit.aws_tools.boto3_tools import LongLivedCredentials
from agentcore_rl_toolkit.aws_tools.ec2_monitor import DEFAULT_POLL_INTERVAL, EC2Monitor
from agentcore_rl_toolkit.aws_tools.persistent_dict import DynamoDBPersister, PersistentDict
from agentcore_rl_toolkit.aws_tools.s3_tools import upload_object
from agentcore_rl_toolkit.concurrency.priority_assigner import LocalPriorityAssigner
from agentcore_rl_toolkit.concurrency.priority_semaphore import LocalPrioritySemaphore
from agentcore_rl_toolkit.concurrency.rate_limiter import ACRRateLimiter

# BaseTrace/TraceRecord are torch-free and aiohttp-free; the heavy gateway pieces are
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

# Distinguishes eval sessions from training ones in the session table, and narrows the
# EC2 monitor's instance scan to this run.
SESSION_PREFIX = "eval_"


# --- endpoints: the one seam that differs from a real rollout -----------------


@runtime_checkable
class Endpoint(Protocol):
    """How the agent reaches inference, and whether the run captures tokens.

    ``start``/``stop`` are per experiment, the session methods per rollout.
    ``build_task_llm`` returns the ``task["llm"]`` client config, or ``None`` when the
    endpoint has no inference (the task then carries no ``llm`` key).
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
    """No inference at all, for no-LLM agent types (``oracle``, ``noop``).

    ``build_task_llm`` returns ``None`` so no ``llm`` key reaches the agent server and a
    session that did read it fails loudly. ``model`` is a label only.
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


# A Bedrock bearer token is a SigV4-presigned URL: it dies with its signing credentials
# whatever X-Amz-Expires claims, and resolved credentials skip provide_token's per-call
# credential-chain walk, which is what gets rate limited in bursts.
_bedrock_credentials = LongLivedCredentials(role_session_name="batch-agent-eval")


def bedrock_token(region: str) -> str:
    """Mint a Bedrock bearer token; call it from inside the bounded run, since the
    token's life starts here."""
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

    # Cap the advertised expiry at the credentials' own, so an outlived token fails as
    # plainly expired rather than as an opaque auth error.
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

    ``model`` is a LiteLLM model id whose ``openai/`` prefix routes at ``base_url``.
    Every rollout mints its own bearer token, from inside the bounded run.
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
    """The agent talks to a rollout gateway fronting the vLLM server at ``vllm_url``.

    ``start`` serves the training capture layer on a background thread (no verl/Ray);
    the gateway is a per-experiment singleton shared by every concurrent rollout, and
    ``finish_session`` drains one rollout's trajectory into ``TraceRecord``s.
    ``tokenizer_path`` must match the served model.
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
        # gateway keys token capture off it.
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
        # Drop turns that produced nothing trainable, as the training side does.
        return [r for r in records if r.token_ids]


@dataclasses.dataclass
class EvalConfig:
    """One evaluation run: a (dataset slice, endpoint, sampling, concurrency) point.

    ``dataset`` is a parquet whose rows are tasks in the schema training reads;
    ``num_tasks`` caps the rows used (None = all) and ``n`` is samples per task.
    ``task_kwargs`` is merged into every task and passed through uninterpreted, so which
    keys an agent needs is a property of this config, not of the driver.
    """

    experiment_name: str
    endpoint: Endpoint
    dataset: str
    num_tasks: int | None
    n: int
    concurrency: int  # max in-flight containers
    session_create_rate: float  # container/session creation req/sec ceiling
    timeout: int  # per-rollout agent_run wall-clock budget (seconds)
    # Written as `{report_dir}/{experiment_name}.json`. No default: only the driver knows
    # where the recipe's artifacts belong. Resolved when the run finishes, so a relative
    # path answers to whatever cwd the driver happened to have.
    report_dir: str
    # max concurrent agent runs (None = `concurrency`, i.e. non-binding). Lower it to
    # hold containers warm while throttling how many talk to inference at once.
    rollout_concurrency: int | None = None
    container_setup_timeout: int = 1200  # per-rollout container provisioning budget
    temperature: float = 1.0
    top_p: float = 1.0
    task_kwargs: dict = dataclasses.field(default_factory=dict)
    # seconds between EC2 monitor polls; 0 disables it.
    ec2_monitor_poll_interval: float = DEFAULT_POLL_INTERVAL


def task_index(task_row: dict) -> int:
    """The dataset's own index for a task row, resolved as verl resolves it.

    verl's ``RLHFDataset`` promotes ``extra_info["index"]`` to a top-level ``index``,
    which the training side reads as the task id; reading it the same way means an eval
    task id names the same dataset row rather than a number that shifts with the slice.
    """
    if "index" in task_row:
        return task_row["index"]
    return (task_row.get("extra_info") or {}).get("index", 0)


def token_stats(records: list[TraceRecord]) -> dict:
    """Token counts for one trajectory, captured as one record per trainable turn.

    Records overlap (a turn's prompt is the previous turns replayed), so ``num_tokens``
    is the trainer's cost of the trajectory rather than the conversation's length, and
    ``max_record_tokens`` -- not the sum -- is what bumps ``max_context_tokens``.
    ``num_generated_tokens`` is the loss-masked span the policy update trains on.
    """
    token_counts = [len(record.token_ids) for record in records]
    return {
        "num_records": len(records),
        "num_tokens": sum(token_counts),
        "num_generated_tokens": sum(sum(record.loss_mask) for record in records),
        "max_record_tokens": max(token_counts, default=0),
    }


def _trace_to_dict(record: TraceRecord) -> dict:
    """JSON-safe view of a TraceRecord."""
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
    """Fill :class:`ContainerBounds` with process-local implementations (training fills
    it with Ray actors). Built once per experiment, which is what makes the semaphores
    cap anything."""
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

    Rollout coroutines are all created up front, so a perishable Bedrock bearer token
    minted at dispatch would rot while its rollout waits for a container slot. ``run`` is
    the last point that still works -- the agent server reads ``task_input["llm"]`` only
    from the rollout-start payload -- so a token's age is bounded by ``agent_run_timeout``
    rather than by the queue. The rest of :class:`RolloutSession` is delegated untouched.
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
        # A ``None`` config means the endpoint has no inference, and the key must stay
        # absent rather than be set to None.
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

    The bounded lifecycle itself is :func:`run_rollout_with_bounds`, the same function
    training calls; this wrapper adds the session state, the capture session, the
    token-trace drain and the S3 dump. Returns the session meta as a flat dict so the
    report reduces one row per rollout. Never raises: a failed rollout is recorded with
    ``aborted=True`` and a null reward so it still counts in pass@k denominators.
    """
    session_id = SESSION_PREFIX + uuid4().hex
    endpoint = config.endpoint
    index = task_index(task_row)
    task_id = str(index)

    # A PersistentDict over this session's item in the table training also writes to;
    # every mutation below persists itself, so DynamoDB tracks a rollout live.
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
    # No task["llm"] yet: LateBoundLlmSession fills it in once the bounded run reaches
    # `run`, and shares this rollout's `meta` so session keys land in the same item.
    session = LateBoundLlmSession(
        AgentCoreSession(
            session_id,
            meta,
            runtime_arn=runtime_arn,
            capacity_provider_arn=capacity_provider_arn,
        ),
        lambda: endpoint.build_task_llm(session_id, config.temperature, config.top_p),
    )

    # The constructor adopts the initial state without writing it.
    await meta.persist()

    rollout: RolloutDumpResponse | None = None
    records: list[TraceRecord] = []
    exception: str | None = None
    try:
        # Register the capture session before the agent can make any LLM call.
        endpoint.open_session(session_id, sampling_params)
        rollout = await run_rollout_with_bounds(
            meta,
            bounds,
            # A group is one task and its n samples, so the task id is the priority key.
            task_id,
            session,
            task,
        )
        # A dump comes back for a rollout that failed inside the container too, so the
        # response is asked rather than assumed; only failures on this side raise.
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
        # Frees the trajectory tree and cancels any in-flight turn. The container is
        # already gone: the bounded run tears it down on the way out.
        try:
            records = await endpoint.finish_session(session_id)
        except Exception as error:
            logger.error("finish_session %s failed: %r", session_id, error)

    # An aborted rollout reports a null reward, which keeps it in the pass@k denominator
    # without counting as a pass, while metrics come off whatever dump exists.
    reward = rollout.reward if rollout is not None and rollout.is_successful() else None
    metrics = dict(rollout.metrics) if rollout is not None else {}
    resolved = bool(reward) if reward is not None else False

    # None, not zeros, for an endpoint that captures nothing.
    tokens = token_stats(records) if endpoint.captures_tokens() else None

    rollout_summary = {"reward": reward, "resolved": resolved, **metrics}
    if tokens is not None:
        rollout_summary.update(tokens)

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

    # The row IS the session meta: reward/resolved, metrics, token counts and timing
    # spans were all folded into it above, flat, under no names the report has to know.
    return dict(meta)


def build_report(config: EvalConfig, experiment_start_at: str, rows: list[dict]) -> dict:
    """This run's rows as a report: :func:`~rollout_report.summarize`'s numbers, plus the
    config that produced them and the rows themselves.

    The arithmetic is deliberately not duplicated here: the same rows are in the session
    table, so any past run can be re-summarized from there. ``k`` is passed explicitly so
    that a run cut short still reports pass@n for the ``n`` it was asked for.
    """
    return {
        "config": dataclasses.asdict(config),
        "experiment_start_at": experiment_start_at,
        **summarize(rows, k=config.n),
        "rollouts": rows,
    }


async def start_eval_ec2_monitor(config: EvalConfig, env: dict) -> EC2Monitor | None:
    """Start the EC2 -> session instance-id poller for this eval, or ``None``.

    ``None`` means the run simply goes unstamped -- nothing downstream depends on it.
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
    """
    # Imported here, not at module scope: these ship in the ``swe-agent`` dependency
    # group, so importing this module stays a base-install operation.
    import polars as pl
    from tqdm import tqdm

    runtime_arn = env["agentcore_runtime_arn"]
    capacity_provider_arn = env["agentcore_capacity_provider_arn"]
    # Required here, unlike in the monitor: every rollout's state is written through it.
    session_table = env["agent_dynamodb_table"]
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
        # The monitor owns the only reference to its polling task.
        if monitor is not None:
            await monitor.stop()

    report = build_report(config, experiment_start_at, rows)

    # A convenience snapshot: every row in it was already persisted to the session table
    # as the rollout produced it, and can be summarized again from there.
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
