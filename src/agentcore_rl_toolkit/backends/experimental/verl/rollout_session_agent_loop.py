"""The verl v1 agent loop for agents that implement the rollout session protocol.

One instance per rollout. It owns the verl-facing contracts -- token budgets, trajectory
rows, staleness tags, reward and metric plumbing -- and delegates everything about the
container to a composed :class:`RolloutSession` chosen by ``rollout_session_backend.backend``,
driven under the cluster-wide limits in :class:`RolloutSessionBounds`.

Two behaviours are worth knowing before reading the code:

* **What a failed rollout contributes is configurable.** Whatever goes wrong on either side
  of the wire, the S3 dump is written first and then ``on_rollout_failure`` decides what verl
  sees: ``raise`` (the default) lets the failure mark the prompt group, ``empty`` returns no
  rows, ``inert_row`` returns one masked zero-reward row (:meth:`make_failed_loop_output`).
  None of the three is known to train best -- the knob exists to compare them; see
  :class:`RolloutSessionAgentLoopConfig` for what each costs. Two exceptions ignore the
  setting and always leave the loop: :class:`RolloutContractError` -- a broken dataset or
  reward function, which every rollout of the run would hit -- and cancellation.
* **A failed rollout's partial trace is discarded.** A trajectory that exists only because
  the container died halfway is still a real sample, but its zero reward describes the
  infrastructure rather than the policy; training it would push the group's advantages
  around for a reason the policy cannot learn from. ``backends/verl/agent_loop.py`` trains
  those partial traces; this loop deliberately does not.

Inside verl, the three settings differ like this (verl 0.9.0):

* ``raise`` -- ``_run_prompt`` catches the exception and marks the group ``failure``. The sync
  replay buffer leaves the group sampleable (``sync_refill_failed_groups=False``), so the
  healthy siblings still train; ``ReplayBufferAsync`` treats ``failure`` as terminal and
  evicts *and refills* the whole group, healthy siblings included.
* ``empty`` -- ``_agent_loop_postprocess`` logs the empty output and writes nothing to the
  TransferQueue. The group stays ``finished`` with fewer than ``n`` rows and every mode trains
  the survivors, but nothing about the failure reaches the step's metrics (no ``extra_fields``
  are written), and a group whose rollouts *all* fail raises -- from DAPO's group filter, or
  from the sync buffer's "no materializable trajectories" guard.
* ``inert_row`` -- a real row, so metrics and row counts survive. verl does not backfill a
  missing group member either way, so this is the only setting under which the group keeps its
  full width -- at the price of a synthetic zero inside it: a group that scored ``[1, 1, 1]``
  plus a failure trains as ``[1, 1, 1, 0]``.
"""

import datetime as dt
import logging
import os
from typing import Any, NotRequired, TypedDict
from uuid import uuid4

import numpy as np
import torch
from hydra.utils import instantiate
from pydantic import BaseModel
from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopMetrics, AgentLoopOutput, register

from agentcore_rl_toolkit.aws_tools.persistent_dict import (
    DynamoDBPersister,
    NullPersister,
    PersistentDict,
    measure_span_persistent,
)
from agentcore_rl_toolkit.aws_tools.s3_tools import upload_object
from agentcore_rl_toolkit.backends.verl.gateway_host import GatewayHandle, get_or_start_gateway
from agentcore_rl_toolkit.rollout_gateway import BaseTrace, TraceRecord
from agentcore_rl_toolkit.rollout_session.errors import RolloutContractError
from agentcore_rl_toolkit.rollout_session.exception_utils import exception_to_string
from agentcore_rl_toolkit.rollout_session.factory import make_session
from agentcore_rl_toolkit.rollout_session.lifecycle import (
    RolloutSession,
    require_task_id,
    run_rollout_with_bounds,
)
from agentcore_rl_toolkit.rollout_session.wire import RolloutDumpResponse

from .config import RolloutSessionAgentLoopConfig
from .rollout_session_resources import get_rollout_session_bounds

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

logging.getLogger("backoff").setLevel(logging.ERROR)

# The canonical name of the rollout's own score inside ``reward_extra_info``. DAPO group
# filtering reads ``algorithm.filter_groups.metric`` from that dict, and the v1 replay buffer
# raises when a finished trajectory lacks the configured key -- so it must be present on
# every row, failures included, and must be the same number the advantage is computed from.
REWARD_EXTRA_INFO_SCORE_KEY = "reward_score"

# Accepted ``on_rollout_failure`` values; see RolloutSessionAgentLoopConfig for what each does.
ON_ROLLOUT_FAILURE_MODES = ("raise", "empty", "inert_row")


class RolloutFailedError(RuntimeError):
    """A rollout failed and ``on_rollout_failure=raise``, with no trainer-side exception to
    re-raise -- the container reported the failure itself, so all we have is its reason."""


def _int_or(value: Any, default: int) -> int:
    """``value`` as an int, or ``default`` when it is absent. ``0`` is a real weight version."""
    return default if value is None else int(value)


def build_llm(model: str, sampling_params: dict[str, Any], inference_url: str, session_id: str, timeout: int):
    llm = dict(
        model=f'openai/{model.split("/")[-1]}',
        base_url=inference_url,
        api_key=session_id,
        temperature=sampling_params["temperature"],
        top_p=sampling_params["top_p"],
        num_retries=0,
        timeout=timeout,
    )
    return llm


class ExtraFields(TypedDict):
    """Extra fields on every loop output; verl requires the same key set for a whole run.

    ``metrics`` entries are each reduced by the trainer to agent_loop/<name>/{mean,min,max}.
    """

    failure_reason: str | None
    max_global_steps: int
    metrics: dict[str, Any]
    min_global_steps: int
    num_trace_records: int
    request_id: str
    # verl only populates this itself when reward_score is None; we set reward_score
    # directly, so supply the key to avoid a KeyError in _agent_loop_postprocess.
    reward_extra_info: dict
    # 1.0 on the stand-in row a failed rollout emits under on_rollout_failure=inert_row,
    # 0.0 on every real row. The other two settings emit no row, so under them a failure
    # is absent from this field rather than reported as 1.0.
    rollout_failed: float
    trace_index: int
    trace_metadata: dict


class AgentSessionMeta(TypedDict):
    aborted: NotRequired[bool]
    experiment_name: str
    experiment_start_at: dt.datetime
    experiment_start_at_session_id: str
    step: NotRequired[int]
    output_s3_uri: NotRequired[str]
    session_id: str
    task_id: NotRequired[str]
    verl_loop_start_at: NotRequired[dt.datetime]
    verl_loop_end_at: NotRequired[dt.datetime]


class RolloutOutput(BaseModel):
    """Detailed rollout output uploaded to S3 (includes large, variable-size objects)."""

    task: dict[str, Any]
    agent_loop_outputs: list[AgentLoopOutput]
    exception: str | None
    meta: AgentSessionMeta
    rollout_dump_response: RolloutDumpResponse | None


@register("rollout_session_agent_loop")
class RolloutSessionAgentLoop(AgentLoopBase):
    """An agent loop over an HTTP agent server running in a container.

    Holds the verl-facing contracts; the container lifecycle is delegated to a composed
    ``RolloutSession`` chosen by the ``rollout_session_backend.backend`` config field.
    """

    def __init__(
        self,
        trainer_config,
        server_manager,
        tokenizer,
        processor,
        dataset_cls,
        data_config,
        **kwargs,  # swallows the YAML entry's `name`, verl's `tools`, and future kwargs
    ):
        super().__init__(trainer_config, server_manager, tokenizer, processor, dataset_cls, data_config, **kwargs)

        if not self.config.trainer.get("use_v1", False):
            raise ValueError(
                "RolloutSessionAgentLoop requires trainer.use_v1=true: it returns "
                "list[AgentLoopOutput] (one per trajectory-tree leaf), which only the v1 "
                "TransferQueue path consumes."
            )

        self.loop_config: RolloutSessionAgentLoopConfig = instantiate(
            self.config.rollout_session_agent_loop, _convert_="all"
        )
        # Validated here rather than at the failure site: a typo would otherwise sit unnoticed
        # until the first rollout failed, hours into a run, and then change what training sees.
        if self.loop_config.on_rollout_failure not in ON_ROLLOUT_FAILURE_MODES:
            raise ValueError(
                f"rollout_session_agent_loop.on_rollout_failure must be one of "
                f"{list(ON_ROLLOUT_FAILURE_MODES)}, got {self.loop_config.on_rollout_failure!r}"
            )

        # extract verl config. rollout.* rather than data.max_*: those are the lengths of the
        # regions a trajectory is actually stored in, and only rollout carries max_model_len.
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length
        self.max_model_len = self.rollout_config.max_model_len
        self._validate_token_budgets()
        # The gateway cannot emit more tokens than verl's response region can store. The two
        # padded regions may sum past max_model_len; valid tokens never do.
        self.max_context_tokens = self.response_length

        self.model = self.config.actor_rollout_ref.model.path
        self.experiment_name = self.config.trainer.experiment_name
        self.experiment_start_at = self.config.trainer.experiment_start_at

        self._rei_defaults = {str(k): float(v) for k, v in (self.loop_config.reward_extra_info_defaults or {}).items()}
        # verl already derives its `reward` validation metric from rm_scores.
        self._rei_defaults.pop("reward", None)

        self.session_id = "verl_" + uuid4().hex

        if self.loop_config.dynamodb_table is not None:
            persister = DynamoDBPersister(
                self.loop_config.dynamodb_table,
                {"session_id": self.session_id},
                region_name=self.loop_config.aws_region,
            )
        else:
            persister = NullPersister()

        self.meta = PersistentDict(data={"session_id": self.session_id}, persister=persister)

        gateway_kwargs = dict(self.loop_config.rollout_gateway)
        # The renderer must tokenize turns exactly as training will, so the trainer's own
        # chat-template kwargs (e.g. enable_thinking) have to reach the gateway.
        gateway_kwargs.setdefault("chat_template_kwargs", dict(self.apply_chat_template_kwargs or {}))
        self._gateway: GatewayHandle = get_or_start_gateway(
            server_manager=server_manager,
            tokenizer=tokenizer,
            **gateway_kwargs,
        )

        self.bounds = get_rollout_session_bounds(self.loop_config.rollout_session_bounds)

        # the session shares this loop's session_id and meta dict
        self.rollout_session: RolloutSession = self._make_session()

        # One instance per rollout: the session id, the gateway capture key, the session
        # record and the `verl_loop` span are all per-rollout state.
        self._ran = False

    def _make_session(self) -> RolloutSession:
        return make_session(
            self.session_id,
            self.loop_config.rollout_session_backend,
            self.meta,
        )

    def _validate_token_budgets(self) -> None:
        """Reject length budgets a trajectory could not be stored in, at construction."""
        if self.max_model_len is None or self.max_model_len <= 0:
            raise ValueError(
                "RolloutSessionAgentLoop requires an explicit positive "
                "actor_rollout_ref.rollout.max_model_len. This is the model context capacity; "
                "verl validates it against the Hugging Face model config."
            )
        for field, value in (("prompt_length", self.prompt_length), ("response_length", self.response_length)):
            if value > self.max_model_len:
                raise ValueError(
                    f"rollout.{field} ({value}) cannot exceed rollout.max_model_len ({self.max_model_len})"
                )
        max_new_tokens = self.loop_config.rollout_gateway_sampling_params.get("max_new_tokens")
        if max_new_tokens is None:
            return
        if type(max_new_tokens) is not int or max_new_tokens <= 0:  # noqa: E721 - reject bool
            raise ValueError(
                f"rollout_gateway_sampling_params.max_new_tokens must be a positive integer, got {max_new_tokens!r}"
            )
        if max_new_tokens > self.max_model_len:
            raise ValueError(
                f"rollout_gateway_sampling_params.max_new_tokens ({max_new_tokens}) cannot exceed "
                f"rollout.max_model_len ({self.max_model_len})"
            )

    # Returning a list deliberately widens AgentLoopBase.run's annotation: the v1 TQ path
    # accepts AgentLoopOutput | list[AgentLoopOutput] (one row per trajectory-tree leaf);
    # __init__ asserts trainer.use_v1 accordingly.
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> list[AgentLoopOutput]:  # type: ignore[override]
        if self._ran:
            raise RuntimeError(
                f"RolloutSessionAgentLoop.run was called twice on {self.session_id}. One instance "
                "runs one rollout: the session id, its gateway capture key and its session record "
                "are per-rollout state, so a second run would append to the first one's trajectory."
            )
        self._ran = True

        # Built before any container or session state exists: a task-contract violation is a
        # config error that would hit every rollout, so it raises here rather than being
        # absorbed by whatever on_rollout_failure does with a failed rollout.
        task = self.make_task(sampling_params, kwargs)
        task_id = require_task_id(task)

        async with self.meta.connection():
            await self.meta.update(
                experiment_name=self.experiment_name,
                experiment_start_at=self.experiment_start_at,
                experiment_start_at_session_id=f"{self.experiment_start_at}:{self.session_id}",
                aborted=False,
                task_id=task_id,
                # v1's staleness metrics do int(tag["min_global_steps"]), and this step stands
                # in for the weight version when the engine reports none -- so it must be a
                # real step number, never a sentinel.
                step=int(kwargs.get("global_steps", 0)),
            )

            async with measure_span_persistent("verl_loop", self.meta):
                rollout_dump_response: RolloutDumpResponse | None = None
                agent_loop_outputs: list[AgentLoopOutput] | None = None
                exception: str | None = None
                failure: Exception | None = None
                try:
                    rollout_dump_response, agent_loop_outputs = await self.run_or_throw(task)
                    if agent_loop_outputs is None:
                        # container-reported failure: its stack trace is already in the
                        # dump, so record and abort instead of raising our own
                        exception = (
                            rollout_dump_response.failure_reason() if rollout_dump_response is not None else None
                        ) or "the rollout produced no trainable trajectory"
                        logger.error(f"Failed rollout {self.session_id} in container: {exception}")
                except Exception as e:
                    logger.error(f"Failed rollout {self.session_id} in trainer: {e}", exc_info=e)
                    exception = exception_to_string(e)
                    failure = e

                rollout_failed = agent_loop_outputs is None
                if rollout_failed:
                    await self.meta.set("aborted", True)
                    agent_loop_outputs = self._failed_rollout_outputs(exception)

                # Written before anything is raised below, so a failure is on record in S3
                # whichever way on_rollout_failure sends it.
                await self.save_to_s3(task, agent_loop_outputs, exception, rollout_dump_response)

                if isinstance(failure, RolloutContractError):
                    # Not a rollout failure: no rollout of this run can satisfy the contract, so
                    # absorbing it would spend the whole job on failures. Raises whatever
                    # on_rollout_failure says.
                    raise failure

                if rollout_failed and self.loop_config.on_rollout_failure == "raise":
                    # verl marks the prompt group `failure`: sync trains the siblings anyway,
                    # the async trainers evict and refill the group.
                    raise failure if failure is not None else RolloutFailedError(exception)

            return agent_loop_outputs

    def _failed_rollout_outputs(self, failure_reason: str | None) -> list[AgentLoopOutput]:
        """The rows a failed rollout contributes to training, per ``on_rollout_failure``.

        Only ``inert_row`` contributes one; ``raise`` and ``empty`` both contribute none and
        differ in what ``run`` does next, not in what it would have returned.
        """
        if self.loop_config.on_rollout_failure == "inert_row":
            return [self.make_failed_loop_output(failure_reason)]
        return []

    async def save_to_s3(self, task, agent_loop_outputs, exception, rollout_dump_response):
        s3_uri = f"{self.loop_config.rollout_output_s3}/{self.experiment_start_at}/{self.session_id}"
        s3_output = RolloutOutput(
            task=task,
            agent_loop_outputs=agent_loop_outputs,
            exception=exception,
            meta=dict(self.meta),  # type: ignore
            rollout_dump_response=rollout_dump_response,
        )
        await upload_object(
            s3_uri=s3_uri,
            region_name=self.loop_config.aws_region,
            data=s3_output.model_dump_json().encode(),
        )
        await self.meta.set("output_s3_uri", s3_uri)

    def make_task(self, sampling_params, kwargs):
        # the TransferQueue runner forwards every batch field into kwargs, including
        # tensors the container does not need and that are not JSON-serializable
        task = {k: v for k, v in kwargs.items() if not isinstance(v, (torch.Tensor, np.ndarray))}
        task.update(self.loop_config.task_kwargs)
        task["sampling_params"] = sampling_params
        inference_url = f"{self._gateway.base_url}/v1"
        task["llm"] = build_llm(
            self.model,
            task["sampling_params"],
            inference_url,
            self.session_id,
            self.loop_config.rollout_session_bounds["agent_run_timeout"],
        )
        return task

    def _group_key(self) -> str:
        """Identify the rollout group (one prompt's rollouts at one step) this session belongs to."""
        return f"{self.meta.get('step', -1)}:{self.meta['task_id']}"

    async def run_or_throw(self, task: dict) -> tuple[RolloutDumpResponse | None, list[AgentLoopOutput] | None]:
        """Run one rollout, raising only on failures of *this* side.

        Returns ``(dump, None)`` when the rollout itself failed: either the container
        reported failure, or it captured no trainable trajectory.
        """

        # Outside the try below: if the session cannot even be created there is nothing to
        # drain, and finish_session on an unknown sid would raise over the real error.
        self._gateway.gateway.create_session(
            self.session_id,
            sampling_defaults=self._sampling_defaults(task["sampling_params"]),
            max_context_tokens=self.max_context_tokens,
        )
        try:
            rollout = await run_rollout_with_bounds(
                self.meta,
                self.bounds,
                self._group_key(),
                self.rollout_session,
                task,
            )
            # A dump describing a failed rollout is data, not an error; keep its metrics for
            # diagnosis either way.
            await self.meta.update(rollout.metrics)
            failed = not rollout.is_successful()
            num_turns = self._gateway.gateway.manager.turn_count(self.session_id)
            engine_extra = self._gateway.backend.pop_extra_fields(self.session_id)
        finally:
            # Always drain: the gateway holds the trajectory tree for this sid until it does.
            records = await self._gateway.gateway.finish_session(
                self.session_id,
                base_sample=BaseTrace(rollout_id=self.session_id),
                reward=0.0,
            )

        if failed:
            # Whatever partial trace exists describes the failure, not the policy -- see the
            # module docstring.
            return rollout, None

        # maybe some responses are empty?
        records = [r for r in records if r.token_ids]
        if not records:
            self._warn_if_static_session_capture()
            return rollout, None

        if len(records) > 1:
            logger.info(
                "session %s forked into %d trace records (trained tokens per record: %s)",
                self.session_id,
                len(records),
                [sum(r.loss_mask) for r in records],
            )
            # verl scores and broadcasts from the last output of a session, so put the primary
            # (most-trained) record last, keeping tree order otherwise.
            primary = max(range(len(records)), key=lambda i: sum(records[i].loss_mask))
            records.append(records.pop(primary))

        # in gateway linear-history mode every record carries the same LinearHealer counters
        healer_stats = records[-1].metadata.get("linear_healer")
        linear_metrics = {f"linear_healer_{k}": float(v) for k, v in (healer_stats or {}).items()}

        # The engine's extra_fields carry the weight versions of the turns it served; a session
        # whose turns never reached this engine (all served before a restart) falls back to the
        # dispatch step rather than inventing staleness.
        dispatch_step = int(self.meta.get("step", 0))
        min_global_steps = _int_or(engine_extra.get("min_global_steps"), dispatch_step)
        max_global_steps = _int_or(engine_extra.get("max_global_steps"), dispatch_step)
        reward_score = float(rollout.reward)  # is_successful() guarantees it is not None

        await self.meta.update(
            num_turns=num_turns,
            # track the true num of records to catch issues
            num_records=len(records),
            min_global_steps=min_global_steps,
            max_global_steps=max_global_steps,
            reward_score=reward_score,
            **linear_metrics,
        )

        reward_extra_info = self._reward_extra_info(rollout, len(records), reward_score)
        outputs = [
            self._record_to_output(
                record,
                index=index,
                num_records=len(records),
                num_turns=num_turns,
                reward_score=reward_score,
                staleness=(min_global_steps, max_global_steps),
                reward_extra_info=reward_extra_info,
            )
            for index, record in enumerate(records)
        ]

        # Session-level totals, so a forked session is not counted once per leaf.
        await self.meta.update(
            llm_generated_length=float(sum(sum(o.response_mask) for o in outputs)),
            context_length=float(sum(len(o.prompt_ids) + len(o.response_ids) for o in outputs)),
        )
        for o in outputs:
            o.extra_fields["metrics"] = dict(self.meta)

        return rollout, outputs

    def _reward_extra_info(
        self,
        rollout: RolloutDumpResponse | None,
        num_records: int,
        reward_score: float,
    ) -> dict[str, float]:
        """The agent's declared scalar metrics, plus this rollout's score.

        Only *declared* keys are carried (``reward_extra_info_defaults``): verl reduces this
        dict across the batch, so a key that only some rollouts report would be averaged over
        the wrong denominator. Every row therefore has the same key set.
        """
        info: dict[str, float] = dict(self._rei_defaults)
        metrics = rollout.metrics if rollout is not None else {}
        for key in self._rei_defaults:
            value = metrics.get(key)
            if isinstance(value, bool | int | float):
                info[key] = float(value)
        info[REWARD_EXTRA_INFO_SCORE_KEY] = reward_score
        info["num_trace_records"] = float(num_records)
        return info

    def _extra_fields(
        self,
        *,
        trace_index: int,
        num_records: int,
        staleness: tuple[int, int],
        reward_extra_info: dict[str, float],
        trace_metadata: dict,
        failed: bool,
        failure_reason: str | None,
    ) -> dict[str, Any]:
        """One output's ``extra_fields``, with the key set verl needs held constant."""
        return dict(
            ExtraFields(
                failure_reason=failure_reason,
                max_global_steps=staleness[1],
                metrics=dict(self.meta),
                min_global_steps=staleness[0],
                num_trace_records=num_records,
                request_id=self.session_id,
                reward_extra_info=dict(reward_extra_info),
                rollout_failed=1.0 if failed else 0.0,
                trace_index=trace_index,
                trace_metadata=dict(trace_metadata),
            )
        )

    def _agent_loop_metrics(self) -> AgentLoopMetrics:
        """verl's fixed three-scalar timing schema, from whatever the session recorded.

        Session-level times reported per output, so a forked session's leaves each carry the
        whole session's timings; verl reduces them across outputs.
        """
        return AgentLoopMetrics(
            # The gateway's own accounting when the session keeps it, else the wall clock of
            # the bounded agent run.
            generate_sequences=float(self.meta.get("llm_latency_sum") or self.meta.get("agent_run", 0.0)),
            tool_calls=float(self.meta.get("tool_calls_time_s", 0.0)),
            compute_score=float(self.meta.get("eval_latency_s", 0.0)),
        )

    def _record_to_output(
        self,
        record: TraceRecord,
        *,
        index: int,
        num_records: int,
        num_turns: int,
        reward_score: float,
        staleness: tuple[int, int],
        reward_extra_info: dict[str, float],
    ) -> AgentLoopOutput:
        response_region_len = len(record.loss_mask)
        prompt_end = len(record.token_ids) - response_region_len
        initial_prompt = list(record.token_ids[:prompt_end])

        # verl stores prompts and responses in fixed-width regions. Preserve the complete
        # token order when the initial prompt exceeds its region by placing the overflow at the
        # start of the response region: these are input tokens, so they stay attended to but
        # carry no policy loss and no rollout logprob.
        prompt_ids = initial_prompt[: self.prompt_length]
        prompt_overflow = initial_prompt[self.prompt_length :]

        # response_length is the whole trajectory's budget (the gateway caps the session at it),
        # so what is left for the response region is that budget minus the stored prompt.
        max_response_length = max(0, self.response_length - len(prompt_ids))
        response_ids = (prompt_overflow + list(record.token_ids[prompt_end:]))[:max_response_length]
        response_mask = ([0] * len(prompt_overflow) + list(record.loss_mask))[:max_response_length]
        response_logprobs = ([0.0] * len(prompt_overflow) + list(record.logprobs))[:max_response_length]

        # The prompt alone can fill the whole budget; an empty response crashes verl's
        # AgentLoopOutput.as_dict, so emit one masked token to keep a valid no-loss sample.
        if not response_ids:
            response_ids = [self._pad_token_id()]
            response_mask = [0]
            response_logprobs = [0.0]

        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=response_logprobs,
            multi_modal_data={},
            # verl counts the initial user turn, which the gateway does not see.
            num_turns=num_turns + 1,
            metrics=self._agent_loop_metrics(),
            reward_score=reward_score,
            extra_fields=self._extra_fields(
                trace_index=index,
                num_records=num_records,
                staleness=staleness,
                reward_extra_info=reward_extra_info,
                trace_metadata=dict(record.metadata),
                failed=False,
                failure_reason=None,
            ),
        )

    def make_failed_loop_output(self, failure_reason: str | None = None) -> AgentLoopOutput:
        """The inert row that stands in for a rollout that failed (``on_rollout_failure=inert_row``).

        Shaped like verl's own padding rows (one prompt token, one masked response token, zero
        reward) so it contributes no loss of its own: with ``response_mask=[0]`` both its
        advantage and its loss are zero whatever the group does. What it does carry is the
        rollout's metrics and the ``rollout_failed`` flag, so the failure is countable in
        ``agent_loop/*``, and a row, so the group keeps its full width.

        It is not neutral for its siblings, though. GRPO baselines the group on the rows it has,
        this zero included, so ``[1, 1, 1]`` plus a failure trains as ``[1, 1, 1, 0]``: every
        sibling's advantage shifts because the container died. That bias is the cost of the
        metrics and the row count, and the reason this is not the default.
        """
        # No engine-reported weight version exists, so stand in the dispatch step: it keeps
        # trajectory_staleness comparable to valid data instead of inflating it to global_steps.
        dispatch_step = int(self.meta.get("step", 0))
        pad = self._pad_token_id()

        return AgentLoopOutput(
            prompt_ids=[pad],
            response_ids=[pad],
            response_mask=[0],
            # Always present, like on a real row: verl derives the rollout_log_probs field from
            # it, and a batch where only some rows have that field cannot be stored.
            response_logprobs=[0.0],
            multi_modal_data={},
            num_turns=0,
            metrics=self._agent_loop_metrics(),
            # Never None: verl only computes rm_scores itself when reward_score is, and this
            # loop's contract is that the agent owns scoring.
            reward_score=0.0,
            extra_fields=self._extra_fields(
                trace_index=0,
                num_records=0,
                staleness=(dispatch_step, dispatch_step),
                reward_extra_info=self._reward_extra_info(None, 0, 0.0),
                trace_metadata={},
                failed=True,
                failure_reason=failure_reason,
            ),
        )

    def _pad_token_id(self) -> int:
        """A token id that carries no training signal, for masked filler positions."""
        eos = getattr(self.tokenizer, "eos_token_id", None)
        return eos if isinstance(eos, int) else 0

    def _warn_if_static_session_capture(self) -> None:
        """Diagnose the stale-agent-image failure mode (see the warning below).

        Warn only, never drop the session: the adapters accept unseen keys by design (that is
        how local runs work) and "EMPTY" is legitimate for local/eval traffic. Without this,
        the misconfiguration trains nothing, silently.
        """
        manager = self._gateway.gateway.manager
        for static_sid in ("EMPTY", "default"):
            if manager.turn_count(static_sid):
                logger.warning(
                    "Rollout %s captured no trace, but turns are accumulating under the static "
                    "session %r -- the deployed agent is likely sending a fixed api_key instead "
                    "of the one the task's `llm` block carries (stale agent image?).",
                    self.session_id,
                    static_sid,
                )
                break

    def _sampling_defaults(self, sampling_params: dict[str, Any]) -> dict[str, Any]:
        """This session's sampling defaults: the configured ones, then verl's."""
        defaults: dict[str, Any] = dict(self.loop_config.rollout_gateway_sampling_params)
        for key in ("temperature", "top_p", "top_k"):
            if key in sampling_params:
                defaults[key] = sampling_params[key]
        return defaults


__all__ = ["RolloutSessionAgentLoop"]
