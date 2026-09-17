"""The verl v1 agent loop for agents that implement the rollout session protocol.

It owns the verl-facing contracts -- token budgets, trajectory rows, staleness tags,
reward and metric plumbing -- and delegates everything about the container to a
composed :class:`RolloutSession` chosen by ``rollout_session_backend.backend``,
driven under the cluster-wide limits in :class:`RolloutSessionBounds`.

Trainer-side exceptions are handled in one of two ways:

* ``raise`` -- ``_run_prompt`` catches the exception and marks the group ``failure``. The sync
  replay buffer leaves the group sampleable (``sync_refill_failed_groups=False``), so the
  healthy siblings still train; ``ReplayBufferAsync`` treats ``failure`` as terminal and
  evicts *and refills* the whole group, healthy siblings included.
* ``empty`` -- ``_agent_loop_postprocess`` logs the empty output and writes nothing to the
  TransferQueue. The group stays ``finished`` with fewer than ``n`` rows and every mode trains
  the survivors, but nothing about the failure reaches the step's metrics (no ``extra_fields``
  are written), and a group whose rollouts *all* fail raises -- from DAPO's group filter, or
  from the sync buffer's "no materializable trajectories" guard.
"""

import logging
import os
from typing import Any, TypedDict
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
from agentcore_rl_toolkit.rollout_session.exception_utils import describe_with_root_cause, exception_to_string
from agentcore_rl_toolkit.rollout_session.factory import SessionBackendConfig, make_session
from agentcore_rl_toolkit.rollout_session.lifecycle import (
    RolloutSession,
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
ON_ROLLOUT_FAILURE_MODES = ("raise", "empty")


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

    max_global_steps: int
    metrics: dict[str, Any]
    min_global_steps: int
    num_trace_records: int
    request_id: str
    # verl only populates this itself when reward_score is None; we set reward_score
    # directly, so supply the key to avoid a KeyError in _agent_loop_postprocess.
    reward_extra_info: dict
    trace_index: int
    trace_metadata: dict


class RolloutOutput(BaseModel):
    """Detailed rollout output uploaded to S3 (includes large, variable-size objects)."""

    task: dict[str, Any]
    agent_loop_outputs: list[AgentLoopOutput] | None
    trainer_exception: str | None
    meta: dict[str, Any]
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

    def _make_session(self) -> RolloutSession:
        return make_session(
            self.session_id,
            SessionBackendConfig(**self.loop_config.rollout_session_backend),
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

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> list[AgentLoopOutput]:  # type: ignore[override]
        # `uid` is the one field a row must carry: it is verl's own prompt-group id, which
        # `make_task` normalizes to the session layer's `group_id`, and which the bounds admit
        # rollouts by. `task_id` is optional here -- recorded as a coordinate when present, and
        # read by no session.
        task = self.make_task(sampling_params, kwargs)

        async with self.meta.connection():
            await self.meta.update(
                # required coordinates
                experiment_name=self.experiment_name,
                experiment_start_at=self.experiment_start_at,
                experiment_start_at_session_id=f"{self.experiment_start_at}:{self.session_id}",
                # optional coordinates
                task_id=str(task.get("task_id", "")),
                step=int(task.get("global_steps", 0)),
                # this is required for priority assignment
                group_id=task["group_id"],
            )

            maybe_agent_loop_outputs = await self.run_rollout_session(task)
            if maybe_agent_loop_outputs is None:
                if self.loop_config.on_rollout_failure == "raise":
                    raise RuntimeError(f"Rollout {self.session_id} has no token records")
                else:
                    return []

            return maybe_agent_loop_outputs

    async def save_to_s3(self, task, agent_loop_outputs, trainer_exception, rollout_dump_response):
        s3_uri = f"{self.loop_config.rollout_output_s3}/{self.experiment_start_at}/{self.session_id}"
        s3_output = RolloutOutput(
            task=task,
            agent_loop_outputs=agent_loop_outputs,
            trainer_exception=exception_to_string(trainer_exception) if trainer_exception is not None else None,
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
        # The session layer groups rollouts by `group_id` and knows nothing about verl; verl's
        # own name for that is `uid`, one uuid4 per prompt shared by its n rollouts. Raw `uid`
        # stays in the task too -- the container sees the row as verl built it.
        task["group_id"] = str(kwargs["uid"])
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

    async def run_rollout_session(self, task: dict) -> list[AgentLoopOutput] | None:
        """Run one rollout and token records if the gateway received any."""

        self._gateway.gateway.create_session(
            self.session_id,
            sampling_defaults=self._sampling_defaults(task["sampling_params"]),
            max_context_tokens=self.max_context_tokens,
        )

        rollout, trainer_exception = None, None
        try:
            async with measure_span_persistent("rollout_session", self.meta):
                rollout = await run_rollout_with_bounds(
                    self.meta,
                    self.bounds,
                    task["group_id"],  # the priority key: one group's rollouts admitted together
                    self.rollout_session,
                    task,
                )
        except Exception as e:
            trainer_exception = e
        finally:
            engine_extra = self._gateway.backend.pop_extra_fields(self.session_id)
            num_turns = self._gateway.gateway.manager.turn_count(self.session_id)
            records = await self._gateway.gateway.finish_session(
                self.session_id,
                base_sample=BaseTrace(rollout_id=self.session_id),
                reward=0.0,
            )

        dump, exc, recs, update = self._process_rollout_result(
            rollout, trainer_exception, records, engine_extra, num_turns
        )
        await self.meta.update(update)
        await self.save_to_s3(task, recs, exc, dump)

        if exc is not None:
            logger.error(
                f"Rollout {self.session_id} has trainer exception " f"= {describe_with_root_cause(exc)}",
                exc_info=trainer_exception,
            )

        if dump is not None and dump.failure_reason() is not None:
            logger.error(f"Rollout {self.session_id} has agent exception = {dump.failure_reason()}")

        return recs

    def _process_rollout_result(self, rollout, trainer_exception, records, engine_extra, num_turns):
        session_update = {}
        agent_exception_str = None
        if rollout is not None:
            agent_exception_str = rollout.failure_reason()
            session_update |= rollout.metrics

        # maybe some responses are empty?
        records = [r for r in records if r.token_ids]

        # aborted flags capture the observed trainer-side events.
        # aborted_by_trainer means that we captured a trainer-side
        # exception that prevented getting the agent's terminal output.
        # aborted_by_agent means the agent returned a response with errors.
        # aborted_by_gateway means there are no tokens to train on
        # aborted_by_trainer and aborted_by_agent are mutually exclusive.
        # aborted_by_gateway is independent of the other abort conditions.
        aborted_by_trainer = trainer_exception is not None
        aborted_by_agent = agent_exception_str is not None
        aborted_by_gateway = len(records) == 0
        aborted = aborted_by_trainer or aborted_by_agent or aborted_by_gateway

        # failures with captured tokens are presumed due to the agent.
        # causally, it's true that the agent can't be responsible for anything before the first token.
        # however, failures after the first token may or may not be due the agent.
        # impls of agent harnesses and rollout sessions may implement appropriate heuristics.

        # no records => no reward, likely an infrastructure/harness failure.
        # has records, no dump => return records with zero reward
        # has records, has dump with reward => return records with given reward
        # has records, has dump without reward => return records with zero reward

        reward_patched, reward_score = None, None
        match (aborted_by_trainer, aborted_by_gateway, aborted_by_agent):
            case (False, False, False):
                assert rollout is not None
                reward_patched = False
                reward_score = float(rollout.reward)
            case (False, False, True):
                reward_patched = True
                reward_score = 0.0
            case (False, True, False):
                assert rollout is not None
                reward_patched = False
                reward_score = float(rollout.reward)
            case (False, True, True):
                reward_patched = False
                reward_score = None
            case (True, False, _):
                reward_patched = True
                reward_score = 0.0
            case (True, True, _):
                reward_patched = False
                reward_score = None

        session_update |= dict(
            aborted_by_gateway=aborted_by_gateway,
            aborted_by_trainer=aborted_by_trainer,
            aborted_by_agent=aborted_by_agent,
            aborted=aborted,
            reward_patched=reward_patched,
            reward_score=reward_score,
        )

        outputs = None
        if not aborted_by_gateway:
            outputs, metrics = self._records_to_outputs(records, rollout, reward_score, engine_extra, num_turns)
            session_update |= metrics
            for output in outputs:
                output.extra_fields["metrics"] = dict(self.meta) | session_update

        return rollout, trainer_exception, outputs, session_update

    def _records_to_outputs(self, records, rollout, reward_score, engine_extra, num_turns):
        metrics = dict(
            num_turns=num_turns,
            num_records=len(records),
        )

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
        metrics |= {f"linear_healer_{k}": float(v) for k, v in (healer_stats or {}).items()}

        # The engine's extra_fields carry the weight versions of the turns it served; a session
        # whose turns never reached this engine (all served before a restart) falls back to the
        # dispatch step rather than inventing staleness.
        dispatch_step = int(self.meta.get("step", 0))
        min_global_steps = _int_or(engine_extra.get("min_global_steps"), dispatch_step)
        max_global_steps = _int_or(engine_extra.get("max_global_steps"), dispatch_step)

        metrics |= dict(
            min_global_steps=min_global_steps,
            max_global_steps=max_global_steps,
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

        metrics |= dict(
            llm_generated_length=float(sum(sum(o.response_mask) for o in outputs)),
            context_length=float(sum(len(o.prompt_ids) + len(o.response_ids) for o in outputs)),
        )

        return outputs, metrics

    def _reward_extra_info(
        self,
        rollout: RolloutDumpResponse | None,
        num_records: int,
        reward_score: float,
    ) -> dict[str, float]:
        """The agent's declared scalar metrics, plus this rollout's score.

        Only *declared* keys are carried (``reward_extra_info_defaults``):
        verl reduces this dict across the batch, so a key that only some
        rollouts report would be averaged over the wrong denominator.
        Every row therefore has the same key set.
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
        prompt_ids, response_ids, response_mask, response_logprobs = self._build_prompt_response(record)

        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=response_logprobs,
            multi_modal_data={},
            # verl counts the initial user turn, which the gateway does not see.
            num_turns=num_turns + 1,
            metrics=AgentLoopMetrics(
                # The gateway's own accounting when the session keeps it, else the wall clock of
                # the bounded agent run.
                generate_sequences=float(self.meta.get("llm_latency_sum") or self.meta.get("agent_run", 0.0)),
                tool_calls=float(self.meta.get("tool_calls_time_s", 0.0)),
                compute_score=float(self.meta.get("eval_latency_s", 0.0)),
            ),
            reward_score=reward_score,
            extra_fields=dict(
                max_global_steps=staleness[1],
                min_global_steps=staleness[0],
                num_trace_records=num_records,
                request_id=self.session_id,
                reward_extra_info=dict(reward_extra_info),
                trace_index=index,
                trace_metadata=dict(record.metadata),
            ),
        )

    def _build_prompt_response(self, record):
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

        return prompt_ids, response_ids, response_mask, response_logprobs

    def _pad_token_id(self) -> int:
        """A token id that carries no training signal, for masked filler positions."""
        eos = getattr(self.tokenizer, "eos_token_id", None)
        return eos if isinstance(eos, int) else 0

    def _sampling_defaults(self, sampling_params: dict[str, Any]) -> dict[str, Any]:
        """This session's sampling defaults: the configured ones, then verl's."""
        defaults: dict[str, Any] = dict(self.loop_config.rollout_gateway_sampling_params)
        for key in ("temperature", "top_p", "top_k"):
            if key in sampling_params:
                defaults[key] = sampling_params[key]
        return defaults


__all__ = ["RolloutSessionAgentLoop"]
