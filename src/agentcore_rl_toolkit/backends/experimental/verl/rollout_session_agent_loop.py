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
from agentcore_rl_toolkit.rollout_session.exception_utils import exception_to_string
from agentcore_rl_toolkit.rollout_session.factory import make_session
from agentcore_rl_toolkit.rollout_session.lifecycle import (
    RolloutSession,
    run_rollout_with_bounds,
)
from agentcore_rl_toolkit.rollout_session.wire import RolloutDumpResponse

from .config import RolloutSessionAgentLoopConfig
from .rollout_session_resources import get_rollout_session_bounds

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

logging.getLogger("backoff").setLevel(logging.ERROR)


def build_llm(model: str, sampling_params: dict[str, Any], inference_url: str, session_id: str):
    llm = dict(
        model=f'openai/{model.split("/")[-1]}',
        base_url=inference_url,
        api_key=session_id,
        temperature=sampling_params["temperature"],
        top_p=sampling_params["top_p"],
        num_retries=0,
        timeout=120,
    )
    return llm


class ExtraFields(TypedDict):
    """Extra fields on every loop output; verl requires the same key set for a whole run.

    ``metrics`` entries are each reduced by the trainer to agent_loop/<name>/{mean,min,max}.
    """

    max_global_steps: int
    metrics: dict[str, float]
    min_global_steps: int
    request_id: str
    # verl only populates this itself when reward_score is None; we set reward_score
    # directly, so supply the key to avoid a KeyError in _agent_loop_postprocess
    reward_extra_info: dict


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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.loop_config: RolloutSessionAgentLoopConfig = instantiate(
            self.config.rollout_session_agent_loop, _convert_="all"
        )

        # extract verl config
        self.prompt_length = self.config.data.max_prompt_length
        self.response_length = self.config.data.max_response_length
        self.model = self.config.actor_rollout_ref.model.path
        self.experiment_name = self.config.trainer.experiment_name
        self.experiment_start_at = self.config.trainer.experiment_start_at

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

        self._gateway: GatewayHandle = get_or_start_gateway(
            server_manager=kwargs["server_manager"],
            tokenizer=self.tokenizer,
            **self.loop_config.rollout_gateway,
        )

        self.bounds = get_rollout_session_bounds(self.loop_config.rollout_session_bounds)

        # the session shares this loop's session_id and meta dict
        self.rollout_session: RolloutSession = self._make_session()

    def _make_session(self) -> RolloutSession:
        return make_session(
            self.session_id,
            self.loop_config.rollout_session_backend,
            self.meta,
        )

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> list[AgentLoopOutput]:
        task = self.make_task(sampling_params, kwargs)
        task_id = str(task["index"])

        async with self.meta.connection():
            await self.meta.update(
                experiment_name=self.experiment_name,
                experiment_start_at=self.experiment_start_at,
                experiment_start_at_session_id=f"{self.experiment_start_at}:{self.session_id}",
                aborted=False,
                task_id=task_id,
                step=int(kwargs.get("global_steps", -1)),
            )

            async with measure_span_persistent("verl_loop", self.meta):
                rollout_dump_response, exception, agent_loop_outputs = None, None, None
                try:
                    rollout_dump_response, agent_loop_outputs = await self.run_or_throw(task)
                    if agent_loop_outputs is None:
                        # container-reported failure: its stack trace is already in the
                        # dump, so record and abort instead of raising our own
                        exception = rollout_dump_response.failure_reason()
                        logger.error(f"Failed rollout {self.session_id} in container: {exception}")
                        await self.meta.set("aborted", True)
                except Exception as e:
                    logger.error(f"Failed rollout {self.session_id} in trainer: {e}", exc_info=e)
                    exception = exception_to_string(e)
                    await self.meta.set("aborted", True)

                if self.meta["aborted"]:
                    agent_loop_outputs = [self.make_failed_loop_output()]
                await self.save_to_s3(task, agent_loop_outputs, exception, rollout_dump_response)

            assert agent_loop_outputs is not None
            return agent_loop_outputs

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
        task["llm"] = build_llm(self.model, task["sampling_params"], inference_url, self.session_id)
        return task

    def _group_key(self) -> str:
        """Identify the rollout group (one prompt's rollouts at one step) this session belongs to."""
        return f"{self.meta.get('step', -1)}:{self.meta['task_id']}"

    async def run_or_throw(self, task: dict) -> tuple[RolloutDumpResponse, list[AgentLoopOutput] | None]:
        """Run one rollout, raising only on failures of *this* side.

        Returns ``(dump, None)`` when the container itself reported failure.
        """

        try:
            self._gateway.gateway.create_session(
                self.session_id,
                sampling_defaults=self._sampling_defaults(task["sampling_params"]),
                max_context_tokens=self.response_length,
            )
            rollout = await run_rollout_with_bounds(
                self.meta,
                self.bounds,
                self._group_key(),
                self.rollout_session,
                task,
            )
            if not rollout.is_successful():
                # no trainable trajectory, but keep the metrics for diagnosis; returning
                # rather than raising still runs the finally below, draining the session
                await self.meta.update(rollout.metrics)
                return rollout, None

            num_turns = self._gateway.gateway.manager.turn_count(self.session_id)
            engine_extra = self._gateway.backend.pop_extra_fields(self.session_id)
        finally:
            records = await self._gateway.gateway.finish_session(
                self.session_id,
                base_sample=BaseTrace(rollout_id=self.session_id),
                reward=0.0,
            )

        # maybe some responses are empty?
        records = [r for r in records if r.token_ids]

        # verl v1 trainer does not yet handle multi-record scenario well, so keep the
        # record with the most trainable tokens
        best_record = max(records, key=lambda r: sum(r.loss_mask))

        # in gateway linear-history mode every record carries the same LinearHealer counters
        healer_stats = best_record.metadata.get("linear_healer")
        linear_metrics = {f"linear_healer_{k}": float(v) for k, v in (healer_stats or {}).items()}

        await self.meta.update(
            rollout.metrics,
            num_turns=num_turns,
            # track the true num of records to catch issues
            num_records=len(records),
            min_global_steps=engine_extra["min_global_steps"],
            max_global_steps=engine_extra["max_global_steps"],
            reward_score=rollout.reward,
            **linear_metrics,
        )

        outputs = [self.make_successful_loop_output(best_record)]

        session_generated_length = float(
            sum(o.extra_fields["metrics"].get("llm_generated_length", 0.0) for o in outputs)
        )
        await self.meta.set("llm_generated_length", session_generated_length)
        session_context_length = float(sum(o.extra_fields["metrics"].get("context_length", 0.0) for o in outputs))
        await self.meta.set("context_length", session_context_length)

        for o in outputs:
            o.extra_fields["metrics"].update(self.meta)

        return rollout, outputs

    def make_successful_loop_output(self, record: TraceRecord) -> AgentLoopOutput:
        response_len = len(record.loss_mask)
        prompt_end = len(record.token_ids) - response_len
        prompt_ids = list(record.token_ids[:prompt_end])
        assert len(prompt_ids) <= self.prompt_length, f"{len(prompt_ids)} > {self.prompt_length}"

        # response_length is the full-trajectory budget (prompt + all turns)
        max_response_length = max(0, self.response_length - len(prompt_ids))

        response_ids = list(record.token_ids[prompt_end:])[:max_response_length]
        response_mask = list(record.loss_mask)[:max_response_length]
        response_logprobs = list(record.logprobs)[:max_response_length]

        # the prompt alone can fill the whole budget; an empty response crashes verl's
        # AgentLoopOutput.as_dict, so emit one masked token to keep a valid no-loss sample
        if not response_ids:
            response_ids = [0]
            response_mask = [1]
            response_logprobs = [0.0]

        llm_generated_length = sum(response_mask[:max_response_length])
        context_length = len(prompt_ids) + len(response_ids)
        metrics = dict(
            llm_generated_length=float(llm_generated_length),
            context_length=float(context_length),
        )

        # verl's fixed AgentLoopMetrics schema needs these three scalars
        llm_latency_sum = self.meta.get("llm_latency_sum", 0.0)
        tool_call_latency = self.meta.get("tool_calls_time_s", 0.0)
        eval_latency_s = self.meta.get("eval_latency_s", 0.0)

        extra_fields = ExtraFields(
            request_id=self.session_id,
            min_global_steps=self.meta["min_global_steps"],
            max_global_steps=self.meta["max_global_steps"],
            reward_extra_info={},
            metrics=metrics,
        )

        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=response_logprobs,
            multi_modal_data={},
            num_turns=self.meta["num_turns"],
            metrics=AgentLoopMetrics(
                generate_sequences=llm_latency_sum,
                tool_calls=tool_call_latency,
                compute_score=eval_latency_s,
            ),
            reward_score=self.meta["reward_score"],
            extra_fields=dict(extra_fields),
        )

    def make_failed_loop_output(self) -> AgentLoopOutput:
        # no engine-reported weight version exists, so stand in the dispatch step: it keeps
        # trajectory_staleness comparable to valid data instead of inflating it to global_steps
        dispatch_step = self.meta["step"]
        extra_fields = ExtraFields(
            max_global_steps=dispatch_step,
            metrics=dict(self.meta),
            min_global_steps=dispatch_step,
            request_id=self.session_id,
            reward_extra_info={},
        )

        return AgentLoopOutput(
            prompt_ids=[0],
            response_ids=[0],
            response_mask=[1],
            response_logprobs=[0.0],
            multi_modal_data={},
            num_turns=0,
            metrics=AgentLoopMetrics(
                generate_sequences=0,
                tool_calls=0.0,
                compute_score=0.0,
            ),
            reward_score=0,  # this will affect the advantage of successful trajectories
            extra_fields=extra_fields,  # type: ignore
        )

    def _sampling_defaults(self, sampling_params: dict[str, Any]) -> dict[str, Any]:
        """This session's sampling defaults: the configured ones, then verl's."""
        defaults: dict[str, Any] = dict(self.loop_config.rollout_gateway_sampling_params)
        for key in ("temperature", "top_p", "top_k"):
            if key in sampling_params:
                defaults[key] = sampling_params[key]
        return defaults
