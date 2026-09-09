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
    """
    Verl requires every loop output to have the same extra key set in a given run.
    request_id is required for rollout data dumping.
    min_global_steps/max_global_steps is required for async training.
    ``metrics`` carries the generic per-rollout scalar metrics (the harness's
    RolloutDumpResponse.metrics plus loop-computed entries); the trainer mixin
    reduces every key/value pair in it to agent_loop/<name>/{mean,min,max}.
    """

    max_global_steps: int
    metrics: dict[str, float]
    min_global_steps: int
    request_id: str
    # verl's _agent_loop_postprocess copies reward_extra_info from the final output
    # to earlier ones for multi-output trajectories; its async-reward path only
    # populates this when reward_score is None, but we set reward_score directly,
    # so we must provide the key ourselves to avoid a KeyError.
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
    """
    The detailed output for uploading to S3 (includes large, variable-size objects).
    """

    task: dict[str, Any]
    agent_loop_outputs: list[AgentLoopOutput]
    exception: str | None
    meta: AgentSessionMeta
    rollout_dump_response: RolloutDumpResponse | None


@register("rollout_session_agent_loop")
class RolloutSessionAgentLoop(AgentLoopBase):
    """An agent based on an HTTP server in a container.

    The verl-facing contracts (config, tokenization, semaphores, AgentLoopOutput
    assembly) live here. The deployment-specific container lifecycle -- Bedrock
    AgentCore vs. a local Docker container -- is delegated to a composed
    :class:`~.lifecycle.RolloutSession`, selected by the
    ``rollout_session_agent_loop.rollout_session_backend.backend`` config field.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # build the typed config from the rollout_session_agent_loop node (resolves
        # interpolations/env vars and validates the schema in config.py).
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

        # the gateway settings are the gateway's own: this node is its whole
        # keyword set, so a new one is added in yaml and read there, not here.
        self._gateway: GatewayHandle = get_or_start_gateway(
            server_manager=kwargs["server_manager"],
            tokenizer=self.tokenizer,
            **self.loop_config.rollout_gateway,
        )

        # The bounds on a rollout.
        self.bounds = get_rollout_session_bounds(self.loop_config.rollout_session_bounds)

        # the container lifecycle is delegated to a rollout session; it shares
        # this loop's session_id and meta dict and persists meta through the
        # same session store.
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
                        # The rollout ran and reported failure, which is a fact about
                        # the container, not an error on this side: whatever stack
                        # trace exists is already in the dump, so record it and abort
                        # the sample rather than raise a traceback of our own that
                        # only ever points back at these lines.
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
        # The TransferQueue runner (main_ppo_sync.AgentLoopWorkerTQ) forwards every
        # batch field into the loop kwargs, including tensor fields (input_ids,
        # attention_mask, position_ids, ...). The container only needs the dataset
        # task fields, and torch.Tensor/np.ndarray values cannot be JSON-serialized
        # when the task is sent to the agent server, so drop them here.
        task = {k: v for k, v in kwargs.items() if not isinstance(v, (torch.Tensor, np.ndarray))}
        task.update(self.loop_config.task_kwargs)
        task["sampling_params"] = sampling_params
        inference_url = f"{self._gateway.base_url}/v1"
        task["llm"] = build_llm(self.model, task["sampling_params"], inference_url, self.session_id)
        return task

    def _group_key(self) -> str:
        """Identify the rollout group this session belongs to.

        A group is one prompt's set of rollouts at a given training step, so all
        members share (step, task_id). ``step`` is absent when the trainer did
        not broadcast global_steps (-1), which keeps every group distinct by
        task_id in that case.
        """
        return f"{self.meta.get('step', -1)}:{self.meta['task_id']}"

    async def run_or_throw(self, task: dict) -> tuple[RolloutDumpResponse, list[AgentLoopOutput] | None]:
        """Run one rollout, raising only on failures of *this* side.

        Returns ``(dump, None)`` when the container itself reported failure -- that
        is a rollout outcome to be recorded, not an exception to be thrown, so the
        caller aborts the sample and keeps the dump.
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
                # Its reward is unusable, so there is no trajectory to train on --
                # but its metrics still describe what the container did before it
                # failed, which is how a bad rollout gets diagnosed. Returning
                # (rather than raising) still runs the finally below, so the gateway
                # session is drained either way.
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

        # verl v1 trainer does not yet handle multi-record scenario well
        best_record = sorted(records, key=lambda r: sum(r.token_ids))[-1]

        # In gateway linear-history mode the LinearHealer stamps its per-session counters
        # onto every record's metadata under "linear_healer" (all records of a session
        # carry the same dict). Forward whatever counters it emitted.
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

        # response_length is the intended full-trajectory token budget (prompt +
        # all turns), which has to be enforced before we return tokens to the trainer.
        max_response_length = max(0, self.response_length - len(prompt_ids))

        response_ids = list(record.token_ids[prompt_end:])[:max_response_length]
        response_mask = list(record.loss_mask)[:max_response_length]
        response_logprobs = list(record.logprobs)[:max_response_length]

        # A zero-length response can never carry a policy gradient and would crash
        # verl's AgentLoopOutput.as_dict (rm_scores[-1] = reward on an empty tensor).
        # This happens when the prompt alone fills the whole trajectory budget
        # (len(prompt_ids) >= self.response_length -> max_response_length == 0).
        # Emit a single masked token so the sample stays a valid, no-loss group member.
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

        # verl's fixed AgentLoopMetrics schema needs these three scalars; pull them
        # by name from the harness metrics (absent -> 0.0).
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
        # A failed rollout produced no engine-reported weight version, so tag it
        # with the dispatch step (kwargs["global_steps"], stored on meta as "step").
        # A valid trajectory's min/max_global_steps is the weight version it was
        # generated on -- essentially the dispatch step in async off-policy -- so
        # reusing that keeps the failure's trajectory_staleness in line with the
        # valid data instead of a bogus (global_steps - 0) inflation. This marker
        # feeds only the staleness metric; the replay buffer's off-policy dropping
        # uses tag["global_steps"] (kwargs), which is set for every output.
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
