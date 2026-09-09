"""OpenHands backend: runs the agent loop over the task repo, then grades the diff."""

import logging
import subprocess
from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from openhands.sdk import LLM, Agent, Conversation, LocalConversation, Tool, Workspace
from openhands.sdk.conversation.exceptions import ConversationRunError
from openhands.sdk.conversation.state import ConversationExecutionStatus
from openhands.sdk.llm.exceptions.types import LLMContextWindowExceedError
from openhands.tools import FileEditorTool, TerminalTool
from pydantic import PrivateAttr
from swe_agent_server.evaluation import run_evaluation
from swe_agent_server.utils import clean_metrics, exc_to_full_string

from agentcore_rl_toolkit.rollout_session.wire import (
    RolloutDumpResponse,
    RolloutStartRequest,
)

# Consecutive no-content responses tolerated before finishing: enough for one stray
# empty turn to recover via the stock nudge, few enough to cut the runaway loop short.
MAX_CONSECUTIVE_NO_CONTENT = 3


class NoContentTerminatingAgent(Agent):
    """An :class:`Agent` that ends the run on repeated no-content model responses.

    The stock dispatch only nudges such responses (``EMPTY``/``REASONING_ONLY``) and
    never sets ``FINISHED``, so a model emitting empty completions loops until the
    container is OOM-killed. Finishing instead is a clean terminal state, so evaluation
    still runs on whatever the agent produced. Any tool call resets the counter.

    ``Agent`` is a pydantic ``DiscriminatedUnionMixin``, so subclasses must be defined at
    module level with a unique name; only a ``PrivateAttr`` is added, so the serialized
    field set is unchanged.
    """

    _consecutive_no_content: int = PrivateAttr(default=0)

    def _handle_no_content_response(
        self, message, llm_response, conversation, state, on_event, *, response_type
    ) -> None:
        self._consecutive_no_content += 1
        if self._consecutive_no_content >= MAX_CONSECUTIVE_NO_CONTENT:
            logging.warning(
                "LLM returned %d consecutive no-content responses - finishing conversation instead of nudging again",
                self._consecutive_no_content,
            )
            # As _handle_content_response does: record the (empty) turn and its tokens so
            # RL capture stays consistent.
            self._emit_message_event(message, llm_response, conversation, on_event)
            self._maybe_emit_vllm_tokens(llm_response, on_event)
            state.execution_status = ConversationExecutionStatus.FINISHED
            return
        super()._handle_no_content_response(
            message,
            llm_response,
            conversation,
            state,
            on_event,
            response_type=response_type,
        )

    def _handle_tool_calls(self, message, llm_response, conversation, state, on_event) -> None:
        self._consecutive_no_content = 0
        super()._handle_tool_calls(message, llm_response, conversation, state, on_event)

    async def _ahandle_tool_calls(self, message, llm_response, conversation, state, on_event) -> None:
        self._consecutive_no_content = 0
        await super()._ahandle_tool_calls(message, llm_response, conversation, state, on_event)


def rollout(request: RolloutStartRequest) -> RolloutDumpResponse:
    conversation = None
    exception = None
    git_diff = None
    num_tool_calls = None
    tool_calls_time_s = None
    llm_latency_sum = None
    eval_report = None
    context_window_exceeded = False
    conversation_state = None
    token_metrics = {}

    try:
        tools = [
            Tool(name=TerminalTool.name),
            Tool(name=FileEditorTool.name),
        ]
        agent = NoContentTerminatingAgent(llm=LLM(**request.task_input["llm"]), tools=tools)
        workspace = Workspace(working_dir=request.task_input["repo_path"])

        conversation = Conversation(workspace=workspace, agent=agent, callbacks=[on_conversation_event])
        message = get_instruction(request.task_input)
        conversation.send_message(message)

        try:
            conversation.run()
        except ConversationRunError as e:
            match e.original_exception:
                case LLMContextWindowExceedError():
                    # Not propagated: we want the LLM to learn about the context limit.
                    context_window_exceeded = True
                case _:
                    raise e

        tool_calls_time_s = compute_tool_calls_time_s(conversation)

        conversation_state = conversation.state.model_dump(mode="json")
        llm_latency_sum = compute_llm_latency_sum(conversation_state)
        token_metrics = compute_token_metrics(conversation_state)

        logging.info(f"Conversation: {conversation.state.model_dump()}")

        git_diff = subprocess.check_output(
            [
                "git",
                "--no-pager",
                "diff",
                "--no-color",
                request.task_input["base_commit"],
            ],
            cwd=request.task_input["repo_path"],
        ).decode()

        eval_report = run_evaluation(request.task_input)

    except Exception as e:
        logging.error("Exception during rollout", exc_info=e)
        exception = exc_to_full_string(e)

    finally:
        # An unclosed conversation leaks a terminal subprocess per tool executor, and
        # leaves its per-conversation root span unended -- and so never exported, which
        # orphans every step of the agent loop in CloudWatch. Its own failure is
        # swallowed: by now the rollout has a result worth more than the cleanup.
        if conversation is not None:
            try:
                conversation.close()
            except Exception as e:
                logging.warning("Exception closing conversation", exc_info=e)

    return RolloutDumpResponse(
        metrics=clean_metrics(
            {
                "num_tool_calls": num_tool_calls,
                "tool_calls_time_s": tool_calls_time_s,
                "llm_latency_sum": llm_latency_sum,
                "eval_latency_s": (eval_report.get("eval_latency_s") if eval_report is not None else None),
                # The container loop reads only top-level fields, never task_output, so
                # anything it should track has to ride on metrics.
                "context_window_exceeded": context_window_exceeded,
                **token_metrics,
            }
        ),
        task_output=dict(
            git_diff=git_diff,
            events=([e.model_dump(mode="json") for e in conversation.state.events] if conversation is not None else []),
            conversation_state=conversation_state,
            eval_report=eval_report,
        ),
        reward=float(eval_report["resolved"]) if eval_report is not None else None,
        exception=exception,
    )


def compute_tool_calls_time_s(c: LocalConversation) -> float:
    """Wall-clock time spent executing tools, summed over action/observation pairs."""
    action_ts: dict[str, datetime] = {}
    for e in c.state.events:
        if e.kind == "ActionEvent":
            action_ts[e.id] = datetime.fromisoformat(e.timestamp)

    total = 0.0
    for e in c.state.events:
        # ObservationEvent / UserRejectObservation link back via `action_id`.
        action_id = getattr(e, "action_id", None)
        if action_id is None:
            continue
        started = action_ts.get(action_id)
        if started is None:
            continue
        delta = (datetime.fromisoformat(e.timestamp) - started).total_seconds()
        if delta > 0:
            total += delta

    return total


def compute_llm_latency_sum(conversation_state: dict) -> float:
    """Total wall-clock seconds spent in the main LLM's completion calls this rollout.

    Only the ``default`` usage bucket (the task-solving LLM), so auxiliary LLMs such as a
    condenser are excluded. Missing stats yield ``0.0``: a metric must not fail a rollout.
    """
    default_metrics = conversation_state.get("stats", {}).get("usage_to_metrics", {}).get("default", {})
    return sum(latency["latency"] for latency in default_metrics.get("response_latencies", []))


# Fields of the OpenHands ``TokenUsage`` that label the call rather than count tokens.
_TOKEN_USAGE_LABEL_FIELDS = frozenset({"model", "response_id"})


def compute_token_metrics(conversation_state: dict) -> dict[str, float]:
    """Per-rollout token counts, lifted out of the OpenHands conversation state.

    Every numeric field of the accumulated ``TokenUsage`` is forwarded under an
    ``openhands_`` prefix, so later SDK additions flow through and nothing collides with
    the loop's own metric names. Two quirks are passed through verbatim rather than fixed
    up: ``per_turn_token`` is overwritten rather than summed by ``TokenUsage.__add__`` (so
    it is the end-of-rollout context length), and ``context_window`` stays 0 unless
    ``llm.max_input_tokens`` is set. ``prompt_tokens`` includes cached reads.
    """
    default_metrics = conversation_state.get("stats", {}).get("usage_to_metrics", {}).get("default", {})
    accumulated = default_metrics.get("accumulated_token_usage") or {}

    return {
        f"openhands_{field}": float(value)
        for field, value in accumulated.items()
        if field not in _TOKEN_USAGE_LABEL_FIELDS and isinstance(value, (int, float))
    }


def on_conversation_event(ev):  # keep it simple
    logging.info("Event: %s", ev)


def get_instruction(
    instance: dict,
) -> str:
    """Generate instruction for the agent."""
    env = Environment(loader=FileSystemLoader(Path(__file__).parent / "prompts"))
    template = env.get_template("benchmarks_r2.j2")

    context = {
        "instance": instance,
    }

    instruction = template.render(context)
    return instruction
