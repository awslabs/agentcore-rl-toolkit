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

# How many *consecutive* no-content responses (empty or reasoning-only) to tolerate
# before ending the conversation. Each such response only earns a corrective nudge
# from the stock harness and never terminates, so a model stuck emitting empty
# completions loops until OOM (see NoContentTerminatingAgent). Three lets a single
# stray empty turn recover via the nudge while cutting the runaway loop short.
MAX_CONSECUTIVE_NO_CONTENT = 3


class NoContentTerminatingAgent(Agent):
    """An :class:`Agent` that ends the run when the model repeatedly returns a
    response with neither a tool call nor user-facing content.

    The stock OpenHands dispatch routes such responses (classified ``EMPTY`` or
    ``REASONING_ONLY``) to ``_handle_no_content_response``, which only emits a
    corrective nudge ("...did not include a function call...Please use a tool...")
    and continues -- it never sets ``FINISHED`` (only the ``CONTENT`` path does).
    A model that keeps returning empty completions therefore loops forever,
    re-sending the whole growing history each turn until the container is
    OOM-killed and the trainer observes the death as a 502.

    This subclass counts *consecutive* no-content responses and, once they reach
    :data:`MAX_CONSECUTIVE_NO_CONTENT`, marks the conversation ``FINISHED`` -- a
    clean terminal state, so evaluation still runs on whatever the agent produced
    -- and skips the nudge. Any tool call resets the counter, so an isolated empty
    response is still tolerated and nudged exactly as before.

    ``Agent`` is a pydantic ``DiscriminatedUnionMixin``: subclasses auto-register
    by class name, but must be defined at module level (local classes raise) with a
    unique name. Only a ``PrivateAttr`` counter is added, so the serialized field
    set is unchanged.
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
            # Mirror _handle_content_response: still record the (empty) turn and its
            # tokens so RL capture stays consistent, then finish cleanly.
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
                    # We don't propagate this exception, since we want LLM to learn about the context limit.
                    context_window_exceeded = True
                case _:
                    # Otherwise, we are in undefined state and should propagate.
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
        # A conversation that is never closed leaves two things behind. Its tool
        # executors hold a terminal subprocess each, and -- the reason this is here --
        # OpenHands ends its per-conversation root span only from close()
        # (LocalConversation.close -> _end_observability_span), and a span that never
        # ends is never exported. That is what left every step of the agent loop in
        # CloudWatch parented to a span id that appears nowhere: measured on session
        # verl_1812a120506c4f3eb1d1350c1fe2a438, `conversation.run` pointing at a
        # parent that was never exported.
        #
        # Last, and its own failure swallowed: by this point the rollout has a result,
        # and losing it to a cleanup error would be the worse outcome. Safe here
        # because close() does not touch conversation.state -- which the response
        # below still reads -- and the git diff and evaluation above are done.
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
                # A generic scalar the trainer can reduce (0.0/1.0). The container loop
                # only reads top-level RolloutDumpResponse fields, not task_output, so a
                # signal it should track has to ride on metrics rather than task_output.
                "context_window_exceeded": context_window_exceeded,
                # Token counts live in the conversation state, which the loop never reads;
                # forwarding them here is what gets them reduced and logged.
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
    """Wall-clock time spent executing tools, in seconds.

    Each ObservationEvent carries the id of the ActionEvent it responds to
    (`action_id`) and both events carry an ISO-format `timestamp`. We sum the
    delta between each action and its observation to approximate the time spent
    running tools (as opposed to LLM generation)."""
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

    OpenHands records one latency per model response under
    ``stats.usage_to_metrics["default"].response_latencies[].latency``. We sum the
    ``default`` bucket (the task-solving LLM) so the value is comparable to the
    Strands backend's ``llm_latency_sum`` (which times each completion directly) and
    excludes auxiliary LLMs (e.g. a condenser) that get their own usage bucket.
    Missing/partial stats yield ``0.0`` rather than raising -- a metric should never
    fail a rollout.
    """
    default_metrics = conversation_state.get("stats", {}).get("usage_to_metrics", {}).get("default", {})
    return sum(latency["latency"] for latency in default_metrics.get("response_latencies", []))


# Fields of the OpenHands ``TokenUsage`` that identify the call rather than count
# tokens, and so have no place in a numeric metrics dict.
_TOKEN_USAGE_LABEL_FIELDS = frozenset({"model", "response_id"})


def compute_token_metrics(conversation_state: dict) -> dict[str, float]:
    """Per-rollout token counts, lifted out of the OpenHands conversation state.

    OpenHands keeps its token accounting under ``stats.usage_to_metrics["default"]``,
    which only reaches us as a side effect of dumping the whole conversation state
    into ``task_output`` -- and the container loop reads only top-level
    RolloutDumpResponse fields, never ``task_output``. Copying the counts into
    ``metrics`` is therefore what makes the trainer reduce and log them (same
    reasoning as ``context_window_exceeded`` in :func:`rollout`).

    Every numeric field of the accumulated ``TokenUsage`` is forwarded under an
    ``openhands_`` prefix, so fields the SDK adds later flow through with no change
    here and nothing collides with the loop's own metric names. Two of those fields
    are quirky, and are passed through verbatim rather than silently fixed up:

    - ``per_turn_token`` is *not* a sum -- ``TokenUsage.__add__`` overwrites it, so
      the accumulated copy holds only the final call's ``prompt + completion``. That
      makes it the end-of-rollout context length, so no separate ``context_length``
      is derived here; note the loop also owns the bare name ``context_length`` for a
      value it computes from real token ids.
    - ``context_window`` stays 0 unless ``llm.max_input_tokens`` is configured.

    Note ``prompt_tokens`` includes cached reads (the litellm convention), so uncached
    input is ``prompt_tokens - cache_read_tokens``.
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

    # Render the instruction
    instruction = template.render(context)
    return instruction
