"""Strands backend: runs the agent loop over the task repo, then grades the diff."""

import asyncio
import json
import logging
import subprocess
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader
from strands import Agent
from strands.agent.conversation_manager import NullConversationManager
from strands.experimental.tools import stop
from strands.handlers import PrintingCallbackHandler
from strands.hooks import (
    AfterToolCallEvent,
    BeforeToolCallEvent,
    HookProvider,
    HookRegistry,
)
from strands.models.litellm import LiteLLMModel
from strands.sandbox.not_a_sandbox_local_environment import NotASandboxLocalEnvironment
from strands.types.exceptions import (
    ContextWindowOverflowException,
    MaxTokensReachedException,
)
from strands.types.streaming import StreamEvent
from strands.vended_tools.bash import make_bash
from strands.vended_tools.file_editor import make_file_editor
from swe_agent_server.evaluation import run_evaluation
from swe_agent_server.utils import clean_metrics, exc_to_full_string

from agentcore_rl_toolkit.rollout_session.wire import (
    RolloutDumpResponse,
    RolloutStartRequest,
)

logger = logging.getLogger(__name__)


logging.getLogger("strands").setLevel(logging.INFO)


class _CapturingLiteLLMModel(LiteLLMModel):
    """LiteLLM provider that records llm call latency."""

    def __init__(self, client_args: dict[str, Any] | None = None, **model_config: Any) -> None:
        super().__init__(client_args=client_args, **model_config)
        # Summed across every assistant turn; the container loop reads it as
        # `llm_latency_sum` without knowing which backend produced it.
        self.llm_latency_sum: float = 0.0

    async def _handle_non_streaming_response(
        self, litellm_request: dict[str, Any]
    ) -> AsyncGenerator[StreamEvent, None]:
        """Time the completion call, then re-emit the turn's Strands stream events.

        The parent ``stream()`` routes here because the model is configured
        non-streaming, and translates a context-window overflow into
        :class:`ContextWindowOverflowException`.
        """
        started = _monotonic()
        response = await self._acompletion(litellm_request)
        self.llm_latency_sum += _monotonic() - started
        for chunk in self._format_non_streaming_response(response):
            yield chunk

    async def _acompletion(self, litellm_request: dict[str, Any]) -> Any:
        """Issue the non-streaming LiteLLM completion. Seam for tests to inject responses."""
        import litellm

        return await litellm.acompletion(**self.client_args, **litellm_request)


class _ToolTimingHooks(HookProvider):
    """Hook provider that counts tool calls and the wall-clock time they take.

    Timings are keyed by tool-use id, so the total excludes model-generation time.
    """

    def __init__(self) -> None:
        self.total_time_s: float = 0.0
        self.num_tool_calls: int = 0
        self._starts: dict[str, float] = {}

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        registry.add_callback(BeforeToolCallEvent, self._on_before)
        registry.add_callback(AfterToolCallEvent, self._on_after)

    def _on_before(self, event: BeforeToolCallEvent) -> None:
        self._starts[event.tool_use["toolUseId"]] = _monotonic()

    def _on_after(self, event: AfterToolCallEvent) -> None:
        self.num_tool_calls += 1
        started = self._starts.pop(event.tool_use["toolUseId"], None)
        if started is not None:
            self.total_time_s += _monotonic() - started


def _monotonic() -> float:
    """Monotonic clock read. Wrapped so tests can control tool-timing deterministically."""
    import time

    return time.monotonic()


def _format_tool_result(result: dict[str, Any]) -> str:
    """Render a Strands ``ToolResult`` into a printable string."""
    parts: list[str] = []
    for block in result.get("content", []):
        if "text" in block:
            parts.append(block["text"])
        elif "json" in block:
            parts.append(json.dumps(block["json"], indent=2, default=str))
        else:
            # image / document / other binary-ish content: name it, don't dump bytes.
            parts.append(f"<{'/'.join(block.keys())} content>")
    return "\n".join(parts)


class _ToolPrintingHooks(HookProvider):
    """Hook provider that prints each tool call's full parameters and result to stdout.

    The streaming callback handler only sees ``contentBlockStart``, which carries neither
    the assembled input nor the result; these hooks fire late enough to have both.
    """

    def __init__(self) -> None:
        self.tool_count = 0

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        registry.add_callback(BeforeToolCallEvent, self._on_before)
        registry.add_callback(AfterToolCallEvent, self._on_after)

    def _on_before(self, event: BeforeToolCallEvent) -> None:
        self.tool_count += 1
        tool_use = event.tool_use
        params = json.dumps(tool_use.get("input", {}), indent=2, default=str)
        print(f"\nTool #{self.tool_count}: {tool_use['name']} (id={tool_use['toolUseId']})")
        print(f"  params: {params}")

    def _on_after(self, event: AfterToolCallEvent) -> None:
        result = event.result
        print(f"  result [{result.get('status', '?')}]: {_format_tool_result(result)}")


# swe_unpack.sh activates the graded project's conda env via ``~/.bashrc``, but the
# vended bash tool runs each command as ``sh -c``, which never sources it -- so activate
# per command instead. The guard makes this a no-op outside the container.
_CONDA_PROFILE = "/opt/miniconda3/etc/profile.d/conda.sh"
_CONDA_ENV = "testbed"
_CONDA_ACTIVATE_PREFIX = f'if [ -f "{_CONDA_PROFILE}" ]; then . "{_CONDA_PROFILE}" && conda activate {_CONDA_ENV}; fi; '


class _RepoLocalEnvironment(NotASandboxLocalEnvironment):
    """Host execution environment whose bash commands default to the repo directory.

    The server already runs inside the SWE task container, so there is no isolation here.
    ``bash`` would otherwise run in the server's working directory; defaulting it to the
    repo mirrors the OpenHands backend's ``Workspace(working_dir=repo_path)``. Each
    command is also prefixed with :data:`_CONDA_ACTIVATE_PREFIX`.
    """

    def __init__(self, working_dir: str) -> None:
        self.working_dir = working_dir

    async def execute_streaming(self, command: str, *, cwd: str | None = None, **kwargs: Any) -> Any:
        async for chunk in super().execute_streaming(
            _CONDA_ACTIVATE_PREFIX + command,
            cwd=cwd if cwd is not None else self.working_dir,
            **kwargs,
        ):
            yield chunk


def rollout(request: RolloutStartRequest) -> RolloutDumpResponse:
    """Run a SWE rollout with the Strands agent and return a full RL trajectory dump.

    Never raises: failures come back in ``exception``.
    """
    agent = None
    exception = None
    git_diff = None
    num_tool_calls = None
    tool_calls_time_s = None
    llm_latency_sum = None
    eval_report = None
    context_window_exceeded = False

    try:
        repo_path = request.task_input["repo_path"]
        model = _build_model(request.task_input["llm"])
        sandbox = _RepoLocalEnvironment(working_dir=repo_path)
        timing = _ToolTimingHooks()

        agent = Agent(
            model=model,
            tools=[make_bash(), make_file_editor(), stop],
            conversation_manager=NullConversationManager(),
            sandbox=sandbox,
            hooks=[timing, _ToolPrintingHooks()],
            callback_handler=PrintingCallbackHandler(verbose_tool_use=False),
        )

        instruction = get_instruction(request.task_input)

        try:
            asyncio.run(agent.invoke_async(instruction))
        except (ContextWindowOverflowException, MaxTokensReachedException):
            # Not propagated: we want the model to learn about the context limit. The
            # gateway signals an over-budget prompt with finish_reason="length", which
            # Strands raises as MaxTokensReachedException, not the overflow exception.
            context_window_exceeded = True

        num_tool_calls = timing.num_tool_calls
        tool_calls_time_s = timing.total_time_s
        llm_latency_sum = model.llm_latency_sum

        git_diff = subprocess.check_output(
            [
                "git",
                "--no-pager",
                "diff",
                "--no-color",
                request.task_input["base_commit"],
            ],
            cwd=repo_path,
        ).decode()

        eval_report = run_evaluation(request.task_input)

    except Exception as error:
        logging.error("Exception during rollout", exc_info=error)
        exception = exc_to_full_string(error)

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
            }
        ),
        task_output=dict(
            git_diff=git_diff,
            messages=(agent.messages if agent is not None else []),
            eval_report=eval_report,
        ),
        reward=float(eval_report["resolved"]) if eval_report is not None else None,
        exception=exception,
    )


def _build_model(llm: dict) -> _CapturingLiteLLMModel:
    """Build the capturing LiteLLM model from the request's ``llm`` config.

    Same shape callers send for the OpenHands backend: ``model``, optional
    ``litellm_extra_body`` (vLLM extras), and remaining keys as LiteLLM client args.
    Forced non-streaming so every turn's response carries token ids.
    """
    llm = dict(llm)
    model_id = llm.pop("model")
    extra_body = llm.pop("litellm_extra_body", {})
    params: dict[str, Any] = {}
    if extra_body:
        params["extra_body"] = extra_body
    return _CapturingLiteLLMModel(client_args=llm, model_id=model_id, params=params, stream=False)


def get_instruction(instance: dict) -> str:
    """Render the agent instruction from the shared task template."""
    env = Environment(loader=FileSystemLoader(Path(__file__).parent / "prompts"))
    template = env.get_template("benchmarks_r2.j2")
    return template.render({"instance": instance})
