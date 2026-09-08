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
        """Initialize the capturing provider.

        Args:
            client_args: Arguments for the LiteLLM client (e.g. ``base_url``, ``api_key``).
            **model_config: LiteLLM model config (``model_id``, ``params``, ``stream``).
        """
        super().__init__(client_args=client_args, **model_config)
        # Wall-clock seconds spent inside the LLM completion call, summed across
        # every assistant turn. This is the Strands analogue of OpenHands's
        # per-response `response_latencies`; the container agent loop reads it as
        # `llm_latency_sum` without knowing which backend produced it.
        self.llm_latency_sum: float = 0.0

    async def _handle_non_streaming_response(
        self, litellm_request: dict[str, Any]
    ) -> AsyncGenerator[StreamEvent, None]:
        """Capture token ids and logprobs, then re-emit the turn's Strands stream events.

        The parent ``stream()`` routes here because the model is configured
        non-streaming; it also translates a context-window overflow raised by the
        completion call into :class:`ContextWindowOverflowException`.

        Yields:
            Formatted Strands stream events for this turn.
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
    """Hook provider that accumulates wall-clock time spent executing tools.

    Records a start timestamp on :class:`BeforeToolCallEvent` and adds the elapsed
    time on the matching :class:`AfterToolCallEvent`, keyed by tool-use id, so the
    total excludes model-generation time. Also counts completed tool calls.
    """

    def __init__(self) -> None:
        """Initialize empty timing state."""
        self.total_time_s: float = 0.0
        self.num_tool_calls: int = 0
        self._starts: dict[str, float] = {}

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        """Subscribe to the before/after tool-call events."""
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
    """Render a Strands ``ToolResult`` into a printable string.

    A ``ToolResult`` carries a ``content`` list whose entries may hold ``text``,
    ``json``, or (rarely for these tools) ``image``/``document`` payloads. We join
    the human-readable parts; non-text payloads are summarized by their key rather
    than dumped verbatim.
    """
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

    The streaming callback handler only sees the ``contentBlockStart`` event, which
    carries the tool name and id but not its input (the arguments stream in as
    later deltas) nor its result. These hooks fire after the input is fully
    assembled and after the tool returns, so they have the complete picture.
    """

    def __init__(self) -> None:
        """Initialize the tool-call counter."""
        self.tool_count = 0

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        """Subscribe to the before/after tool-call events."""
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


# The SWE task container ships the graded project's Python in a conda env named
# ``testbed`` under ``/opt/miniconda3`` (see ``swe_agent/image/swe_unpack.sh``).
# swe_unpack.sh activates it via ``~/.bashrc``, but the vended bash tool runs each
# command as ``sh -c`` -- a non-interactive, non-login shell that never sources
# ``~/.bashrc`` -- so we activate the env explicitly per command instead. The
# guard keeps this a no-op outside the container (e.g. in unit tests), where the
# conda profile script is absent.
_CONDA_PROFILE = "/opt/miniconda3/etc/profile.d/conda.sh"
_CONDA_ENV = "testbed"
_CONDA_ACTIVATE_PREFIX = f'if [ -f "{_CONDA_PROFILE}" ]; then . "{_CONDA_PROFILE}" && conda activate {_CONDA_ENV}; fi; '


class _RepoLocalEnvironment(NotASandboxLocalEnvironment):
    """Host execution environment whose bash commands default to the repo directory.

    The server runs inside the SWE task container, so commands and file operations
    execute on the host (no isolation) -- matching how OpenHands runs its tools
    locally. The vended ``file_editor`` requires absolute paths, but ``bash`` runs
    with no explicit ``cwd`` and would otherwise land in the server's working
    directory; defaulting it to the repo path mirrors the OpenHands backend's
    ``Workspace(working_dir=repo_path)`` so bare shell commands behave the same.

    Each command is prefixed to activate the container's ``testbed`` conda env so
    the graded project's Python is on ``PATH``; see :data:`_CONDA_ACTIVATE_PREFIX`.
    """

    def __init__(self, working_dir: str) -> None:
        """Initialize with the repository working directory."""
        self.working_dir = working_dir

    async def execute_streaming(self, command: str, *, cwd: str | None = None, **kwargs: Any) -> Any:
        """Execute a command in the repo directory with the testbed conda env active."""
        async for chunk in super().execute_streaming(
            _CONDA_ACTIVATE_PREFIX + command,
            cwd=cwd if cwd is not None else self.working_dir,
            **kwargs,
        ):
            yield chunk


def rollout(request: RolloutStartRequest) -> RolloutDumpResponse:
    """Run a SWE rollout with the Strands agent and return a full RL trajectory dump.

    Drives the agent to edit the repository, captures per-turn tokens/logprobs,
    computes the git diff and grades the result -- the same contract every agent
    rollout here answers. Never raises: failures come back in ``exception``.
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
            # Don't propagate: we want the model to learn about the context limit.
            # The gateway caps per-turn generation and, when the prompt exceeds the
            # context budget, returns finish_reason="length" -- both surface through
            # Strands as MaxTokensReachedException, not ContextWindowOverflowException.
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
                # A generic scalar the trainer can reduce (0.0/1.0). The container loop
                # only reads top-level RolloutDumpResponse fields, not task_output, so a
                # signal it should track has to ride on metrics rather than task_output.
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

    The ``llm`` dict follows the same shape callers already send for the OpenHands
    backend: ``model`` (the LiteLLM model id), optional ``litellm_extra_body`` (vLLM
    extras such as ``return_token_ids`` and ``logprobs``), and any remaining keys
    (``base_url``, ``api_key``, ...) which are passed through as LiteLLM client args.
    The model is forced non-streaming so every turn's response carries token ids.
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
