"""TurnCapHook: ends the search after SEARCH_TURNS turns or near the context limit, and asks for the answer."""

from strands.hooks import (
    AfterModelCallEvent,
    AfterToolCallEvent,
    BeforeModelCallEvent,
    BeforeToolCallEvent,
    HookProvider,
    HookRegistry,
)

from rollout.agent_prompts import FINAL_NUDGE, SEARCH_TURNS

CONTEXT_BUDGET_TOKENS = 44_000  # leaves room for one more worst-case turn and the answer in a 64k context
MAX_FINAL_ATTEMPTS = 2
FINAL_USER_TURN = (
    "[FINAL ANSWER PHASE] Searching is finished. Do not call any tools, write a patch, or continue searching."
)


def _approx_tokens(messages) -> int:
    """A deliberately high estimate: characters / 3.5, over text, tool inputs and tool results."""
    chars = 0
    for message in messages:
        for block in message.get("content") or []:
            if "text" in block:
                chars += len(str(block["text"]))
            elif "toolUse" in block:
                chars += len(str(block["toolUse"].get("input", "")))
            elif "toolResult" in block:
                chars += sum(len(str(item.get("text", ""))) for item in block["toolResult"].get("content") or [])
    return int(chars / 3.5)


def _context_tokens(agent) -> int:
    """The next prompt's size: what the server reported for the last call, plus an estimate for the
    tool results since."""
    reported = getattr(agent.event_loop_metrics, "projected_context_size", None)
    if reported is None:
        return _approx_tokens(agent.messages)
    last_assistant = max((i for i, m in enumerate(agent.messages) if m["role"] == "assistant"), default=-1)
    return int(reported) + _approx_tokens(agent.messages[last_assistant + 1 :])


class TurnCapHook(HookProvider):
    """Once the budget is reached, tool calls are cancelled, a final user turn asks for the answer, and
    the model gets at most two more calls to give it."""

    def __init__(self) -> None:
        self.final_phase = False
        self.budget_reached = False  # set after a tool batch, acted on at the next model call
        self.final_attempts = 0
        self.nudged = False

    def register_hooks(self, registry: HookRegistry, **kwargs) -> None:
        registry.add_callback(BeforeModelCallEvent, self._before_model)
        registry.add_callback(AfterModelCallEvent, self._after_model)
        registry.add_callback(BeforeToolCallEvent, self._before_tool)
        registry.add_callback(AfterToolCallEvent, self._after_tool)

    def _over_budget(self, agent) -> bool:
        turns = sum(m["role"] == "assistant" for m in agent.messages)
        return turns >= SEARCH_TURNS or _context_tokens(agent) >= CONTEXT_BUDGET_TOKENS

    def _before_model(self, event: BeforeModelCallEvent) -> None:
        if not self.final_phase and (self.budget_reached or self._over_budget(event.agent)):
            self.final_phase = True
            event.agent.messages.append({"role": "user", "content": [{"text": FINAL_USER_TURN + FINAL_NUDGE}]})
        if not self.final_phase:
            return
        if self.final_attempts >= MAX_FINAL_ATTEMPTS:
            event.cancel = "Final submission attempts exhausted."
        else:
            self.final_attempts += 1

    def _after_model(self, event: AfterModelCallEvent) -> None:
        if event.exception is not None and self.final_phase:
            self.final_attempts -= 1  # a failed call is retried, and the retry is not another attempt

    def _before_tool(self, event: BeforeToolCallEvent) -> None:
        if self.final_phase:
            event.cancel_tool = "Search budget exhausted. " + FINAL_NUDGE

    def _after_tool(self, event: AfterToolCallEvent) -> None:
        # Decided here and remembered: the reported context size can change once the tool batch ends.
        if self.final_phase or not self._over_budget(event.agent):
            return
        self.budget_reached = True
        if not self.nudged:
            event.result.setdefault("content", []).append({"text": FINAL_NUDGE})
            self.nudged = True
