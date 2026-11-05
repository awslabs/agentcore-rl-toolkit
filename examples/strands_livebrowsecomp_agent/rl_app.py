"""LiveBrowseComp: identify the one entity that a set of obscure clues describes (BrowseComp-style questions).

The paper's protocol (arXiv 2605.28721, Appendix C): its default system prompt, answer in <answer></answer>; tools
`search` and `visit` on the live web; 250 steps, 8K output tokens per turn, 256K context. When the steps or the
context run out, the paper's force-answer prompt gets one more turn. The reward is the official judge's verdict
(reward.py). A failing search gateway or summarizer fails the rollout instead of reaching the model.
"""

import logging
import re
from pathlib import Path

from botocore.exceptions import ClientError
from models import InvocationRequest
from reward import LiveBrowseCompReward, judge
from strands import Agent
from strands.agent.conversation_manager import NullConversationManager
from strands.hooks import BeforeModelCallEvent, HookProvider
from strands.models import BedrockModel
from strands.models.openai import OpenAIModel
from strands.types.exceptions import EventLoopException, MaxTokensReachedException
from tools import make_tools

from agentcore_rl_toolkit import AgentCoreRuntimeApp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = AgentCoreRuntimeApp()
reward_fn = LiveBrowseCompReward()

PROMPTS = Path(__file__).parent / "prompts"
SYSTEM_PROMPT = (PROMPTS / "system_default.md").read_text().strip()
FORCED_ANSWER = (PROMPTS / "force_answer.md").read_text().strip()
MAX_TURNS = 250
MAX_TOKENS = 8192
MAX_CONTEXT = 262144  # prompt + maximum output, the vLLM rule
FORCE_MARGIN = 512  # output tokens a forced reply gives up, so its longer prompt still fits
# what Strands writes in place of a tool call cut off at the output limit
TRUNCATED_TOOL_USE = re.compile(
    r"The selected tool .+'s tool use was incomplete due to maximum token limits being reached\."
)


class _Guard(HookProvider):
    """Stops the episode before a model call that would not fit the context, or after an infrastructure error.
    Records the history length of the last call that fit, for the forced reply."""

    def __init__(self, state: dict):
        self.state = state

    def register_hooks(self, registry, **_):
        registry.add_callback(BeforeModelCallEvent, self._before)

    def _before(self, event):
        if "infra_error" in self.state:
            event.cancel = True
        elif (event.projected_input_tokens or 0) + self.state.get("max_tokens", MAX_TOKENS) > MAX_CONTEXT:
            self.state["context_exceeded"] = True
            event.cancel = True
        else:
            self.state["fit_len"] = len(event.agent.messages)


def make_model(config: dict):
    """An OpenAI-compatible endpoint (training: the rollout gateway; vLLM) or, without base_url, a Bedrock model."""
    sampling = config.get("sampling_params", {})
    if config.get("base_url"):
        return OpenAIModel(
            client_args={"api_key": config.get("api_key") or "EMPTY", "base_url": config["base_url"]},
            model_id=config["model_id"],
            params={"max_tokens": MAX_TOKENS, **sampling},
        )
    return BedrockModel(
        model_id=config["model_id"],
        region_name=config.get("region", "us-west-2"),
        max_tokens=MAX_TOKENS,
        additional_request_fields=config.get("bedrock_fields"),
        **sampling,
    )


def set_max_tokens(agent: Agent, state: dict, n: int):
    state["max_tokens"] = n
    if isinstance(agent.model, OpenAIModel):
        agent.model.update_config(params={**agent.model.get_config().get("params", {}), "max_tokens": n})
    else:
        agent.model.update_config(max_tokens=n)


def call(agent: Agent, prompt: str | None, turns: int) -> str:
    """agent(prompt) for up to `turns` turns -> its stop reason. A reply cut off at the output limit stays in the
    history as written, without its truncated tool calls; a reply Bedrock blocks mid-stream is a refusal."""
    try:
        return agent(prompt, limits={"turns": turns}).stop_reason
    except (ClientError, EventLoopException) as e:  # the Bedrock error, as such or wrapped by the event loop
        if "blocked by content filtering policy" not in str(e):
            raise
        return "content_filtered"
    except MaxTokensReachedException:
        last = agent.messages[-1]
        last["content"] = [b for b in last["content"] if not TRUNCATED_TOOL_USE.fullmatch(b.get("text", ""))]
        if not last["content"]:
            agent.messages.pop()
        return "max_tokens"


def run(agent: Agent, state: dict, prompt: str, turns: int) -> str:
    """The agent on `prompt` for up to `turns` turns -> completed | max_turns | max_tokens | context_exceeded |
    refusal. A reply cut off at the output limit counts as the reply (completed); a turn cut off before any text
    is max_tokens. After context_exceeded, the history is back at the last call that fit."""
    reason = call(agent, prompt, turns)
    if state.pop("context_exceeded", False):
        del agent.messages[state["fit_len"] :]
        return "context_exceeded"
    if reason == "max_tokens":
        return "completed" if agent.messages[-1]["role"] == "assistant" else "max_tokens"
    # content_filtered: the model refused (e.g. Claude on some epidemic-modeling requests): no answer
    return {"limit_turns": "max_turns", "content_filtered": "refusal"}.get(reason, "completed")


def force_reply(agent: Agent, state: dict, prompt: str):
    """`prompt`, then one more turn with the tools attached, on the full history or, if that does not fit, on the
    last history that fit. The prompt joins trailing tool results, so roles keep alternating."""
    set_max_tokens(agent, state, MAX_TOKENS - FORCE_MARGIN)
    for _ in range(2):
        n = len(agent.messages)
        last = agent.messages[-1]
        if last["role"] == "user":
            last["content"].append({"text": prompt})
            call(agent, None, 1)
        else:
            call(agent, prompt, 1)
        if not state.pop("context_exceeded", False):
            break
        del agent.messages[state["fit_len"] :]
        if state["fit_len"] >= n:  # no older history to fall back to
            break
    set_max_tokens(agent, state, MAX_TOKENS)


def text_of(message: dict) -> str:
    return "".join(b["text"] for b in message.get("content") or [] if "text" in b).strip()


def single_turn(agent: Agent, state: dict, request: InvocationRequest) -> tuple[str, str, bool]:
    stop = run(agent, state, request.question, MAX_TURNS)
    forced = stop in ("max_turns", "max_tokens", "context_exceeded") and "infra_error" not in state
    if forced:
        force_reply(agent, state, FORCED_ANSWER)
    if stop == "refusal":  # no answer
        return "", stop, forced
    final = next((m for m in reversed(agent.messages) if m["role"] == "assistant"), {})
    return text_of(final), stop, forced


@app.entrypoint
def invoke_agent(payload: dict):
    request = InvocationRequest(**payload)
    state = {}
    agent = Agent(
        model=make_model(request.config),
        tools=make_tools(state),
        system_prompt=SYSTEM_PROMPT,
        hooks=[_Guard(state)],
        conversation_manager=NullConversationManager(),
        callback_handler=None,
    )
    response_text, stop, forced = single_turn(agent, state, request)
    if "infra_error" in state:
        raise RuntimeError(f"infrastructure error: {state['infra_error']}")

    correct, judge_reply = judge(request.question, response_text, request.gold)
    reward = reward_fn(correct=correct)
    logger.info("stop=%s forced=%s reward=%s", stop, forced, reward)
    return {
        "reward": reward,
        "stop": stop,
        "forced": forced,
        "judge": judge_reply,
        "response": response_text,
        "turns": sum(m["role"] == "assistant" for m in agent.messages),
    }


if __name__ == "__main__":
    app.run()
