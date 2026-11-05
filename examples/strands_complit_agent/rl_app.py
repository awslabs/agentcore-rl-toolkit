"""CompLit (single-turn settings): find the one arXiv paper a researcher describes, or say that none qualifies.

Upstream's setup with web search off: no system prompt; upstream's user prompt; tools arxiv_search and read_paper
(served by env/serve.py); 15 turns (upstream's limit of 30 messages), 8K output tokens per turn, 128K context. When
the turns or the context run out, upstream's forced-answer prompt gets one more turn. The reward is upstream's
deterministic rule. An unavailable environment fails the rollout instead of reaching the model.
"""

import logging

from models import InvocationRequest
from prompts import FORCED_ANSWER, build_prompt
from reward import CompLitReward
from strands import Agent
from strands.agent.conversation_manager import NullConversationManager
from strands.hooks import BeforeModelCallEvent, HookProvider
from strands.models import BedrockModel
from strands.models.openai import OpenAIModel
from tools import make_tools

from agentcore_rl_toolkit import AgentCoreRuntimeApp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = AgentCoreRuntimeApp()
reward_fn = CompLitReward()

MAX_TURNS = 15
MAX_TOKENS = 8192
MAX_CONTEXT = 131072  # prompt + maximum output, the vLLM rule
FORCE_MARGIN = 512  # output tokens the forced answer gives up, so its longer prompt still fits


class _Guard(HookProvider):
    """Stops the episode before a model call that would not fit the context, or after an infrastructure error.
    Records the history length of the last call that fit, for the forced answer."""

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


def force_answer(agent: Agent, state: dict):
    """Upstream's forced-answer prompt, then one more turn with the tools still attached."""
    state["max_tokens"] = MAX_TOKENS - FORCE_MARGIN
    if isinstance(agent.model, OpenAIModel):
        params = agent.model.get_config().get("params", {})
        agent.model.update_config(params={**params, "max_tokens": state["max_tokens"]})
    else:
        agent.model.update_config(max_tokens=state["max_tokens"])
    last = agent.messages[-1]
    if last["role"] == "user":  # tool results: the prompt joins them, keeping roles alternating
        last["content"].append({"text": FORCED_ANSWER})
        return agent(None, limits={"turns": 1})
    return agent(FORCED_ANSWER, limits={"turns": 1})


def text_of(message: dict) -> str:
    return "".join(b["text"] for b in message.get("content") or [] if "text" in b)


@app.entrypoint
def invoke_agent(payload: dict):
    request = InvocationRequest(**payload)
    state = {}
    agent = Agent(
        model=make_model(request.config),
        tools=make_tools(state),
        hooks=[_Guard(state)],
        conversation_manager=NullConversationManager(),
        callback_handler=None,
    )

    result = agent(
        build_prompt(request.question, use_read_paper=True, use_web_search=False), limits={"turns": MAX_TURNS}
    )
    if state.pop("context_exceeded", False):
        stop = "context_exceeded"
    else:
        # content_filtered: the model refused (e.g. Claude on some epidemic-modeling requests): no answer, reward 0
        stop = {"limit_turns": "max_turns", "content_filtered": "refusal"}.get(result.stop_reason, "completed")
    forced = stop in ("max_turns", "context_exceeded") and "infra_error" not in state
    if forced:  # on the full history after max_turns; else, or if that does not fit, on the last history that fit
        if stop == "max_turns":
            result = force_answer(agent, state)
        if stop == "context_exceeded" or state.pop("context_exceeded", False):
            del agent.messages[state["fit_len"] :]
            result = force_answer(agent, state)
    if "infra_error" in state:
        raise RuntimeError(f"infrastructure error: {state['infra_error']}")
    response_text = text_of(result.message)

    reward = reward_fn(response_text=response_text, gold=request.gold, answerable=request.answerable)
    logger.info("stop=%s forced=%s reward=%s", stop, forced, reward)
    return {
        "reward": reward,
        "stop": stop,
        "forced": forced,
        "response": response_text,
        "turns": sum(m["role"] == "assistant" for m in agent.messages),
    }


if __name__ == "__main__":
    app.run()
