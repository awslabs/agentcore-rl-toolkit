"""HSCodeComp: classify an e-commerce product listing into its 10-digit US HTS code.

The official setup: the system prompt is the official one with the General Rules of Interpretation (prompts/,
translated from Chinese); the user message is the task's question; tools web_search, web_extract and the three
official HTS tools. 40 turns (the official default), 8K output tokens per turn, 128K context; there is no forced
answer, so an episode that runs out of turns or context has no answer. The reward is the official 10-digit match.
An unavailable search gateway fails the rollout instead of reaching the model.
"""

import logging
from pathlib import Path

from models import InvocationRequest
from reward import HSCodeReward, prefix_matches
from strands import Agent
from strands.agent.conversation_manager import NullConversationManager
from strands.hooks import BeforeModelCallEvent, HookProvider
from strands.models import BedrockModel
from strands.models.openai import OpenAIModel
from strands.types.exceptions import MaxTokensReachedException
from tools import make_tools

from agentcore_rl_toolkit import AgentCoreRuntimeApp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = AgentCoreRuntimeApp()
reward_fn = HSCodeReward()

PROMPTS = Path(__file__).parent / "prompts"
SYSTEM_PROMPT = (PROMPTS / "hscode_system.md").read_text() + "\n\n" + (PROMPTS / "gri_rules.md").read_text()
MAX_TURNS = 40
MAX_TOKENS = 8192
MAX_CONTEXT = 131072  # prompt + maximum output, the vLLM rule


class _Guard(HookProvider):
    """Stops the episode before a model call that would not fit the context, or after an infrastructure error."""

    def __init__(self, state: dict):
        self.state = state

    def register_hooks(self, registry, **_):
        registry.add_callback(BeforeModelCallEvent, self._before)

    def _before(self, event):
        if "infra_error" in self.state:
            event.cancel = True
        elif (event.projected_input_tokens or 0) + MAX_TOKENS > MAX_CONTEXT:
            self.state["context_exceeded"] = True
            event.cancel = True


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
    try:
        reason = agent(request.question, limits={"turns": MAX_TURNS}).stop_reason
    except MaxTokensReachedException:  # a reply cut off at the output limit is the answer, as written
        reason = "end_turn"
    if "infra_error" in state:
        raise RuntimeError(f"infrastructure error: {state['infra_error']}")
    if state.get("context_exceeded"):
        stop = "context_exceeded"
    else:
        # content_filtered: the model refused: no answer
        stop = {"limit_turns": "max_turns", "content_filtered": "refusal"}.get(reason, "completed")
    final = agent.messages[-1] if stop == "completed" and agent.messages[-1]["role"] == "assistant" else {}
    response_text = "".join(b["text"] for b in final.get("content") or [] if "text" in b)

    reward = reward_fn(response_text=response_text, gold=request.gold)
    logger.info("stop=%s reward=%s", stop, reward)
    return {
        "reward": reward,
        "stop": stop,
        "prefix_matches": prefix_matches(response_text, request.gold),
        "response": response_text,
        "turns": sum(m["role"] == "assistant" for m in agent.messages),
    }


if __name__ == "__main__":
    app.run()
