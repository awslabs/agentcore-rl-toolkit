"""Seal-Hard: answer fact-seeking questions on which web search returns conflicting, noisy or unhelpful results.

SealQA publishes no agent harness, so the question is the whole prompt (no system prompt, as officially) and the
tools are a live-web `search` and `visit`. 40 turns, 8K output tokens per turn, 128K context; there is no forced
answer, so an episode that runs out of turns or context has no answer. The reward is the official grader's verdict
(reward.py). A failing search gateway, summarizer or rate-limited site fails the rollout instead of reaching the
model.
"""

import logging

from models import InvocationRequest
from reward import SealQAReward, grade
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
reward_fn = SealQAReward()

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

    verdict, judge_reply = grade(request.question, request.gold, response_text)
    reward = reward_fn(grade=verdict)
    logger.info("stop=%s grade=%s reward=%s", stop, verdict, reward)
    return {
        "reward": reward,
        "stop": stop,
        "grade": verdict,
        "judge": judge_reply,
        "response": response_text,
        "turns": sum(m["role"] == "assistant" for m in agent.messages),
    }


if __name__ == "__main__":
    app.run()
