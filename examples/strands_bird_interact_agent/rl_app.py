"""BIRD-Interact (a-Interact): resolve an ambiguous text-to-SQL request by exploring a PostgreSQL database and asking
a simulated user, then solve its follow-up.

Upstream's setting (BIRD-Interact@451fe2c, bird_interact_agent/): its system prompt adapted to tool calling
(prompts/), the actions as tools with their costs, and the budget (patience 6). The environment service (env/)
keeps the budget, the user simulator and the tests. 60 turns (upstream's official runs), 8K output tokens per turn,
128K context. A turn without a tool call costs 0.2 and is answered with an error, as upstream answers a reply
without an action. The episode ends when the service says the task is finished, or the turns or the context run
out. The reward is upstream's: 0.7 for phase 1, plus 0.3 for phase 2 (reward.py). An unavailable database or user
simulator fails the rollout instead of reaching the model.
"""

import logging
import uuid
from pathlib import Path

from botocore.exceptions import ClientError
from models import InvocationRequest
from reward import BirdInteractReward
from strands import Agent
from strands.agent.conversation_manager import NullConversationManager
from strands.hooks import BeforeModelCallEvent, HookProvider
from strands.models import BedrockModel
from strands.models.openai import OpenAIModel
from strands.types.exceptions import EventLoopException, MaxTokensReachedException
from tools import act, make_tools, post

from agentcore_rl_toolkit import AgentCoreRuntimeApp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = AgentCoreRuntimeApp()
reward_fn = BirdInteractReward()

SYSTEM_PROMPT = (Path(__file__).parent / "prompts/system.md").read_text().strip()
MAX_TURNS = 60
MAX_TOKENS = 8192
MAX_CONTEXT = 131072  # prompt + maximum output, the vLLM rule


class _Guard(HookProvider):
    """Stops the episode before a model call once the task is finished, the context would overflow, or after an
    infrastructure error."""

    def __init__(self, state: dict):
        self.state = state

    def register_hooks(self, registry, **_):
        registry.add_callback(BeforeModelCallEvent, self._before)

    def _before(self, event):
        if "infra_error" in self.state or self.state.get("done"):
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


def call(agent: Agent, prompt: str, turns: int) -> str:
    """agent(prompt) for up to `turns` turns -> its stop reason; a reply cut off at the output limit ends the turn as
    written, and a reply Bedrock blocks mid-stream is a refusal."""
    try:
        return agent(prompt, limits={"turns": turns}).stop_reason
    except MaxTokensReachedException:
        return "end_turn"
    except (ClientError, EventLoopException) as e:  # the Bedrock error, as such or wrapped by the event loop
        if "blocked by content filtering policy" not in str(e):
            raise
        return "content_filtered"


def episode(agent: Agent, state: dict, opening: str) -> str:
    """The agent until the task is finished -> finished | max_turns | context_exceeded | refusal."""
    prompt = opening
    while True:
        turns = MAX_TURNS - sum(m["role"] == "assistant" for m in agent.messages)
        reason = call(agent, prompt, turns) if turns > 0 else "limit_turns"
        if "infra_error" in state or state.get("done"):
            return "finished"
        if state.get("context_exceeded"):
            return "context_exceeded"
        if reason in ("limit_turns", "content_filtered"):
            return {"limit_turns": "max_turns", "content_filtered": "refusal"}[reason]
        prompt = act(state, None)  # a turn without an action


@app.entrypoint
def invoke_agent(payload: dict):
    request = InvocationRequest(**payload)
    state = {"session_id": uuid.uuid4().hex}
    started = post(state, "/start", {"session_id": state["session_id"], "instance_id": request.instance_id})
    if started is None:
        raise RuntimeError(f"infrastructure error: {state['infra_error']}")
    agent = Agent(
        model=make_model(request.config),
        tools=make_tools(state),
        system_prompt=SYSTEM_PROMPT,
        hooks=[_Guard(state)],
        conversation_manager=NullConversationManager(),
        callback_handler=None,
    )
    try:
        stop = episode(agent, state, started["observation"])
    finally:
        result = post(state, "/end", {"session_id": state["session_id"]})
    if "infra_error" in state:
        raise RuntimeError(f"infrastructure error: {state['infra_error']}")

    reward = reward_fn(**result)
    logger.info("stop=%s phase1=%s phase2=%s reward=%s", stop, result["phase1"], result["phase2"], reward)
    return {
        "reward": reward,
        "stop": stop,
        "phase1": result["phase1"],
        "phase2": result["phase2"],
        "turns": sum(m["role"] == "assistant" for m in agent.messages),
    }


if __name__ == "__main__":
    app.run()
