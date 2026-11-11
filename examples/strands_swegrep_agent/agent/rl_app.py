"""AgentCore Runtime entrypoint of the SWE-Grep agent.

The trainer (or eval/full_benchmark.py) invokes it with a task row plus ``_config``: the model's
``base_url``, ``model_id`` and the rollout's ``api_key``. The agent searches the task's checkout, answers
with code regions, and returns its reward, metrics, regions and assistant turns.
"""

import logging

import httpx
import openai
from openai import AsyncOpenAI
from rollout.agent_prompts import build_system_prompt
from rollout.custom_openai_model import CustomOpenAIModel
from rollout.repo_store import ensure_checkout
from rollout.repo_tools import build_repo_tools, top_level_listing
from rollout.turn_cap import TurnCapHook
from scoring.reward import compute_reward
from strands import Agent, ModelRetryStrategy
from strands.agent.conversation_manager import NullConversationManager

from agentcore_rl_toolkit import AgentCoreRuntimeApp

logger = logging.getLogger(__name__)
app = AgentCoreRuntimeApp()


class RetryTransientErrors(ModelRetryStrategy):
    """Strands retries rate limits; also retry dropped connections, server errors and vLLM's
    "Already borrowed" tokenizer race. Timeouts are not retried: each already took 5 minutes."""

    def is_retryable(self, exc: Exception) -> bool:
        if isinstance(exc, (openai.APITimeoutError, httpx.TimeoutException)):
            return False
        return (
            super().is_retryable(exc)
            or isinstance(exc, (openai.APIConnectionError, openai.InternalServerError, httpx.TransportError))
            or type(exc) is openai.APIError  # an error event inside a stream
            or (isinstance(exc, openai.BadRequestError) and "already borrowed" in str(exc).lower())
        )


@app.entrypoint
def invoke_agent(payload: dict, context=None) -> dict:
    config = payload["_config"]
    client = AsyncOpenAI(
        base_url=config["base_url"],
        api_key=config["api_key"],  # the rollout gateway attributes each model call to the rollout by it
        timeout=httpx.Timeout(300, connect=120),
        max_retries=0,  # RetryTransientErrors retries instead
    )
    result, _ = run_episode(payload, CustomOpenAIModel(client=client, model_id=config["model_id"]))
    return result


def run_episode(payload: dict, model) -> tuple[dict, Agent | None]:
    """Run one task with ``model`` as the policy; returns the runtime's result and the agent.
    eval/claude_bench.py calls this directly, with Claude as the model."""
    agent = None
    repo_root = ""
    crashed = None
    try:
        repo_root = str(ensure_checkout(payload["repo"], payload["base_commit"]))
        agent = Agent(
            model=model,
            tools=build_repo_tools(repo_root),
            system_prompt=build_system_prompt(payload["repo"], top_level_listing(repo_root)),
            hooks=[TurnCapHook()],
            retry_strategy=RetryTransientErrors(max_attempts=4, initial_delay=0.5, max_delay=4),
            conversation_manager=NullConversationManager(),  # never drop history: the task is in it
            callback_handler=None,
        )
        agent(payload["problem_statement"])
    except Exception as exc:  # noqa: BLE001
        logger.warning("rollout failed: %s", exc)
        crashed = exc

    turns = assistant_turns(agent.messages if agent else [])
    if crashed and not turns:
        # Nothing reached the model (no checkout, model unreachable): an infrastructure error, which
        # gets reward 0 rather than a penalty so it doesn't skew its group's advantages.
        return {"reward": 0.0, "error": repr(crashed)}, agent
    metrics, answer = compute_reward(agent.messages, payload["gold_lines"], repo_root, crashed=bool(crashed))
    regions = [[r.path, r.start_line, r.end_line] for r in answer.regions]
    return {"reward": metrics["reward"], "metrics": metrics, "regions": regions, "messages": turns}, agent


def assistant_turns(messages: list[dict]) -> list[dict]:
    """The assistant messages with only their text and tool calls (no reasoning), to keep results small."""
    turns = []
    for message in messages:
        if message.get("role") != "assistant":
            continue
        content = []
        for block in message.get("content") or []:
            if "text" in block:
                content.append({"text": block["text"]})
            elif "toolUse" in block:
                content.append({"toolUse": {"name": block["toolUse"]["name"], "input": block["toolUse"]["input"]}})
        turns.append({"role": "assistant", "content": content})
    return turns


if __name__ == "__main__":
    app.run()
