import logging

from models import InvocationRequest
from reward import GSM8KReward
from strands import Agent
from strands.agent.conversation_manager import NullConversationManager
from strands.models.openai import OpenAIModel
from strands_tools import calculator

from agentcore_rl_toolkit import AgentCoreRuntimeApp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = AgentCoreRuntimeApp()

system_prompt = (
    "Your task is to solve the math problem. "
    + "Use the calculator tool to compute all mathematical expressions. "
    + 'Let\'s think step by step and output the final answer after "####".'
)

reward_fn = GSM8KReward()


@app.entrypoint
def invoke_agent(payload: dict, context):
    """
    Invoke the math agent with a payload.

    Expected fields:
    - prompt: question from gsm8k
    - answer: ground truth (str)
    - _config: model config with base_url, model_id, and optional api_key / sampling_params

    Requests carrying the `_agentcore_runtime` envelope (sent by `AgentCoreHttpClient`)
    run in the foreground or background, and the returned dict is persisted for `get`.
    Requests without the envelope are served as ordinary HTTP invocations.
    """
    config = payload["_config"]
    # During training the rollout gateway keys trajectory capture off the api-key
    # slot, and the trainer supplies the session key in the payload. "EMPTY" (the
    # vLLM convention) for local runs and plain evaluation endpoints, which ignore it.
    api_key = config.get("api_key") or "EMPTY"

    model = OpenAIModel(
        client_args={"api_key": api_key, "base_url": config["base_url"]},
        model_id=config["model_id"],
        params=config.get("sampling_params", {}),
    )

    agent = Agent(
        model=model,
        tools=[calculator],
        system_prompt=system_prompt,
        conversation_manager=NullConversationManager(),
    )

    # Validate the payload: `prompt: str` rejects non-string input (e.g. toolUse
    # content blocks) before it reaches the agent.
    request = InvocationRequest(**payload)

    logger.info("User input: %s", request.prompt)

    response = agent(request.prompt)

    # Join the text blocks instead of indexing content[0]: a reasoning-parsing server
    # returns a leading reasoningContent block before the text block.
    content = response.message.get("content") or []
    response_text = "".join(block["text"] for block in content if "text" in block)

    return {"reward": float(reward_fn(response_text=response_text, ground_truth=request.answer))}


if __name__ == "__main__":
    app.run()
