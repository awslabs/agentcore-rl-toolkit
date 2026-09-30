"""The Strands calculator agent using HTTP result retrieval."""

from models import InvocationRequest
from reward import GSM8KReward
from strands import Agent
from strands.models.openai import OpenAIModel
from strands_tools import calculator

from agentcore_rl_toolkit import AgentCoreRuntimeApp

app = AgentCoreRuntimeApp()
reward_fn = GSM8KReward()

system_prompt = (
    "Your task is to solve the math problem. "
    "Use the calculator tool to compute all mathematical expressions. "
    'Let\'s think step by step and output the final answer after "####".'
)


@app.entrypoint
def invoke_agent(payload: dict, context):
    request = InvocationRequest(**payload)
    config = context.config
    model = OpenAIModel(
        client_args={"api_key": config.get("api_key") or "EMPTY", "base_url": config["base_url"]},
        model_id=config["model_id"],
        params=config.get("sampling_params", {}),
    )
    agent = Agent(model=model, tools=[calculator], system_prompt=system_prompt)
    response = agent(request.prompt)
    response_text = "".join(block["text"] for block in response.message.get("content", []) if "text" in block)
    return {"reward": reward_fn(response_text=response_text, ground_truth=request.answer)}


if __name__ == "__main__":
    app.run()
