"""Calculator agent for testing ordinary Runtime HTTP responses."""

import re

from bedrock_agentcore.runtime import BedrockAgentCoreApp
from pydantic import BaseModel
from strands import Agent
from strands.models.openai import OpenAIModel
from strands_tools import calculator

app = BedrockAgentCoreApp()


class Task(BaseModel):
    prompt: str
    answer: str


@app.entrypoint
def invoke(payload):
    task = Task(**payload)
    config = payload["_config"]
    model = OpenAIModel(
        client_args={"api_key": config["api_key"], "base_url": config["base_url"]},
        model_id=config["model_id"],
    )
    agent = Agent(
        model=model,
        tools=[calculator],
        system_prompt=(
            "Your task is to solve the math problem. "
            "Use the calculator tool to compute all mathematical expressions. "
            'Let\'s think step by step and output the final answer after "####".'
        ),
    )
    response = agent(task.prompt)
    text = "".join(block["text"] for block in response.message.get("content", []) if "text" in block)
    answers = re.findall(r"#### (-?[0-9.,]+)", text[-300:])
    return {"reward": float(bool(answers) and answers[-1].replace(",", "") == task.answer)}


if __name__ == "__main__":
    app.run()
