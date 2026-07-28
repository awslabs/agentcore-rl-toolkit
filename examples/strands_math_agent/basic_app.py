import logging

from bedrock_agentcore.runtime import BedrockAgentCoreApp
from dotenv import load_dotenv
from models import InvocationRequest
from strands import Agent
from strands.models import BedrockModel
from strands_tools import calculator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = BedrockAgentCoreApp()

load_dotenv()

model = BedrockModel(model_id="us.anthropic.claude-sonnet-4-20250514-v1:0")

agent = Agent(
    model=model,
    tools=[calculator],
    system_prompt=(
        "Your task is to solve the math problem. "
        + "Use calculator when applicable. "
        + 'Let\'s think step by step and output the final answer after "####".'
    ),
)


@app.entrypoint
def invoke_agent(payload):
    """
    Invoke the agent with a payload
    """
    # Validate the payload: `prompt: str` rejects non-string input (e.g. toolUse
    # content blocks) before it reaches the agent.
    request = InvocationRequest(**payload)
    user_input = request.prompt

    logger.info("User input: %s", user_input)

    response = agent(user_input)

    return response.message["content"][0]["text"]


if __name__ == "__main__":
    app.run()
