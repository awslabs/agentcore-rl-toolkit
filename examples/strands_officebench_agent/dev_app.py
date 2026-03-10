import logging
import os
import time

os.environ["BYPASS_TOOL_CONSENT"] = "true"
os.environ["STRANDS_NON_INTERACTIVE"] = "true"

from dotenv import load_dotenv
from models import InvocationRequest
from reward import OfficeBenchReward
from strands import Agent
from strands.agent.conversation_manager import NullConversationManager
from strands.models import BedrockModel
from strands_tools import shell
from tools import ALL_TOOLS
from utils import load_task_from_s3, setup_testbed

from agentcore_rl_toolkit import AgentCoreRLApp
from agentcore_rl_toolkit.frameworks.strands.vllm_model import vLLMModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = AgentCoreRLApp()

load_dotenv()

SYSTEM_PROMPT = (
    "You are an AI office assistant for user {username}. "
    "Today is {date} ({weekday}). The current time is {time}.\n\n"
    "Help solve the office task using the available tools.\n"
    "Task files are in /testbed/data/. Calendar files are in /testbed/calendar/. "
    "Email files are in /testbed/emails/.\n\n"
    "Important:\n"
    "- Use absolute paths starting with /testbed/ for all file operations.\n"
    "- When creating new files, place them under /testbed/data/ unless specified otherwise.\n"
    "- When the task is complete, stop calling tools and summarize what you did."
)

reward_fn = OfficeBenchReward()


@app.rollout_entrypoint
def invoke_agent(payload: dict):
    rollout_config = payload.get("_rollout", {})

    # Choose model based on config
    if rollout_config.get("base_url"):
        model = vLLMModel(
            client_args={"api_key": "EMPTY", "base_url": rollout_config["base_url"]},
            model_id=rollout_config["model_id"],
            params=rollout_config.get("sampling_params", {}),
        )
    else:
        sampling = rollout_config.get("sampling_params", {})
        bedrock_kwargs = {
            "model_id": rollout_config.get("model_id", "us.anthropic.claude-sonnet-4-5-20250929-v1:0"),
            "temperature": sampling.get("temperature"),
            "top_p": sampling.get("top_p"),
            "max_tokens": sampling.get("max_tokens"),
        }
        thinking_budget = sampling.get("thinking_budget")
        if thinking_budget:
            bedrock_kwargs["additional_request_fields"] = {
                "thinking": {"type": "enabled", "budget_tokens": int(thinking_budget)}
            }
        model = BedrockModel(**bedrock_kwargs)

    request = InvocationRequest(**payload)

    # Load task config from S3
    start_time = time.time()
    task_config = load_task_from_s3(request.task_uri)
    logger.info(f"Loaded task config from {request.task_uri} (took {time.time() - start_time:.2f}s)")

    # Setup testbed directory
    start_time = time.time()
    setup_testbed(request.testbed_uri)
    logger.info(f"Testbed setup complete (took {time.time() - start_time:.2f}s)")

    # Build system prompt with task context
    system_prompt = SYSTEM_PROMPT.format(
        username=task_config.get("username", "User"),
        date=task_config.get("date", "unknown"),
        weekday=task_config.get("weekday", "unknown"),
        time=task_config.get("time", "unknown"),
    )

    agent = Agent(
        model=model,
        tools=[shell, *ALL_TOOLS],
        system_prompt=system_prompt,
        conversation_manager=NullConversationManager(),
    )

    # Run agent on the task
    user_input = task_config["task"]
    logger.info(f"Task: {user_input}")

    response = agent(user_input)
    logger.info(f'Agent response: {response.message["content"][0]["text"]}')

    # Collect token data (only available with vLLMModel)
    rollout_data = {}
    if hasattr(model, "get_token_data"):
        rollout_data = model.get_token_data()

    # Collect full conversation history
    messages = [
        {"role": msg.get("role", "unknown"), "content": msg.get("content", [])}
        for msg in agent.messages
    ]

    # Evaluate reward
    reward = reward_fn(
        testbed_dir="/testbed",
        evaluation_config=task_config["evaluation"],
    )
    logger.info(f"Reward: {reward}")

    return {
        "rollout_data": rollout_data,
        "messages": messages,
        "rewards": reward,
    }


if __name__ == "__main__":
    app.run()
